import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from utils import get_optimizer, get_criterion
from ..data.dataset import MoleculeDataset
from ..data.featurizer import get_featurizer

class MoleculeGNN(torch.nn.Module):
    
    def __init__(self, num_layers, emb_dim, conv, JK, drop_ratio=0.0, residual=False):
        super(MoleculeGNN,self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.residual = residual
        self.JK = JK

        self.conv_layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.conv_layers.append(conv(emb_dim))

        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.layer_norms[layer](h_list[layer])
            h = self.gnns[layer](h, edge_index, edge_attr)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [torch.unsqueeze(h,dim=0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [torch.unsqueeze(h,dim=0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

class MoleculeGNNVN(torch.nn.Module):
    
    def __init__(self, num_layers, emb_dim, conv, JK, drop_ratio=0.0, residual=False):
        super(MoleculeGNNVN,self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.residual = residual
        self.JK = JK

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.conv_layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.conv_layers.append(conv(emb_dim))

        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.vn_mlps = torch.nn.ModuleList()
        for _ in range(self.num_layers-1):
            self.vn_mlps.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), 
                                                    torch.nn.BatchNorm1d(emb_dim), 
                                                    torch.nn.ReLU(), 
                                                    torch.nn.Linear(emb_dim, emb_dim)))

    def forward(self, x, edge_index, edge_attr, batch):

        vn_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        h_list = [x]
        for layer in range(self.num_layers):
            h_list[layer] = h_list[layer] + vn_embedding[batch]
            h = self.conv_layers[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            if layer != self.num_layers-1:
                vn_embedding_temp = global_add_pool(h_list[layer], batch) + vn_embedding
                if self.residual:
                    vn_embedding = vn_embedding + F.dropout(self.mlp_virtualnode_list[layer](vn_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    vn_embedding = F.dropout(self.mlp_virtualnode_list[layer](vn_embedding), self.drop_ratio, training = self.training)
        
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [torch.unsqueeze(h,dim=0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [torch.unsqueeze(h,dim=0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

class MoleculePredictiveNetwork(torch.nn.Module):

    def __init__(self, num_layers, emb_dim, conv, JK, pooling = "sum", VN=False, drop_ratio=0.0, residual=False):
        self.drop_ratio = drop_ratio

        self.emb_dim = emb_dim

        if VN:
            self.GNN = MoleculeGNN(num_layers, emb_dim, conv, JK, drop_ratio=0.0, residual=residual)
        else:
            self.GNN = MoleculeGNNVN(num_layers, emb_dim, conv, JK, drop_ratio=0.0, residual=residual)

        if pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # initialized during training
        self.graph_pred = None

        self.node_encoder = None

        self.bond_encoder = None

    def forward(self, batch):

        x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        node_embeddings = self.GNN(x=self.node_encoder(x),
                                   edge_index=edge_index,
                                   edge_attr=self.bond_encoder(edge_attr))

        graph_embeddings = self.pool(node_embeddings,batch)

        preds = self.graph_pred(graph_embeddings)

        return preds

class MoleculePredictor:

    def __init__(self, num_layers, emb_dim, conv, JK, pooling = "sum", VN=False, drop_ratio=0.0, residual=False):
        self.model = MoleculePredictiveNetwork(num_layers, emb_dim, conv, JK, pooling = "sum", VN=False, drop_ratio=0.0, residual=False)

    def train(self, data_path, val_data_path=None, save_model_path=None,
              task='regression', optimizer='adam',
              epoch=100, lr=0.001, batch_size=256, decay=1.0, device=None):

        self.batch_size = batch_size

        # TODO: train should takes in paths to data and create dataloaders accordingly
        train_data = MoleculeDataset(data_path)
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        if val_data_path:
            val_data = MoleculeDataset(val_data_path)
            valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        else:
            valloader = None

        _, num_tasks = dataloader[0].y.shape
        self.model.graph_pred = torch.nn.Sequential(
            torch.nn.Dropout(self.drop_ratio),
            torch.nn.Linear(self.emb_dim,num_tasks)
        )

        featurizer_name = train_data.featurizer_name
        self.model.node_encoder = get_featurizer(featurizer_name).get_atom_encoder()
        self.model.bond_encoder = get_featurizer(featurizer_name).get_bond_encoder()

        optimizer = get_optimizer(optimizer)(self.model.parameters(), lr=lr)

        criterion = get_criterion(task)()

        if device:
            device = torch.device(device)
        else:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        scheduler = StepLR(optimizer, step_size=1, gamma=decay)

        self.model.to(device)
        self.optimizer.to(device)
        self.criterion.to(device)

        min_so_far = 1000
        for _ in range(epoch):

            self.model.train()
            train_loss = 0
            for step, batch in enumerate(dataloader):      
                batch = batch.to(device)
                pred = self.model(batch)
                y = batch.y
                loss = criterion(pred.squeeze(),y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.cpu().item())
            train_loss = train_loss/step

            val_loss = None
            if valloader:
                self.model.eval()
                val_loss = 0
                for step, batch in enumerate(valloader):
                    batch = batch.to(device)
                    pred = self.model(batch)
                    y = batch.y
                    loss = self.criterion(pred.squeeze(),y)
                    val_loss += float(loss.cpu().item())
                val_loss = val_loss/step

            scheduler.step()

            # If a validation set is provide, then the model with the best valid score is saved
            # Else, the model is saved every training epoch
            if val_loss:
                if val_loss < min_so_far:
                    min_so_far = val_loss
                    # TODO: Save model
            else:
                # TODO: Save model
                pass

        # TODO: Save training log

    def evaluate(self, eval_data_path, save_result_path=None, device=None):
        if self.model.graph_pred == None:
            raise RuntimeError("Model has not been fitted.")

        if device:
            device = torch.device(device)
        else:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.model.to(device)
        self.criterion.to(device)

        train_data = MoleculeDataset(eval_data_path)
        dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

        predictions = []
        for step, data in enumerate(dataloader):
            batch = batch.to(device)
            predictions.append(self.model(data))
        predictions = torch.cat(predictions,dim=0)

        # TODO: save predictions

        return predictions

    def single_point_evaluate(self, data):
        if self.model.graph_pred == None:
            raise RuntimeError("Model has not been fitted.")

        # TODO: implement






        