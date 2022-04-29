import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, softmax

#https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
class GCNConv(MessagePassing):
    def __init__(self, 
                 emb_dim=256):
        '''
            emb_dim (int): node/edge/model embedding dimensionality
        '''
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.root_emb.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_attr, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

#https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
class GINEConv(MessagePassing):
    def __init__(self, 
                 emb_dim=256,
                 train_ep=False):
        '''
            emb_dim (int): node/edge/model embedding dimensionality
        '''

        super(GINEConv, self).__init__(aggr = "add")

        self.train_ep = train_ep

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        if train_ep:
            self.eps = torch.nn.Parameter(torch.Tensor([0]))
        else:
            self.register_buffer('eps', torch.Tensor([0.]))

    def reset_parameters(self):
        self.mlp.reset_parameters()
        if self.train_ep:
            zeros(self.eps)

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps)*x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out

#https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gatv2_conv.html#GATv2Conv

class GATConv(MessagePassing):

    _alpha: None

    def __init__(
        self,
        emb_dim,
        heads = 1,
        negative_slope = 0.2,
        fill_value = 'mean',
    ):
        super().__init__(node_dim=0)

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.fill_value = fill_value

        self.lin_l = Linear(emb_dim, heads * emb_dim, bias=True,
                            weight_initializer='glorot')
        self.lin_r = Linear(emb_dim, heads * emb_dim,
                            bias=True, weight_initializer='glorot')

        self.att = torch.nn.Parameter(torch.Tensor(1, heads, emb_dim))

        self.lin_edge = Linear(emb_dim, heads * emb_dim, bias=False,
                                weight_initializer='glorot')

        self.bias = torch.nn.Parameter(torch.Tensor(heads * emb_dim))

        self._alpha = None

        self.reset_parameters()


    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x, edge_index, edge_attr):

        H, C = self.heads, self.emb_dim

        assert x.dim() == 2
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        # add self loops
        num_nodes = x_l.size(0)
        if x_r is not None:
            num_nodes = min(num_nodes, x_r.size(0))
        edge_index, edge_attr = remove_self_loops(
            edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=self.fill_value,
            num_nodes=num_nodes)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        self._alpha = None

        out = out.view(-1, self.heads * self.emb_dim)

        out += self.bias

        return out


    def message(self, x_j, x_i, edge_attr, index, ptr=None, size_i=None):
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        return x_j * alpha.unsqueeze(-1)