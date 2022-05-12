from PyQt5.QtCore import QObject, pyqtSignal
from data.dataset import MoleculeDataset
from models.predictor import MoleculePredictor
from models.conv import *

import torch

class DataPreparationModel(QObject):

    signal_obj = pyqtSignal(str,str)
    finished = pyqtSignal()

    def __init__(self):
        super(DataPreparationModel, self).__init__()

    def create_dataset(self, root, data_file_path = None, smile_column = None, featurizer = None):
        MoleculeDataset(root, data_file_path, smile_column, featurizer, signal_obj = self.signal_obj)
        self.finished.emit()

class TrainingModel(QObject):

    signal_obj = pyqtSignal(str,str)
    finished = pyqtSignal()

    CONV = {
        'GAT': GATConv,
        'GINE': GINEConv,
        'GCN': GCNConv
    }

    def __init__(self):
        super(TrainingModel, self).__init__()

    def train(self, num_layers, emb_dim, conv, JK, pooling, VN, drop_ratio, residual,
              data_path, val_data_path, save_model_path, task, optimizer, epoch, lr, batch_size, decay):

        conv_layer = self.CONV[conv]
        predictor = MoleculePredictor(num_layers, emb_dim, conv_layer, JK, pooling = pooling, VN=VN, 
                                      drop_ratio=drop_ratio, residual=residual)
        predictor.train(data_path=data_path, val_data_path=val_data_path, save_model_path=save_model_path,
                        task=task, optimizer=optimizer, epoch=epoch, lr=lr, batch_size=batch_size, decay=decay, signal_obj = self.signal_obj)
        self.finished.emit()

class EvaluatingModel(QObject):

    signal_obj = pyqtSignal(str,str)
    finished = pyqtSignal()

    def __init__(self):
        super(EvaluatingModel, self).__init__()

    def eval(self, model_path, data_path, labeled):
        predictor = torch.load(model_path)
        predictor.evaluate(data_path, labeled = labeled, signal_obj = self.signal_obj)
        self.finished.emit()
        
        
