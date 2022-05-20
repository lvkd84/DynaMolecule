import configparser
import argparse
from src.models.conv import *
from src.models.predictor import MoleculePredictor

CONV = {
    'GAT': GATConv,
    'GINE': GINEConv,
    'GCN': GCNConv
}

def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    data_path=config.get('PATH','ProcessedDataFilePath')
    val_data_path=config.get('PATH','ValidationDataFilePath')
    save_model_path=config.get('PATH','SavingPath')

    data_path=config.get('PATH','ProcessedDataFilePath')
    val_data_path=None if config.get('PATH','ValidationDataFilePath') == '' else config.get('PATH','ValidationDataFilePath'), 
    save_model_path=None if config.get('PATH','SavingPath') == '' else config.get('PATH','SavingPath'), 

    num_layers=int(config.get('OPTIONS','NumLayers'))
    emb_dim=int(config.get('OPTIONS','EmbeddingDimension'))
    conv=config.get('OPTIONS','Convolution')
    JK=config.get('OPTIONS','JumpingKnowledge')
    pooling=config.get('OPTIONS','Pooling')
    VN=eval(config.get('OPTIONS','VirtualNode'))
    drop_ratio=float(config.get('OPTIONS','DropoutRatio'))
    residual=eval(config.get('OPTIONS','ResidualConnection'))
    task=config.get('OPTIONS','LearningTask')
    optimizer=config.get('OPTIONS','Optimizer')
    epoch=int(config.get('OPTIONS','LearningRate'))
    lr=float(config.get('OPTIONS','DecayRate'))
    batch_size=int(config.get('OPTIONS','NumEpochs'))
    decay=float(config.get('OPTIONS','BatchSize'))

    conv_layer = CONV[conv]
    predictor = MoleculePredictor(num_layers, emb_dim, conv_layer, JK, pooling = pooling, VN=VN, 
                                    drop_ratio=drop_ratio, residual=residual)
    predictor.train(data_path=data_path, val_data_path=val_data_path, save_model_path=save_model_path,
                    task=task, optimizer=optimizer, epoch=epoch, lr=lr, batch_size=batch_size, decay=decay, signal_obj = None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    args = parser.parse_args()

    main(args.config_path)