import configparser
import argparse

from src.data.dataset import MoleculeDataset

def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    root = config.get('PATH','SavingPath')
    data_file_path = config.get('PATH','DataFilePath')

    smile_column = config.get('PROCESS','SmilesColumn')
    featurizer = config.get('PROCESS','Featurizer')

    MoleculeDataset(root, data_file_path=data_file_path, smile_column=smile_column, featurizer=featurizer, signal_obj = None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    args = parser.parse_args()

    main(args.config_path)