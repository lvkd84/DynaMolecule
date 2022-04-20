import os
import os.path as osp
import shutil
import gzip
import pandas as pd

from utils import smiles2graph
from featurizer import OGBFeaturizer
# from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_file_path, smile_column, root = 'dataset', featurizer = None, transform=None, pre_transform = None):

        self.original_root = root
        self.original_file_path = data_file_path
        self.smile_column = smile_column
        if featurizer == None:
            self.featurizer = OGBFeaturizer()
        else:
            self.featurizer = featurizer
        _, data_folder = osp.split(osp.splitext(data_file_path)[0])
        self.folder = osp.join(root, data_folder)

        super(MoleculeDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    # Instead of downloading, move user provided data to the raw directory
    def download(self):
        with open(self.original_file_path, 'rb') as f_in:
            with gzip.open(osp.join(self.raw_dir,self.raw_file_names), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))

        assert (self.smile_column in data_df.columns)

        tasks = [column for column in data_df.columns if column != self.smile_column]
        smiles_list = data_df[self.smile_column]
        task_list = data_df[tasks]

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in range(len(smiles_list)):
            
            data = Data()

            smiles = smiles_list[i]
            task_labels = task_list.iloc[i].values
            graph = smiles2graph(smiles,self.featurizer)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor(task_labels)

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
