import os
import os.path as osp
import shutil
import gzip
import pandas as pd

from .utils import smiles2graph
from .featurizer import OGBFeaturizer, get_featurizer

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, data_file_path, smile_column, featurizer = None, signal_obj = None):

        self.data_file_path = data_file_path
        self.smile_column = smile_column
        if featurizer == None:
            self.featurizer_name = 'OGB'
            self.featurizer = OGBFeaturizer()
        else:
            self.featurizer_name = featurizer
            self.featurizer = get_featurizer(featurizer)()
        self.folder = root

        self.signal_obj = signal_obj

        super(MoleculeDataset, self).__init__(self.folder, transform = None, pre_transform = None)

        self.data, self.slices, self.featurizer_name = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    # Instead of downloading, move user provided data to the raw directory
    def download(self):
        with open(self.data_file_path, 'rb') as f_in:
            with gzip.open(osp.join(self.raw_dir,self.raw_file_names), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))

        if not (self.smile_column in data_df.columns):
            if self.signal_obj:
                self.signal_obj.emit('ERROR: The specified SMILES column name is not found in the data file.', 'error')
            raise ValueError("The specified SMILES column name is not found in the data file.")

        if data_df.isnull().values.any():
            if self.signal_obj:
                self.signal_obj.emit('ERROR: Missing values found in the data file', 'error')
            raise ValueError("Missing values found in the data file.")

        tasks = [column for column in data_df.columns if column != self.smile_column]
        smiles_list = data_df[self.smile_column]
        task_list = data_df[tasks]

        if self.signal_obj:
            self.signal_obj.emit('Converting SMILES strings into graphs...', 'log')
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
            data.y = torch.Tensor(task_labels)[None,:]

            data_list.append(data)

            if self.signal_obj:
                self.signal_obj.emit(str(i/len(smiles_list)),"progress")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        if self.signal_obj:
            self.signal_obj.emit('Saving...','log')

        torch.save((data, slices, self.featurizer_name), self.processed_paths[0])

        if self.signal_obj:
            self.signal_obj.emit('Done!','log')