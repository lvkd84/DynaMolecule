from PyQt5.QtCore import QObject, pyqtSignal
from data.dataset import MoleculeDataset

class DataPreparationModel(QObject):

    process_one_data_pt = pyqtSignal(str,str)

    def __init__(self):
        super(DataPreparationModel, self).__init__()

    def create_dataset(self, root, data_file_path = None, smile_column = None, featurizer = None):
        MoleculeDataset(root, data_file_path, smile_column, featurizer, signal_obj = self.process_one_data_pt)
        
