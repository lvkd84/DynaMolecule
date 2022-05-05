from PyQt5.QtWidgets import QFileDialog
from model import *
from math import ceil

class DataPreparationController():

    def __init__(self, view):
        super(DataPreparationController, self).__init__()
        self.view = view

    def browseFolder(self):
        fname = QFileDialog.getExistingDirectory(self.view, "Select saving location")
        self.view.rootPathText.setText(fname)

    def browseFile(self):
        fname = QFileDialog.getOpenFileName(self.view, "Select data file")
        self.view.dataPathText.setText(fname[0])
    
    def clear(self):
        self.view.rootPathText.clear()
        self.view.dataPathText.clear()
        self.view.smilesColumnName.clear()
        self.view.featurizerSpinBar.setCurrentText('')

    def process(self):
        if self._check_process_input():
            model = DataPreparationModel()
            model.process_one_data_pt.connect(self._log)
            model.create_dataset(self.view.rootPathText.text(),
                                 self.view.dataPathText.text(),
                                 self.view.smilesColumnName.text(),
                                 self.view.featurizerSpinBar.currentText())
            self.clear()
        else:
            self._log("Need to specify processed data root, raw data path, and SMILES column name.")

    def _check_process_input(self):
        if self.view.rootPathText.text() == '':
            return False
        if self.view.dataPathText.text() == '':
            return False
        if self.view.smilesColumnName.text() == '':
            return False
        return True

    def _log(self, text, type):
        if type == "log":
            self.view.processingLog.append(str(text))
        if type == "progress":
            self.view.progressBar.setValue(ceil(float(text)*100))
        if type == "warning":
            pass
        if type == "error":
            self.view.processingLog.append(str(text))
            self.view.processingLog.append("Please make sure to delete the following folders:")
            self.view.processingLog.append("raw")
            self.view.processingLog.append("processed")
            self.view.processingLog.append("from the specified saving path at " + self.view.rootPathText.text() + " before retrying.")
            self.view.progressBar.setValue(0)