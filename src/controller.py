import os
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
            model.signal_obj.connect(self._log)
            model.create_dataset(self.view.rootPathText.text(),
                                 self.view.dataPathText.text(),
                                 self.view.smilesColumnName.text(),
                                 self.view.featurizerSpinBar.currentText())
            self.clear()

    def _check_process_input(self):
        if self.view.rootPathText.text() == '':
            self._log("Need to specify the processed data root.", "log")
            return False
        else:
            if not os.path.isdir(self.view.rootPathText.text()):
                self._log("The provided processed data root does not exist.", "log")
                return False
        if self.view.dataPathText.text() == '':
            self._log("Need to specify the raw data path.", "log")
            return False
        else:
            if not os.path.isfile(self.view.dataPathText.text()):
                self._log("The provided raw data file does not exist.", "log")
                return False
        if self.view.smilesColumnName.text() == '':
            self._log("Need to specify the SMILES column name.", "log")
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

class ModelTrainingController():

    def __init__(self, view):
        super(ModelTrainingController, self).__init__()
        self.view = view

    def browseFolder(self,text_line):
        fname = QFileDialog.getExistingDirectory(self.view, "Select folder")
        if text_line == "data-path":
            self.view.dataPathText.setText(fname)
        elif text_line == "valid-path":
            self.view.validDataPathText.setText(fname)
        elif text_line == "saving-path":
            self.view.savingPathText.setText(fname)
        else:
            raise ValueError

    def clear(self):
        self.view.dataPathText.clear()
        self.view.validDataPathText.clear()
        self.view.savingPathText.clear()
        self.view.numLayers.clear()
        self.view.embeddDim.clear()
        self.view.dropRatio.clear()
        self.view.learningRate.clear()
        self.view.decayRate.clear()
        self.view.numEpoch.clear()
        self.view.batchSize.clear()
        self.view.learningTask.setCurrentText('')

    def train(self):
        if self._check_process_input():
            model = TrainingModel()
            model.signal_obj.connect(self._log)
            model.train(num_layers=int(self.view.numLayers.text()), 
                        emb_dim=int(self.view.embeddDim.text()), 
                        conv=self.view.convType.currentText(), 
                        JK=self.view.jumpingKnowledge.currentText(), 
                        pooling=self.view.poolingType.currentText(), 
                        VN=eval(self.view.virtualNode.currentText()), 
                        drop_ratio=float(self.view.dropRatio.text()), 
                        residual=eval(self.view.residualConnection.currentText()),
                        data_path=self.view.dataPathText.text(), 
                        val_data_path=None if self.view.validDataPathText.text() == '' else self.view.validDataPathText.text(), 
                        save_model_path=None if self.view.savingPathText.text() == '' else self.view.savingPathText.text(), 
                        task=self.view.learningTask.currentText(), 
                        optimizer=self.view.optimizerType.currentText(), 
                        epoch=int(self.view.numEpoch.text()), 
                        lr=float(self.view.learningRate.text()), 
                        batch_size=int(self.view.batchSize.text()), 
                        decay=float(self.view.decayRate.text()))
            self.clear()

    def _check_process_input(self):
        if self.view.dataPathText.text() == '':
            self._log("Need to specify the processed data root.", "log")
            return False
        else:
            if not os.path.isdir(self.view.dataPathText.text()):
                self._log("The provided processed data root does not exist.", "log")
                return False
        if self.view.validDataPathText.text() != '':
            if not os.path.isdir(self.view.validDataPathText.text()):
                self._log("The provided validation data root does not exist.", "log")
                return False
        if self.view.savingPathText.text() != '':
            if not os.path.isdir(self.view.savingPathText.text()):
                self._log("The provided saving location does not exist.", "log")
                return False
        if self.view.learningTask.currentText() == '':
            self._log("Need to specify the learning task.", "log")
            return False
        if self.view.numLayers.text() != '':
            if not str.isdigit(self.view.numLayers.text()):
                self._log("Number of layers must be a positive integer.", "log")
                return False
        if self.view.embeddDim.text() != '':
            if not str.isdigit(self.view.embeddDim.text()):
                self._log("Number of Embedding dimensions must be a positive integer.", "log")
                return False
        if self.view.dropRatio.text() != '':
            if self._isFloat(self.view.dropRatio.text()):
                drop_ratio = float(self.view.dropRatio.text())
                if drop_ratio < 0 or drop_ratio > 1:
                    self._log("Drop-out ratio must be between 0 and 1.", "log")
                    return False
            else:
                self._log("Drop-out ratio must be a number between 0 and 1.", "log")
                return False
        if self.view.learningRate.text() != '':
            if self._isFloat(self.view.learningRate.text()):
                learning_rate = float(self.view.learningRate.text())
                if learning_rate <= 0:
                    self._log("Learning rate must be a positive number.", "log")
                    return False
            else:
                self._log("Learning rate must be a positive number.", "log")
                return False
        if self.view.decayRate.text() != '':
            if self._isFloat(self.view.decayRate.text()):
                decay = float(self.view.decayRate.text())
                if decay <= 0:
                    self._log("Decay rate must be a positive number.", "log")
                    return False
            else:
                self._log("Decay rate must be a positive number.", "log")
                return False
        if self.view.numEpoch.text() != '':
            if not str.isdigit(self.view.numEpoch.text()):
                self._log("Number of Epochs must be a positive integer.", "log")
                return False
        if self.view.batchSize.text() != '':
            if not str.isdigit(self.view.batchSize.text()):
                self._log("Batch size must be a positive integer.", "log")
                return False
        return True

    def _log(self, text, type):
        if type == "log":
            self.view.trainingLog.append(str(text))
        if type == "epoch-progress":
            self.view.epochProgressBar.setValue(ceil(float(text)*100))
        if type == "training-progress":
            self.view.trainingProgressBar.setValue(ceil(float(text)*100))
        if type == "warning":
            pass
        if type == "error":
            self.view.trainingLog.append(str(text))
            self.view.epochProgressBar.setValue(0)
            self.view.trainingProgressBar.setValue(0)

    def _isFloat(self,num):
        try:
            float(num)
            return True
        except ValueError:
            return False

