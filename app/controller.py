from PyQt5.QtWidgets import QFileDialog

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
        pass

    def process(self):
        pass