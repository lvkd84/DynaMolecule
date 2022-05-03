import sys
from PyQt5.QtWidgets import *
from functools import partial

class MainWindow(QMainWindow):

    def __init__(self, scr_size):
        super(MainWindow,self).__init__()

        self.scr_size = scr_size

        self.setWindowTitle("DynaMolecule")
        self.setFixedSize(int(self.scr_size/2),int(self.scr_size/2))
        self.tabWidget = NavigationTabs(self)
        self.setCentralWidget(self.tabWidget)

        self.show()

class NavigationTabs(QTabWidget):

    def __init__(self, parent):
        super(NavigationTabs,self).__init__(parent)
        self.UI()

    def UI(self):
        self.homeTab = HomeTab()
        self.dataprepTab = DataPreparationTab()
        self.trainTab = ModelTrainingTab()
        self.evalTab = ModelEvaluationTab()

        self.addTab(self.homeTab, 'Home')
        self.addTab(self.dataprepTab, 'Data Preparation')
        self.addTab(self.trainTab, 'Model Training')
        self.addTab(self.evalTab, 'Model Evaluation')

class HomeTab(QWidget):

    def __init__(self):
        super(HomeTab,self).__init__()
        self.tabUI()

    def tabUI(self):
        self.text = QTextEdit()
        self.text.setPlainText("DynaMolecule")
        self.text.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.text)

        self.setLayout(layout)

class DataPreparationTab(QWidget):

    def __init__(self):
        super(DataPreparationTab,self).__init__()
        self.tabUI()

    def tabUI(self):
        ###############################
        # User Input Form
        ###############################
        self.formWidget = QWidget()
        # Get processed data root
        self._rootUI()
        # Get data path
        self._dataPathUI()
        # Form: smile column, featurizer, transform, pre-transform
        self._processingFormUI()
        # Submit Buttons
        self._submitUI()
        # Form layout
        self.formLayout = QVBoxLayout()
        self.formLayout.addWidget(self.rootBox)
        self.formLayout.addWidget(self.dataPathBox)
        self.formLayout.addWidget(self.processingFormBox)
        self.formLayout.addWidget(self.submitBox)
        self.formWidget.setLayout(self.formLayout)

        ###############################
        # Log
        ###############################
        self.processingLog = QTextEdit()
        self.processingLog.setReadOnly(True)

        ###############################
        # Main Layout
        ###############################
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.formWidget)
        self.mainLayout.addWidget(self.processingLog)

        self.setLayout(self.mainLayout)

    def _rootUI(self):
        self.rootBox = QGroupBox("Saving Location of the Processed Data", self.formWidget)
        self.rootPathText = QLineEdit(self.rootBox)
        self.rootBrowseButton = QPushButton("Browse", self.rootBox)
        self.rootBrowseButton.clicked.connect(partial(self._browseFile,text=self.rootPathText,title="Select saving location",folder=True))
        self.rootLayout = QHBoxLayout()
        self.rootLayout.addWidget(self.rootPathText)
        self.rootLayout.addWidget(self.rootBrowseButton)
        self.rootBox.setLayout(self.rootLayout)

    def _dataPathUI(self):
        self.dataPathBox = QGroupBox("Data File Path (.csv)", self.formWidget)
        self.dataPathText = QLineEdit(self.dataPathBox)
        self.dataBrowseButton = QPushButton("Browse", self.dataPathBox)
        self.dataBrowseButton.clicked.connect(partial(self._browseFile,text=self.dataPathText,title="Select data file",folder=False))
        self.dataPathLayout = QHBoxLayout()
        self.dataPathLayout.addWidget(self.dataPathText)
        self.dataPathLayout.addWidget(self.dataBrowseButton)
        self.dataPathBox.setLayout(self.dataPathLayout)

    def _processingFormUI(self):
        self.processingFormBox = QGroupBox("Data Processing Options", self.formWidget)
        self.smilesColumnName = QLineEdit(self.processingFormBox)
        self.featurizerSpinBar = QComboBox(self.processingFormBox)
        self.processingFormLayout = QFormLayout()
        self.processingFormLayout.addRow(QLabel("SMILES Column"),self.smilesColumnName)
        self.processingFormLayout.addRow(QLabel("Featurizer"),self.featurizerSpinBar)
        self.processingFormBox.setLayout(self.processingFormLayout)

    def _submitUI(self):
        self.submitBox = QGroupBox(self.formWidget)
        self.processButton = QPushButton("Process",self.submitBox)
        self.processButton.setStyleSheet("background-color : green")
        self.clearButton = QPushButton("Clear",self.submitBox)
        self.clearButton.setStyleSheet("background-color : red")
        self.submitLayout = QGridLayout()
        self.submitLayout.addWidget(self.processButton, 0, 1)
        self.submitLayout.addWidget(self.clearButton, 0, 2)
        self.submitBox.setLayout(self.submitLayout)

    def _browseFile(self,text,title,folder=False):
        if folder:
            fname = QFileDialog.getExistingDirectory(self, title)
            text.setText(fname)
        else:
            fname = QFileDialog.getOpenFileName(self, title)
            text.setText(fname[0])

    def _submit(self):
        pass

    def _clear(self):
        pass

class ModelTrainingTab(QWidget):

    def __init__(self):
        super(ModelTrainingTab,self).__init__()

        self.tabUI()

    def tabUI(self):
        pass

class ModelEvaluationTab(QWidget):

    def __init__(self):
        super(ModelEvaluationTab,self).__init__()

        self.tabUI()

    def tabUI(self):
        pass

def main():
   app = QApplication(sys.argv)

   scr_size = app.primaryScreen().size()
   window = MainWindow(scr_size.height())
   window.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()