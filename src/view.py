from PyQt5.QtWidgets import *
from functools import partial
from controller import *

class MainWindow(QMainWindow):

    def __init__(self, scr_size):
        super(MainWindow,self).__init__()

        self.scr_size = scr_size

        self.setWindowTitle("DynaMolecule")
        # self.setFixedSize(int(self.scr_size/3*2),int(self.scr_size/3*2))
        self.setFixedSize(800,800)
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
        # Progress Bar
        ###############################
        self.progressBar = QProgressBar()

        ###############################
        # Main Layout
        ###############################
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.formWidget)
        self.mainLayout.addWidget(self.processingLog)
        self.mainLayout.addWidget(self.progressBar)

        self.setLayout(self.mainLayout)

        self.controller = DataPreparationController(self)

        self._hook_to_controller()

    def _rootUI(self):
        self.rootBox = QGroupBox("Saving Location of the Processed Data (*)", self.formWidget)
        self.rootPathText = QLineEdit(self.rootBox)
        self.rootBrowseButton = QPushButton("Browse", self.rootBox)
        self.rootLayout = QHBoxLayout()
        self.rootLayout.addWidget(self.rootPathText)
        self.rootLayout.addWidget(self.rootBrowseButton)
        self.rootBox.setLayout(self.rootLayout)

    def _dataPathUI(self):
        self.dataPathBox = QGroupBox("Data File Path (*)", self.formWidget)
        self.dataPathText = QLineEdit(self.dataPathBox)
        self.dataBrowseButton = QPushButton("Browse", self.dataPathBox)
        self.dataPathLayout = QHBoxLayout()
        self.dataPathLayout.addWidget(self.dataPathText)
        self.dataPathLayout.addWidget(self.dataBrowseButton)
        self.dataPathBox.setLayout(self.dataPathLayout)

    def _processingFormUI(self):
        self.processingFormBox = QGroupBox("Data Processing Options", self.formWidget)
        self.smilesColumnName = QLineEdit(self.processingFormBox)
        self.featurizerSpinBar = QComboBox(self.processingFormBox)
        self.featurizerSpinBar.addItems(['','OGB'])
        self.processingFormLayout = QFormLayout()
        self.processingFormLayout.addRow(QLabel("SMILES Column (*)"),self.smilesColumnName)
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

    def _hook_to_controller(self):
        self.rootBrowseButton.clicked.connect(self.controller.browseFolder)
        self.dataBrowseButton.clicked.connect(self.controller.browseFile)
        self.clearButton.clicked.connect(self.controller.clear)
        self.processButton.clicked.connect(self.controller.process)

class ModelTrainingTab(QWidget):

    def __init__(self):
        super(ModelTrainingTab,self).__init__()
        self.tabUI()

    def tabUI(self):
        ###############################
        # User Input Form
        ###############################
        self.formWidget = QWidget()
        # Get processed data location
        self._processedDataUI()
        # Get validation data location
        self._validDataUI()
        # Get the location to save the trained model
        self._modelSavingPathUI()
        # Form: 
        # Model options: num_layers, emb_dim, conv, JK, pooling, VN, drop_ratio, residual
        # Training options: optimizer, learning rate, decay, epoch, batch size
        self._trainingFormUI()
        # Submit Buttons
        self._submitUI()
        # Form layout
        self.formLayout = QVBoxLayout()
        self.formLayout.addWidget(self.dataPathBox)
        self.formLayout.addWidget(self.validDataPathBox)
        self.formLayout.addWidget(self.savingPathBox)
        self.formLayout.addWidget(self.trainingFormBox)
        self.formLayout.addWidget(self.submitBox)
        self.formWidget.setLayout(self.formLayout)

        ###############################
        # Log
        ###############################
        self.trainingLog = QTextEdit()
        self.trainingLog.setReadOnly(True)

        ###############################
        # Progress Bars
        ###############################
        self.epochProgressBar = QProgressBar()
        self.trainingProgressBar = QProgressBar()

        ###############################
        # Main Layout
        ###############################
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.formWidget)
        self.mainLayout.addWidget(self.trainingLog)
        self.mainLayout.addWidget(self.epochProgressBar)
        self.mainLayout.addWidget(self.trainingProgressBar)

        self.setLayout(self.mainLayout)

        self.controller = ModelTrainingController(self)

        self._hook_to_controller()

    def _processedDataUI(self):
        self.dataPathBox = QGroupBox("Location of the Processed Data (*)", self.formWidget)
        self.dataPathText = QLineEdit(self.dataPathBox)
        self.dataBrowseButton = QPushButton("Browse", self.dataPathBox)
        self.dataLayout = QHBoxLayout()
        self.dataLayout.addWidget(self.dataPathText)
        self.dataLayout.addWidget(self.dataBrowseButton)
        self.dataPathBox.setLayout(self.dataLayout)

    def _validDataUI(self):
        self.validDataPathBox = QGroupBox("Location of the Processed Validation Data (Optional)", self.formWidget)
        self.validDataPathText = QLineEdit(self.validDataPathBox)
        self.validDataBrowseButton = QPushButton("Browse", self.validDataPathBox)
        self.validDataLayout = QHBoxLayout()
        self.validDataLayout.addWidget(self.validDataPathText)
        self.validDataLayout.addWidget(self.validDataBrowseButton)
        self.validDataPathBox.setLayout(self.validDataLayout)

    def _modelSavingPathUI(self):
        self.savingPathBox = QGroupBox("Saving Location of the Trained Model (Optional)", self.formWidget)
        self.savingPathText = QLineEdit(self.savingPathBox)
        self.savingBrowseButton = QPushButton("Browse", self.savingPathBox)
        self.savingLayout = QHBoxLayout()
        self.savingLayout.addWidget(self.savingPathText)
        self.savingLayout.addWidget(self.savingBrowseButton)
        self.savingPathBox.setLayout(self.savingLayout)

    def _trainingFormUI(self):
        self.trainingFormBox = QGroupBox("Model and Training Options", self.formWidget)
        # num_layers, emb_dim, conv, JK, pooling, VN, drop_ratio, residual
        self.numLayers = QLineEdit(self.trainingFormBox)
        self.embeddDim = QLineEdit(self.trainingFormBox)
        self.convType = QComboBox(self.trainingFormBox)
        self.convType.addItems(['GINE','GCN','GAT'])
        self.jumpingKnowledge = QComboBox(self.trainingFormBox)
        self.jumpingKnowledge.addItems(['last','sum','max','concat'])
        self.poolingType = QComboBox(self.trainingFormBox)
        self.poolingType.addItems(['mean','sum','max'])
        self.virtualNode = QComboBox(self.trainingFormBox)
        self.virtualNode.addItems(['True','False'])
        self.dropRatio = QLineEdit(self.trainingFormBox)
        self.residualConnection = QComboBox(self.trainingFormBox)
        self.residualConnection.addItems(['True','False'])
        self.modelFormLayout = QFormLayout()
        self.modelFormLayout.addRow(QLabel("Num Layers"),self.numLayers)
        self.modelFormLayout.addRow(QLabel("Embedding Dimension"),self.embeddDim)
        self.modelFormLayout.addRow(QLabel("Convolution"),self.convType)
        self.modelFormLayout.addRow(QLabel("Jumping Knowledge"),self.jumpingKnowledge)
        self.modelFormLayout.addRow(QLabel("Pooling"),self.poolingType)
        self.modelFormLayout.addRow(QLabel("Virtual Node"),self.virtualNode)
        self.modelFormLayout.addRow(QLabel("Dropout Ratio"),self.dropRatio)
        self.modelFormLayout.addRow(QLabel("Residual Conenction"),self.residualConnection)
        # optimizer, learning rate, decay, epoch, batch size
        self.learningTask = QComboBox(self.trainingFormBox)
        self.learningTask.addItems(['','regression','classification','binary classification'])
        self.optimizerType = QComboBox(self.trainingFormBox)
        self.optimizerType.addItems(['Adam','AdamW','SGD'])
        self.learningRate = QLineEdit(self.trainingFormBox)
        self.decayRate = QLineEdit(self.trainingFormBox)
        self.numEpoch = QLineEdit(self.trainingFormBox)
        self.batchSize = QLineEdit(self.trainingFormBox)
        self.trainingFormLayout = QFormLayout()
        self.trainingFormLayout.addRow(QLabel("Learning Task (*)"),self.learningTask)
        self.trainingFormLayout.addRow(QLabel("Optimizer"),self.optimizerType)
        self.trainingFormLayout.addRow(QLabel("Learning Rate"),self.learningRate)
        self.trainingFormLayout.addRow(QLabel("Decay Rate"),self.decayRate)
        self.trainingFormLayout.addRow(QLabel("Num Epochs"),self.numEpoch)
        self.trainingFormLayout.addRow(QLabel("Batch Size"),self.batchSize)

        self.trainingOuterLayout = QHBoxLayout()
        self.trainingOuterLayout.addLayout(self.modelFormLayout)
        self.trainingOuterLayout.addLayout(self.trainingFormLayout)
        self.trainingFormBox.setLayout(self.trainingOuterLayout)

    def _submitUI(self):
        self.submitBox = QGroupBox(self.formWidget)
        self.processButton = QPushButton("Start Training",self.submitBox)
        self.processButton.setStyleSheet("background-color : green")
        self.clearButton = QPushButton("Clear",self.submitBox)
        self.clearButton.setStyleSheet("background-color : red")
        self.submitLayout = QGridLayout()
        self.submitLayout.addWidget(self.processButton, 0, 1)
        self.submitLayout.addWidget(self.clearButton, 0, 2)
        self.submitBox.setLayout(self.submitLayout)

    def _hook_to_controller(self):
        self.dataBrowseButton.clicked.connect(partial(self.controller.browseFolder,text_line='data-path'))
        self.validDataBrowseButton.clicked.connect(partial(self.controller.browseFolder,text_line='valid-path'))
        self.savingBrowseButton.clicked.connect(partial(self.controller.browseFolder,text_line='saving-path'))
        self.processButton.clicked.connect(self.controller.train)
        self.clearButton.clicked.connect(self.controller.clear)

class ModelEvaluationTab(QWidget):

    def __init__(self):
        super(ModelEvaluationTab,self).__init__()

        self.tabUI()

    def tabUI(self):
        pass