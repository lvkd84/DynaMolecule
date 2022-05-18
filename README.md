**DynaMolecule**
================

> **DynaMolecule is a framework for machine learning on molecular data.**

Features
--------

*   Modern graph-based learning methods
*   Code-free interface
*   Automated data featurizing
*   Automated model training and evaluation

Data Preparation
----------------

The data preparation module converts data from an input file into training-ready data format.

The input data file is expected to be tabular with one column containing SMILES and the other columns containing numerical labels. For example:

| SMILES      | Weight | XlogP3      |
| ----------- | ----------- | ----------- |
| CC1(C2CCC1(C(=O)C2)C)C      | 152.23       | 2.2         |
| C(C(C1C(=C(C(=O)O1)O)O)O)O   | 176.12        | -1.6        |

Some situations that may lead to errors while preparing data:

*   Inappropriate format of the data file (non-tabular data)
*   Missing data
*   Non-numerical labels
*   Incorrect SMILES strings

The data file is expected to be cleaned before passing into the data preparation module.

Data preparation options:

*   **Saving Location of the Processed Data** (required): The folder where the training ready data files are stored.
*   **Data File Path** (required): The location where the unprocessed tabular data file is stored.
*   **SMILES Column** (required): The name of the column containing the SMILES strings in the unprocessed tabular data file.
*   **Featurizer** (optional, default: OGB Featurizer): The name of the featurizer to transform atom and bond properties to numerical vectors. The values of these vectors are tuned via training.

Model Training
--------------

Automated training of a Graph Neural Network model on processed molecular data.

Only one processed data location is required for training. Optionally, an additional validation data source can be provided. Note that the validation data's labels need to match those of the training data (think of them as different splits from the same data source) else an error will be thrown.

After training is done, if a saving location is provided, the trained model can be saved and can be retrieve later for evaluation or for further training. More specifically, if no validation data is provided, a snapshot of the model is saved every training epoch. If validation data is provided, the snapshot that achieves the best validation performance is saved.

Model training options:

*   **Location of the Processed Data** (required): The folder where the processed (training-ready) data is stored.
*   **Location of the Processed Validation Data** (optional): The folder where the processed validation data is stored.
*   **Saving Location of the Trained Model** (optional): The location where the best performing snapshot of the model is saved.
*   **Num Layers** (optional): Number of convolution layers in the Graph Neural Network, i.e, the depth of the model.
*   **Embedding Dimension** (optional): Number of embedding dimensions of each layer, i.e, the width of the model.
*   **Convolution** (optional): The type of convolution. This option define the main difference between different models.
*   **Jumping Knowledge** (optional): Techniques for combining embeddings from different layers to form the final embeddings of the nodes in the molecular graphs. By default, simply use the last layer's embeddings as the final embeddings.
*   **Pooling** (optional): Techniques for aggregating the embeddings of the nodes to form the embedding of the graph. By default, simply sum over the node embeddings.
*   **Virtual Node** (optional): Specifying whether a virtual node is added to the molecular graph. A virtual node is a global node that connects to every node in the graph which provides a global view to each node and helps information flow better. Empirically, models with virtual nodes often perform better than their vanilla version.
*   **Dropout Ratio** (optional): A number ranging from 0 to 1 that help regularize the model and control overfitting. A large dropout value greatly suppresses the model's complexity.
*   **Residual Connection** (optional): Specifying whether to use residual connections in the network. Residual connections help information flow in deep neural network.
*   **Learning Task (**required): Specifying the learning task so that an appropriate learning objective can be set up internally. Note that all the labels in a data source must be of the same learning task.
*   **Optimizer** (optional): Specifying the algorithm to optimize (learn) the model's parameters.
*   **Learning Rate** (optional): Updating rate of the model's parameters.
*   **Decay Rate** (optional): The decay rate of the learning rate over each epoch.
*   **Num Epochs** (optional): Number of learning epochs.
*   **Batch Size** (optional): The batch size used in batch training.

Model Evaluation
----------------
