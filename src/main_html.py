html_str = """
<p>&nbsp;</p>
<h1 style="text-align: center;"><span style="color: #333399;"><strong>DynaMolecule</strong></span></h1>
<p>&nbsp;</p>
<blockquote>
<p style="text-align: center;"><strong>DynaMolecule is a framework for machine learning on molecular data.</strong></p>
</blockquote>
<p>&nbsp;</p>
<h2>Features</h2>
<ul>
<li>Modern graph-based learning methods</li>
<li>Code-free interface</li>
<li>Automated data featurizing</li>
<li>Automated model training and evaluation</li>
</ul>
<h2>Data Preparation</h2>
<p>The data preparation module converts data from an input file into training-ready data format.</p>
<p>The input data file is expected to be tabular with one column containing SMILES and the other columns containing numerical labels. For example:</p>
<table style="width: 70%; border-collapse: collapse; margin-left: auto; margin-right: auto;" border="1">
<tbody>
<tr>
<td style="width: 33.3333%; text-align: center;">SMILES</td>
<td style="width: 33.3333%; text-align: center;">Weight</td>
<td style="width: 33.3333%; text-align: center;">XLogP3</td>
</tr>
<tr>
<td style="width: 33.3333%; text-align: center;">CC1(C2CCC1(C(=O)C2)C)C</td>
<td style="width: 33.3333%; text-align: center;">152.23</td>
<td style="width: 33.3333%; text-align: center;">2.2</td>
</tr>
<tr>
<td style="width: 33.3333%; text-align: center;">C(C(C1C(=C(C(=O)O1)O)O)O)O</td>
<td style="width: 33.3333%; text-align: center;">176.12</td>
<td style="width: 33.3333%; text-align: center;">-1.6</td>
</tr>
</tbody>
</table>
<p>Some situations that may lead to errors while preparing data:</p>
<ul>
<li>Inappropriate format of the data file (non-tabular data)</li>
<li>Missing data</li>
<li>Non-numerical labels</li>
<li>Incorrect SMILES strings</li>
</ul>
<p>The data file is expected to be cleaned before passing into the data preparation module.</p>
<p>Data preparation options:</p>
<ul>
<li><strong>Saving Location of the Processed Data</strong> (required): The folder where the training ready data files are stored.</li>
<li><strong>Data File Path</strong> (required): The location where the unprocessed tabular data file is stored.</li>
<li><strong>SMILES Column</strong> (required): The name of the column containing the SMILES strings in the unprocessed tabular data file.</li>
<li><strong>Featurizer</strong> (optional, default: OGB Featurizer): The name of the featurizer to transform atom and bond properties to numerical vectors. The values of these vectors are tuned via training.</li>
</ul>
<h2>Model Training</h2>
<p>Automated training of a Graph Neural Network model on processed molecular data.</p>
<p>Only one processed data location is required for training. Optionally, an additional validation data source can be provided. Note that the validation data's labels need to match those of the training data (think of them as different splits from the same data source) else an error will be thrown.</p>
<p>After training is done, if a saving location is provided, the trained model can be saved and can be retrieve later for evaluation or for further training. More specifically, if no validation data is provided, a snapshot of the model is saved every training epoch. If validation data is provided, the snapshot that achieves the best validation performance is saved.</p>
<p>Model training options:</p>
<ul>
<li><strong>Location of the Processed Data</strong> (required): The folder where the processed (training-ready) data is stored.</li>
<li><strong>Location of the Processed Validation Data</strong> (optional): The folder where the processed validation data is stored.</li>
<li><strong>Saving Location of the Trained Model</strong> (optional): The location where the best performing snapshot of the model is saved.</li>
<li><strong>Num Layers</strong> (optional): Number of convolution layers in the Graph Neural Network, i.e, the depth of the model.</li>
<li><strong>Embedding Dimension</strong> (optional): Number of embedding dimensions of each layer, i.e, the width of the model.</li>
<li><strong>Convolution</strong> (optional): The type of convolution. This option define the main difference between different models.</li>
<li><strong>Jumping Knowledge</strong> (optional): Techniques for combining embeddings from different layers to form the final embeddings of the nodes in the molecular graphs. By default, simply use the last layer's embeddings as the final embeddings.</li>
<li><strong>Pooling</strong> (optional): Techniques for aggregating the embeddings of the nodes to form the embedding of the graph. By default, simply sum over the node embeddings.</li>
<li><strong>Virtual Node</strong> (optional): Specifying whether a virtual node is added to the molecular graph. A virtual node is a global node that connects to every node in the graph which provides a global view to each node and helps information flow better. Empirically, models with virtual nodes often perform better than their vanilla version.</li>
<li><strong>Dropout Ratio</strong> (optional): A number ranging from 0 to 1 that help regularize the model and control overfitting. A large dropout value greatly suppresses the model's complexity.</li>
<li><strong>Residual Connection</strong> (optional): Specifying whether to use residual connections in the network. Residual connections help information flow in deep neural network.</li>
<li><strong>Learning Task (</strong>required): Specifying the learning task so that an appropriate learning objective can be set up internally. Note that all the labels in a data source must be of the same learning task.</li>
<li><strong>Optimizer</strong> (optional): Specifying the algorithm to optimize (learn) the model's parameters.</li>
<li><strong>Learning Rate</strong> (optional): Updating rate of the model's parameters.</li>
<li><strong>Decay Rate</strong> (optional): The decay rate of the learning rate over each epoch.</li>
<li><strong>Num Epochs</strong> (optional): Number of learning epochs.</li>
<li><strong>Batch Size</strong> (optional): The batch size used in batch training.</li>
</ul>
<h2>Model Evaluation</h2>
"""