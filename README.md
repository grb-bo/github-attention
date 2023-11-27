# **An Attention For GitHub**
_A Graph Attention Network for the GitHub Dataset_

| ![whole_graph](tasks/plots/whole_graph.jpg) | 
|:--:| 
| *The whole GitHub dataset, visualized as a graph in the Spring layout; nodes are colored accordingly to the community detected by the Louvain method.* |

## Table of Contents
- [Introduction](#introduction)
- [How to run](#run)
- [Code commentary](#commentary)
- [References](#references)

## Introduction
<a name="introduction"></a>
The **GitHub Social Network** is a large dataset of developers who have starred at least 10 repositories, describing the connections between them. The members are labelled as *web* or *machine learning* developers, and their relations in the dataset correspond to mutual following on the website.

The dataset was collected from the wesite public API by the by the MUSAE project in 2019 [1].

Given that the dataset has a network structure and given that the data are classified in a binary way, the GitHub dataset is particularly suited to be studied through a **graph neural network** with **attention mechanism**, one of the most promising areas of research in the field of machine learning.

In this project, a machine learning model will be trained to perform binary classification on the dataset. The project is organized in three tasks:
1. The analysis of the dataset through its graph representation.
2. The machine learning model and its training.
3. The evaluation of the model.

## How to run
<a name="run"></a>
The project has been implemented to run in two possible equivalent ways: as a Jupyter notebook and in python.

### Jupyter notebook
The code is entirely runnable as a notebook. Simply run _attention_for_github.ipynb_.

### Python
To run the project on python instead of as a notebook, you first need to install the necessary packages:
```
networkx
torch
torch-geometric
pyyaml
```

to get them, simply run the command  **`pip install -r requirements.txt`**.

Before running, if you wish to execute just a portion of the 3 tasks, select them in _config.yaml_ via the true/false setting. By default all 3 tasks are running.

To execute, run _main.py_.

All output figures will be saved in the folder _plots_.


![whole_dataset_detail](tasks/plots/whole_dataset_detail.jpg)


## Code commentary
<a name="commentary"></a>

### **Dataset Examination: Graph Properties Analysis**

The first step after data loading is a brief analysis of the dataset through its graph representation.


#### Basic properties

Having 37,700 nodes and 289,003 edges, both graph order and graph size can be considered large.

The distribution of node labels highlights an imbalance in the dataset: the majority of nodes (74.17%) are labelled by 0 (*_"web developers"_*) while the rest are labelled by 1 (_"machine learning developers"_). The classification task will therefore be **binary**. This difference in the proportion must be accounted for during the classifier training phase, via the implementation of class weights.


#### Connectedness and density

The observation of a single connected component implies a fully connected graph, where every node is reachable from every other node.

The density is the proportion of existing edges relative to the maximum possible number of edges the graph can have, ranging from 0 to 1. It can be seen as a way to infer how closely the nodes are connected in the network, at a global level.
For this graph, the density is of the order ~0.0001, so the arrangement of connections can be considered sparse.

#### Community detection

Graphs can present local areas of higher density even if globally they are sparsely connected.
If a (appropriately large) subgroup of nodes has a **relatively** higher density compared to the surroundings, it is called a _community_.
Community detection algorithms have the task to find subgraphs where the nodes are more connected to each other than to the neighborhood.

This can aid the visualization of a graph, when nodes belonging to the given community are colored in the same way.

Aleatory factors play an important role when dealing with a community-finding algorithm, so the number of *detected* communities is subject to statistical fluctuation. For the GitHub dataset, it is around ~**35-39**.

#### Local Cluster Coefficients
Cluster coefficients tell if the neighbours of a node are also connected to each other. It is the most local indicator of density: a lower cluster coefficients implies that the node is less likely to be in a (local neighbourhood) group.  

Both classes of this dataset exhibit a moderate clustering average, but it is evident that machine learning developers tend to cluster *less* than web developers.

#### Degrees
The degree of a node is the count of how many links it has to other nodes.

In the GitHub dataset, the number of nodes having a specific degree follow an exponentially decreasing trend, indicating a **scale free network**.

While most nodes in the network have a relatively limited number of edges, a smaller subset of nodes (usually called *hubs*) exhibit an important network of connections. Some of the developers in the GitHub dataset are therefore acting as hubs of connections.

#### Local Cluster Coefficients vs Degrees
The average neighbour degree distribution links two of their fundamental parameters: the cluster coefficients and the degrees.

Plotting it for the GitHub dataset highlights how, in this dataset, nodes with lower degree tend to have a higher clustering coefficient. Developers with many connections will connect to developers that are more likely to not interact with each other; the less connected nodes instead are more likely to connect to nodes that share edges between themselves (higher _clustering_ behaviour).

### **PyTorch Geometric**

After the dataset preprocessing & examination, machine learning starts. The library adopted by this project is PyTorch Geometric, a state-of-the-art framework for graph neural networks.

The graph has to be converted to a suitable type for the library: `torch_geometric.data` describing a homogeneous graph.

#### Features and Message Passing

In Pytorch Geometric, the  `data.x` are the input tensors of the neural network, i.e. the ***features*** of each node, while the `data.y` are the ground truth lables, i.e. the classes that the model will consider as ***prediction targets***.

In a GNN, the features are processed non-linearly via the **message passing** mechanism, which incorporate information from the edges (and/or) the adjacent nodes (more specifically: with a *k* number of layers, the *k*-adjacent nodes will be the nodes separated by a degree *k*).

#### A 3-Fold Splitting

In this code, the network hyperparameters are fine-tuned with respect to the *validation set*, while the final model's performance as a classifier is evaluated with respect to the *testing set*.

### **A Graph Attention Network**

While a general Graph Neural Network processes graph-structured data, a *Graph Attention Network* does so through the **attention mechanism**.

This model is composed of 3 attention-based layers with a varying numbers of attention heads, which are **indipendent** ways of aggregate information between nodes during the message passing. During training the layers learn to optimize how much "importance" each node gives to neighboring nodes, fine-tuning the attention weights. More heads means that the model is examining different aspects of the neighborhood at the same time. More layers means that the influence of more distant (_k_-adjacent) nodes is being examined.

As previously stated, the input of the model are the nodes features, which get processed through message passing.

Between the attention mechanism layers, non-linearity is introduced via Sigmoid Linear Unit (SiLU) activation function, and a dropout is applied to prevent overfitting.

The classifier output is finally produced with a (log) SoftMax function.

### **Training the GAT Model**

The Graph Attention Network is trained with PyTorch Geometric.

#### Setup

- **Epochs:** 60.
- **Class Weights:** necessary due to the class inbalance; a total of 27961 nodes are labelled by "0", and 9739 by "1", meaning a 2.871:1 ratio in the set.
- **3-Fold Approach**: the parameters that are chosen for the model are those that minimize the loss on the validation set; their performance will be evaluated on the testing set.
- **Optimizer:** Adam algorithm, with a learning rate of 0.01 and weight decay of 5e-4, is used for stochastic gradient descent.

#### Training Loop

1. **Forward Pass** predicts the class labels.
2. **Loss** for the training data is calculated with **negative log-likelihood** (due to the log-SoftMax), considering class weights to address imbalances.
3. **Backward Pass** computes gradients via backpropagation for the training set.
4. **Parameters Update** by Adam optimizer, based on the computed gradients.
5. **Validation** loss is calculated after the model is switched to evaluation mode.
6. **Logging** of training and validation losses.
7. **Saving of Parameters** of the best performing model with respect to the validation set.

### Results

The model has successfully learned to distinguish between the two classes, demonstrating a good ROC's AUC & scores, and no indications of overtraining.

## References
<a name="references"></a>
[1]: Benedek Rozemberczki, Carl Allen, Rik Sarkar. (2019). "Multi-scale Attributed Node Embedding." [arXiv:1909.13021](https://arxiv.org/abs/1909.13021). [GitHub Repository](https://github.com/benedekrozemberczki/MUSAE).