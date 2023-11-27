import numpy as np
import json
import os

import networkx as nx
import seaborn as sns

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import pickle
import yaml

from tasks.preprocessing import preprocessing
from tasks.dataset_analysis import dataset_analysis
from tasks.model_training import model_training
from tasks.model_evaluation import model_evaluation


def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if config['dataset_analysis'] or config['model_training']:
        graph, node_labels = preprocessing()

    if config['dataset_analysis']:
        dataset_analysis(graph, node_labels)

    if config['model_training']:
        model_training(graph, node_labels)

    if config['model_evaluation']:
        model_evaluation()


if __name__ == "__main__":
    main()
