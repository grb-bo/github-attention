import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt



def get_clustering_coefficient(label_value, clustering_coefficients_dict, node_labels):
    """
    - Selects the clustering coefficients belonging to a given class

    Arguments:
    ----------
    label_value (int)
        The class for which to select the clustering coefficients.

    clustering_coefficients_dict (dict)
        A dictionary containing nodes as keys and corresponding clustering coefficients as values.

    node_labels (dict)
        Dictionary with nodes as keys and corresponding label as value    

    Returns:
    ----------
    clustering_coefficients_given_label (list)
        A list of clustering coefficients for nodes in the specified class.
    """
    label_nodes = [node for node, label in node_labels.items() if label == label_value]
    clustering_coefficients_given_label = [clustering_coefficients_dict[node] for node in label_nodes]
    return clustering_coefficients_given_label


def evaluate_clustering_coefficients(coefficient_labels, threshold=0.1, bins=20):
    """
    - For each label fed into the function, counts the nodes up to a given threshold of clustering coefficient; computes their percentage with respect to all the nodes belonging to the same label.
    - Plots the clustering coefficients of all the labels fed into the function.

    Arguments:
    ----------
    coefficient_labels (dict)
    A dictionary where each key is a label and the corresponding value is a list of clustering coefficients.

    threshold (float)
    Upper value of clustering coefficient to be computed for each label.

    bins (int)
    Total number of bins in the plotted histogram.

    Returns:
    ----------
    plt (matplotlib.pyplot.plot)
        Plot of the clustering coefficients of the dataset    
    """

    print("Proportion of nodes within {} clustering coefficient:".format(threshold))
    print("Class\tNodes\tPercentage wrt whole class")
    average_clustering_coefficient = {}
    j = 0
    for label, coefficients in coefficient_labels.items():
        hist, _ = np.histogram(coefficients, bins=bins)
        total_nodes = len(coefficients)
        nodes_in_range = sum(1 for coef in coefficients if coef <= threshold)
        percentage = (nodes_in_range / total_nodes) * 100
        print("{}\t{}\t{:.2f}%".format(label, nodes_in_range, percentage))
        histtype = "step" if j == 0 else "stepfilled"
        j=j+1
        plt.hist(coefficients, bins=bins, alpha=0.5, label='Class {}'.format(label), linewidth=3, histtype=histtype, density= True)
        average_clustering_coefficient[label] = np.mean(coefficients)
    print("")
    for label, average in average_clustering_coefficient.items():
        print(f"Total average clustering coefficient for Class {label}: {average:.4f}")
    print("")
    plt.title("Clustering Coefficients Histogram")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Normalized number of nodes")
    plt.legend()
    return plt



def dataset_analysis(graph, node_labels):

    plotsdir = 'tasks/plots/dataset_analysis'
    if not os.path.exists(plotsdir):
        os.makedirs(plotsdir)

    datadir = 'tasks/data'
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    #-----------------------------
    # Dataset examination
    #-----------------------------
    print("\nExamining the dataset:\n")

    # Graph statistics
    print("Number of nodes (graph order):", graph.number_of_nodes())
    print("Number of edges (graph volume):", graph.number_of_edges())

    # Distribution of node labels - necessary investigation for the training phase of the machine learning model
    unique_labels = set(node_labels.values())
    label_counts = {label: list(node_labels.values()).count(label) for label in unique_labels}  # Dictionary with label as key and number of nodes as value
    total_nodes = len(graph.nodes)
    label_ratios = {label: count / total_nodes for label, count in label_counts.items()} # Dictionary with label as key and ratio of nodes wrt total as value
    print("\nNode label distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: Count {count}, Ratio {label_ratios[label]:.4f}")

    # Weights calculator
    label_weights = {label: 1 / ratio for label, ratio in label_ratios.items()}
    min_weight = min(label_weights.values()) # Finds the minimum weight so that it can be set to 1
    normalized_label_weights = {label: weight / min_weight for label, weight in label_weights.items()}
    print("\nNode label normalized weights:")
    for label, weight in normalized_label_weights.items():
        print(f"Label {label}: Weight {weight:.4f}")

    # Creating a list of weights for the machine learning model
    weights_list = []
    for label in sorted(normalized_label_weights.keys()):
        weights_list.append(normalized_label_weights[label])

    # Connected components in the graph
    components = list(nx.connected_components(graph))
    print("\nNumber of Connected Components:", len(components))

    # Density
    density = nx.density(graph)
    print("\nGraph Density (scale 0-1): {:.4f}\n".format(density))


    #-----------------------------
    # Clustering coefficients and node degree distribution
    #-----------------------------

    # Extract clustering coefficients and degrees of the graph
    clustering_coefficients = nx.clustering(graph)  # less than a minute on Colab GPU
    clustering_coefficients_all = list(clustering_coefficients.values())
    degree_dict = dict(graph.degree())
    degrees_all = list(degree_dict.values())

    # Examine the clustering coefficients for each class
    label_coefficient_dict = {label: get_clustering_coefficient(label, clustering_coefficients, node_labels) for label in unique_labels}
    clustering_plot = evaluate_clustering_coefficients(label_coefficient_dict, 0.1)
    clustering_plot.savefig(f"{plotsdir}/Clustering.pdf")

    # Visualize the degree distribution to study the connectivity patterns
    plt.figure()
    plt.hist(degrees_all, bins=range(1, 50), alpha = 0.5)
    plt.title("Degree Histogram")
    plt.ylabel("Number of nodes")
    plt.xlabel("Degree")
    plt.savefig(f"{plotsdir}/Degree.pdf")
    plt.show()

    # Find the relations between clustering coefficients and degrees for the whole graph
    unique_degrees = np.unique(degrees_all)
    display_degrees = unique_degrees[unique_degrees >= 3]
    mean_coefficients = [np.mean([clustering_coefficients_all[i] for i in range(len(degrees_all)) if degrees_all[i] == deg]) for deg in unique_degrees]
    display_coefficients = [mean_coefficients[i] for i, degree in enumerate(unique_degrees) if degree in display_degrees]

    plt.figure()
    plt.scatter(degrees_all, clustering_coefficients_all, color="red", alpha=0.5, s=0.5)
    plt.plot(display_degrees, display_coefficients, color="blue", linestyle="-", linewidth=0.3, label='Average')
    plt.title("Average Neighbour Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Clustering Coefficient")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(left=1.8)
    plt.xlim(right=1100)
    plt.ylim(bottom=0.005)
    plt.ylim(top=1.05)
    plt.legend()
    plt.savefig(f"{plotsdir}/Degree_vs_clustering.pdf")
    plt.show()
    print(f"Plots have been saved in folder: {plotsdir}")