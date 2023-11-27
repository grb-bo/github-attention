def preprocessing():
    """
    - Creates a graph representing the GitHub dataset.

    Returns:
    ----------
    graph (networkx.Graph)
        Graph representing the GitHub dataset.

    node_labels (dict)
        Dictionary with nodes as keys and corresponding label as value
    """

    #-----------------------------
    # Data loading
    #-----------------------------
    
    print("Loading data and creating the graph.")

    # Load node labels
    labels_file = "data/git_targets.txt"
    nodes = []
    node_labels = {}  # Dictionary with nodes as keys and corresponding label as value
    with open(labels_file, "r") as file:
        next(file)    # Skip header
        for line in file:
            node, _, label = map(str, line.strip().split(","))
            node_labels[node] = int(label)
            nodes.append(node)

    # Load edges
    edges_file = "data/git_edges.txt"
    edges = []
    with open(edges_file, "r") as file:
        next(file)
        for line in file:
            node1, node2 = map(str, line.strip().split(","))
            edges.append((node1, node2))

    # Load features
    features_file = "data/git_features.json"
    features_dict = {}
    with open(features_file, "r") as file:
        features_dict = json.load(file)

    # One-hot encoding for the features
    num_nodes = len(features_dict)
    all_features_list=[]
    for node in range(num_nodes):
        all_features_list += features_dict[str(node)]

    max_feature = max(all_features_list)

    encoded_features_dict={}
    for node in range(num_nodes):
        node = str(node)
        encoded_features = np.array([0] * (max_feature + 1))
        node_features = features_dict[str(node)]
        encoded_features[node_features] = 1
        encoded_features_dict[node] = list(encoded_features)

    #-----------------------------
    # Graph creation
    #-----------------------------

    # The dataset can be represented by a homogeneous graph: a single type of node, a single type of edge.
    # Ingredients: nodes and connections between nodes. In addition, labels for the node classification task.
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, node_labels, 'label')
    return graph, node_labels