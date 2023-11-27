#-----------------------------
# The model class
#-----------------------------

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(data.num_node_features, 32, heads=2)  # 2 attention heads
        self.conv2 = GATConv(32*2, 16, heads=2)                    # 2 attention heads
        self.conv3 = GATConv(16*2, 2, heads=1)                     # 1 attention head

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.silu(x)
        x = F.dropout(x, p=0.2)
        x = self.conv2(x, edge_index)
        x = F.silu(x)
        x = F.dropout(x, p=0.2)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
    

def model_training(graph, node_labels):

    plotsdir = 'plots/model_training'
    if not os.path.exists(plotsdir):
        os.makedirs(plotsdir)

    modeldir = 'model_parameters'
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    #-----------------------------
    # PyTorch Geometric tensors
    #-----------------------------

    data = from_networkx(graph)   
    data.y = torch.tensor([node_labels[node] for node in graph.nodes()])  # Tensor containing the ground truths labels

    # Input of the model will be the feature tensor
    data.x = torch.zeros(num_nodes, max_feature + 1)
    for node in range(num_nodes):
        data.x[node] = torch.tensor(encoded_features_dict[str(node)])

    #-----------------------------
    # 3-Fold data splitting
    #-----------------------------

    # Data shuffling & splitting into three sets (60% training, 20% validation, 20% test)
    indices = torch.arange(data.num_nodes) # Array of indices
    train_indices, temp_indices = train_test_split(indices, train_size=0.6, shuffle=True, random_state=30)
    val_indices, test_indices = train_test_split(temp_indices, train_size=0.5, shuffle=True, random_state=30) # Of the remaining indices, half is for validation and the other half is for testing

    # Create boolean masks (only the mask indices are set to True)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = 1
    data.val_mask[val_indices] = 1
    data.test_mask[test_indices] = 1

    # Dictionary to store the masks, useful for the metrics in the evaluation phase
    masks = {'Training': data.train_mask, 'Testing': data.test_mask, 'Validation': data.val_mask}
    masks_testing = {key: value for key, value in masks.items() if key != 'Validation'}


    #-----------------------------
    # Instantiating the GNN
    #-----------------------------

    model = GATNet()


    #-----------------------------
    # Training
    #-----------------------------

    n_epochs = 60

    # Weights: necessary for class imbalance
    weights = torch.tensor([1.0, 2.871])

    # Lists to store losses. 3-fold approach: the model with the best validation loss will be chosen, and afterwards will be evaluated on the testing set
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Gradient descent algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()   # Necessary in Pytorch as it accumulates the gradient

        # Forward pass to get predictions
        out = model(data)

        # Calculating train loss
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=weights) # nll_loss since softmax is log
        train_losses.append(train_loss.item())

        # Backpropagation & parameters update
        train_loss.backward()
        optimizer.step()

        # Calculating validation loss
        model.eval()
        with torch.no_grad():  # Necessary in Pytorch to disable gradient computation
            val_out = model(data)
            val_loss = F.nll_loss(val_out[data.val_mask], data.y[data.val_mask], weight=weights)
            val_losses.append(val_loss.item())

        # Logging of losses
        print(f"Epoch: {epoch + 1}/{n_epochs}, Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

        # Saving the model when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "{}/best_model_params.pth".format(modeldir))

        model.train()

    # Plotting the losses as a function of epochs
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss as a function of epochs')
    plt.legend()
    plt.show()
    plt.savefig(f"{plotsdir}/Losses.pdf")
    print(f"Best validation loss occoured at epoch: {best_epoch}")
    print(f"Training plot has been saved in folder: {plotsdir}")
    print(f"Best model has been saved in folder: {modeldir}")

    # Saving the files for model_evaluation
    with open('data/data_and_masks.pkl', 'wb') as f:
        pickle.dump((data, masks, masks_testing), f)