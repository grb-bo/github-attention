import numpy as np
import os
import seaborn as sns
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pickle
from tasks.model_training import GATNet


def model_evaluation():

    #-----------------------------
    # Setup
    #-----------------------------

    print("\nEvaluating the model.\n")

    plotsdir = 'tasks/plots/model_evaluation'
    if not os.path.exists(plotsdir):
        os.makedirs(plotsdir)

    modeldir = 'tasks/model_parameters'
    model_file = "{}/best_model_params.pth".format(modeldir)
    assert os.path.exists(model_file), "File of model's parameters not found. Either: train the model via setting <model_training: true> in config.yalm, or make sure that the path of the model (<model_file>) is correct in model_evaluation.py"

    datafile = f"{modeldir}/data_and_masks.pkl"
    assert os.path.exists(datafile), "Data file not found. Either: set <model_training: true> in config.yalm and re-run main.py, or make sure that the path of the data (<datafile>) is correct in model_evaluation.py"
    with open(datafile, 'rb') as f:
        data, masks, masks_testing = pickle.load(f)


    #-----------------------------
    # Evaluation
    #-----------------------------

    # Loading the model with lowest validation loss
    best_model = GATNet()
    best_model.load_state_dict(torch.load(model_file))

    # Evaluating the best model & extracting the arrays for the metrics: true classes, predicted classes, probabilities of Class 1
    best_model.eval()
    with torch.no_grad():
        output = best_model(data)
        probs, pred = output.max(dim=1)

    y_true = data.y.cpu().detach().numpy()            # True classes
    y_pred = pred.cpu().detach().numpy()              # Predicted classes

    softmax_output = F.softmax(output, dim=1)  # Convert the log to probabilities
    out_probs = softmax_output[:, 1].cpu().detach().numpy() # Probabilities of Class 1
    #out_probs = output[:, 1].cpu().detach().numpy()   # Probabilities of Class 1 (when final activation function is not log)

    print("True classes - Predicted classes - Probabilities of Class 1:")
    print(y_true)
    print(y_pred)
    print(out_probs)
    print("")


    #-----------------------------
    # Performance metrics
    #-----------------------------

    # Accuracy
    correct = float((y_true[data.test_mask] == y_pred[data.test_mask]).sum())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.3f}'.format(acc))

    # Precision, recall, f1 score - weighted according to class
    precision = precision_score(y_true[data.test_mask], y_pred[data.test_mask], average='weighted')
    recall = recall_score(y_true[data.test_mask], y_pred[data.test_mask], average='weighted')
    f1 = f1_score(y_true[data.test_mask], y_pred[data.test_mask], average='weighted')

    print('Precision: {:.3f}'.format(precision))
    print('Recall: {:.3f}'.format(recall))
    print('F1 Score: {:.3f}'.format(f1))
    print("")

    #-----------------------------
    # Normalized Confusion Matrix
    #-----------------------------

    # Class weights
    weights = np.array([1.0, 2.871])

    for dataset_name, mask in masks_testing.items():
        cm = confusion_matrix(y_true[mask], y_pred[mask])
        cm = cm * weights[:, np.newaxis]

        # Normalize the weighted confusion matrices by class
        cm = cm / cm.sum(axis=1)[:, np.newaxis]

        # Plotting
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='.2f')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.savefig(f"{plotsdir}/Conf_matrix_{dataset_name}.pdf")
        plt.show()

    #-----------------------------
    # ROC curve and AUC
    #-----------------------------

    plt.figure()
    for dataset_name, mask in masks.items():
        fpr, tpr, _ = roc_curve(y_true[mask], out_probs[mask])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{dataset_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='midnightblue', lw=2, linestyle='--') # Reference line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{plotsdir}/ROC.pdf")
    plt.show()

    #-----------------------------
    # Histogram of Output Scores
    #-----------------------------

    # Separation of the output probabilities based on the ground truth labels
    out_probs_train_0 = out_probs[data.train_mask][y_true[data.train_mask] == 0]
    out_probs_train_1 = out_probs[data.train_mask][y_true[data.train_mask] == 1]
    out_probs_test_0 = out_probs[data.test_mask][y_true[data.test_mask] == 0]
    out_probs_test_1 = out_probs[data.test_mask][y_true[data.test_mask] == 1]

    # Plotting
    plt.figure()
    plt.hist(out_probs_test_0, bins=20, alpha=0.5, label='Class 0 (Testing)', weights=[1]*len(out_probs_test_0), density=True, color='red')
    plt.hist(out_probs_test_1, bins=20, alpha=0.5, label='Class 1 (Testing)', weights=[2.871]*len(out_probs_test_1), density=True, color='blue')
    plt.hist(out_probs_train_0, bins=20, alpha=0.5, label='Class 0 (Training)', weights=[1]*len(out_probs_train_0), density=True, color='darkred', histtype='step', linewidth=2, linestyle="--")
    plt.hist(out_probs_train_1, bins=20, alpha=0.5, label='Class 1 (Training)', weights=[2.871]*len(out_probs_train_1), density=True, color='midnightblue', histtype='step', linewidth=2, linestyle="--")
    plt.title('Histogram of Predictions')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Normalized Count')
    plt.legend(loc='upper center')
    plt.savefig(f"{plotsdir}/Classifier.pdf")
    plt.show()
    print(f"Plots have been saved in folder: {plotsdir}")
    print("")