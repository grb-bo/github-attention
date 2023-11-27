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
