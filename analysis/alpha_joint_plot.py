import numpy as np
import pandas as pd
from data_utils import read_data


if __name__ == "__main__":

    alphas = [0.1, 1.0]
    datasets = ["EMNIST", "Cifar10"]
    num_classes = {"EMNIST": 10, "Cifar10": 10}
    num_clients = 20

    read_alpha = []
    read_dataset = []
    read_id = []
    read_num_classes = []
    read_num_samples = []

    for alpha in alphas:
        for dataset in datasets:
            samples = []
            for id_ in range(num_clients):

                data = read_data(dataset, id_, num_clients, alpha, is_train=True)

                y_train = data['y'].astype(int)

                read_alpha.append(alpha)
                read_dataset.append(dataset)
                read_id.append(id_)
                read_num_classes.append(len(np.unique(y_train)) / num_classes[dataset])
                samples.append(len(y_train))
            samples = np.array(samples) / np.sum(samples)
            read_num_samples += list(samples)

    df = pd.DataFrame({'Alpha': read_alpha, 'Dataset': read_dataset, 'Id': read_id, 'Clients': num_clients, 'Classes': read_num_classes, 'Samples': read_num_samples})

    print(df)
