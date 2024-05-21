import numpy as np
import pandas as pd
from data_utils import read_data

import copy

import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


if __name__ == "__main__":

    alphas = [0.1, 1.0, 3.0, 5.0]
    datasets = ["Cifar10", "GTSRB"]
    num_classes = {"EMNIST": 10, "Cifar10": 10, "GTSRB": 43}
    num_clients = 20

    read_alpha = []
    read_dataset = []
    read_id = []
    read_num_classes = []
    read_num_samples = []

    read_std_alpha = []
    read_std_dataset = []
    read_num_samples_std = []

    for alpha in alphas:
        for dataset in datasets:
            samples = []
            for id_ in range(num_clients):

                data = read_data(dataset, id_, num_clients, alpha, is_train=True)

                y_train = data['y'].astype(int)

                read_alpha.append(alpha)
                read_dataset.append(dataset)
                read_id.append(id_)
                read_num_classes.append((100 * len(np.unique(y_train))) / num_classes[dataset])
                unique_count = {i: 0 for i in range(num_classes[dataset])}
                unique, count = np.unique(y_train, return_counts=True)
                balanced_samples = len(y_train) / num_classes[dataset]
                data_unique_count_dict = dict(zip(unique, count))
                for class_ in data_unique_count_dict:
                    unique_count[class_] = (100 * ((balanced_samples - data_unique_count_dict[class_])))/balanced_samples
                d = np.array(list(unique_count.values()))
                s = np.mean(d[d>0])
                samples.append(s)
            # samples = (100 * np.array(samples)) / np.sum(samples)
            read_num_samples += list(samples)
            read_std_alpha.append(alpha)
            read_std_dataset.append(dataset)
            read_num_samples_std.append(np.std(samples))

    second = 'Imbalance level'
    df = pd.DataFrame({'\u03B1': read_alpha, 'Dataset': read_dataset, 'Id': read_id, 'Clients': num_clients, 'Classes (%)': read_num_classes, second: read_num_samples})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})

    print(df)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/data_distribution/{}/{}/""".format(x_order, hue_order)

    fig, axs = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(6, 5))
    bar_plot(df=df, base_dir=base_dir, ax=axs[0], x_order=x_order,
             file_name="""unique_classes_{}""".format(datasets), x_column='\u03B1', y_column='Classes (%)',
             title="""Clients' local classes""", tipo="classes", y_max=10, hue='Dataset', hue_order=hue_order)
    i = 0
    axs[i].get_legend().remove()

    axs[i].set_xlabel('')
    bar_plot(df=df, base_dir=base_dir, ax=axs[1], x_order=x_order,
             file_name="""imbalance_level_{}""".format(datasets),
             x_column='\u03B1', y_column=second, title="""Clients's local samples""", y_max=100,
             hue='Dataset', tipo="classes", hue_order=hue_order)
    i = 1
    axs[i].get_legend().remove()
    axs[i].legend(fontsize=10)
    fig.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    fig.savefig(
        """{}distribution_{}_clients.png""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}distribution_{}_clients.svg""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
