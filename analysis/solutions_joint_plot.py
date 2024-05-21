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

    alphas = ['0.1', '5.0']
    configuration = {"dataset": ["Cifar10", "EMNIST"], "alpha": [5.0, 0.1]}
    datasets = configuration["dataset"]
    solutions = ["FedNome", "FedAvg", "FedFairMMFL"]
    num_classes = {"EMNIST": 10, "Cifar10": 10, "GTSRB": 43}
    num_clients = 20
    fc = 0.3
    rounds = 10
    epochs = 1

    read_alpha = []
    read_dataset = []
    read_id = []
    read_num_classes = []
    read_num_samples = []

    read_std_alpha = []
    read_std_dataset = []
    read_num_samples_std = []

    d = """results/clients_{}/alpha_{}/fc_{}/rounds_{}/epochs_{}/""".format(num_clients, alphas, fc, rounds, epochs)
    read_solutions = []
    read_accs = []
    read_loss = []
    for solution in solutions:
        acc = []
        loss = []
        for dataset in datasets:
            df = pd.read_csv("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            acc += df["Accuracy"].tolist()
            loss += df["Loss"].tolist()
        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss

    first = 'Accuracy'
    second = 'Loss'
    df = pd.DataFrame({'Solution': read_solutions, first: np.array(read_accs) * 100, second: read_loss})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})

    print(df)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/{}/{}/""".format(x_order, hue_order)

    fig, axs = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(6, 5))
    bar_plot(df=df, base_dir=base_dir, ax=axs[0],
             file_name="""solutions_{}""".format(datasets), x_column='Solution', y_column=first,
             title="""Clients' local classes""", tipo="e", y_max=100)
    i = 0
    axs[i].get_legend().remove()

    axs[i].set_xlabel('')
    bar_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}""".format(datasets),
             x_column='Solution', y_column=second, title="""Clients's local samples""", y_max=5,
             tipo="e")
    i = 1
    axs[i].get_legend().remove()
    axs[i].legend(fontsize=10)
    fig.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    fig.savefig(
        """{}solutions_{}_clients.png""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients.svg""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
