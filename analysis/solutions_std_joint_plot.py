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


def bar(df, base_dir, x_column, first, second, x_order, hue_order):
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 5))
    bar_plot(df=df, base_dir=base_dir, ax=axs[0],
             file_name="""solutions_{}_std""".format(datasets), x_column=x_column, y_column=first, y_lim=True,
             title="""Average accuracy""", tipo=None, y_max=1)
    i = 0
    axs[i].get_legend().remove()

    axs[i].set_xlabel('')
    # axs[i].set_ylabel(first)
    bar_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}_std""".format(datasets),
             x_column=x_column, y_column=second, title="""Average loss""", y_max=1, y_lim=True,
             tipo=None)
    i = 1
    axs[i].get_legend().remove()
    axs[i].set_ylabel(second, labelpad=16)
    axs[i].set_ylim(0, 5)
    # axs[i].legend(fontsize=10)
    # fig.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    fig.savefig(
        """{}solutions_{}_clients_bar_std.png""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_bar_std.svg""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)

def line(df, base_dir, x_column, first, second, hue, ci=None):
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 5))
    line_plot(df=df, base_dir=base_dir, ax=axs[0],
             file_name="""solutions_{}_std""".format(datasets), x_column=x_column, y_column=first,
             hue=hue, ci=ci, title="""Average accuracy""", tipo=None, y_lim=True, y_max=1)
    i = 0
    axs[i].get_legend().remove()

    axs[i].set_xlabel('')
    line_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}_std""".format(datasets),
             x_column=x_column, y_column=second, title="""Average loss""", y_lim=True, y_max=0.03,
             hue=hue, ci=ci, tipo=None)
    i = 1
    # axs[i].get_legend().remove()
    axs[i].set_ylabel(second, labelpad=16)
    axs[i].legend(fontsize=6)

    # fig.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    fig.savefig(
        """{}solutions_{}_clients_line_std.png""".format(base_dir,
                                                num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_line_std.svg""".format(base_dir,
                                                num_clients), bbox_inches='tight',
        dpi=400)

if __name__ == "__main__":

    alphas = ['0.1', '5.0']
    # alphas = ['5.0', '0.1']
    configuration = {"dataset": ["Cifar10", "GTSRB"], "alpha": [5.0, 0.1]}
    datasets = configuration["dataset"]
    solutions = ["FedNome",  "MultiFedAvgRR", "FedFairMMFL", "MultiFedAvg"]
    num_classes = {"EMNIST": 47, "Cifar10": 10, "GTSRB": 43}
    num_clients = 40
    fc = 0.3
    rounds = 40
    epochs = 1

    read_alpha = []
    read_dataset = []
    read_id = []
    read_num_classes = []
    read_num_samples = []

    read_std_alpha = []
    read_std_dataset = []
    read_num_samples_std = []

    d = """results/clients_{}/alpha_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(num_clients, alphas, datasets, fc, rounds, epochs)
    read_solutions = []
    read_accs = []
    read_loss = []
    read_round = []
    first = 'Std Accuracy'
    second = 'Std loss'
    for solution in solutions:
        acc = []
        loss = []
        for dataset in datasets:
            print("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            df = pd.read_csv("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            acc += df[first].tolist()
            loss += df[second].tolist()
            read_round += df["Round"].tolist()
        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss


    df = pd.DataFrame({'Solution': read_solutions, first: np.array(read_accs) * 100, second: read_loss, "Round (t)": read_round})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})

    print(df)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, rounds)

    bar(df, base_dir, "Solution", first, second, x_order, hue_order)
    plt.plot()
    line(df, base_dir, "Round (t)", first, second, "Solution", None)

