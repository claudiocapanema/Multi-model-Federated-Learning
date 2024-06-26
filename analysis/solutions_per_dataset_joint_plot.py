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
from numpy import trapz

def group_by(df, first, second):

    area_first = trapz(df[first].to_numpy(), dx=1)
    area_second = trapz(df[second].to_numpy(), dx=1)

    solution = df["Solution"].to_numpy()[0]

    return pd.DataFrame({"Solution": [solution], first + " AUC": [area_first], second + " AUC": [area_second]})

def bar_auc(df, base_dir, x_column, first, second, x_order, hue_order):

    df_2 = df.groupby("Solution").apply(lambda x: group_by(x, first, second))
    print(df_2)
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 5))
    bar_plot(df=df_2, base_dir=base_dir, ax=axs[0],
             file_name="""solutions_{}""".format(datasets), x_column=x_column, y_column=first + " AUC", y_lim=True,
             title="""Average accuracy""", tipo="auc", y_max=10000)
    i = 0
    axs[i].set_ylim(0, 10000)
    axs[i].get_legend().remove()


    axs[i].set_xlabel('')
    # axs[i].set_ylabel(first)
    bar_plot(df=df_2, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}""".format(datasets),
             x_column=x_column, y_column=second + " AUC", title="""Average loss""", y_max=10000, y_lim=True,
             tipo="auc")
    i = 1
    axs[i].set_ylim(0, 10000)
    axs[i].get_legend().remove()

    axs[i].set_ylabel(second + " AUC", labelpad=7)

    # axs[i].legend(fontsize=10)
    # fig.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    fig.savefig(
        """{}solutions_{}_clients_bar_per_dataset_auc.png""".format(base_dir,
                                                                num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_bar_per_dataset_auc.svg""".format(base_dir,
                                                                num_clients), bbox_inches='tight',
        dpi=400)

def bar(df, base_dir, x_column, first, second, x_order, hue_order):
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 5))
    bar_plot(df=df, base_dir=base_dir, ax=axs[0],
             file_name="""solutions_{}_per_dataset""".format(datasets), x_column=x_column, y_column=first, y_lim=True,
             title="""Average accuracy""", tipo=None, y_max=100)
    i = 0
    axs[i].get_legend().remove()

    axs[i].set_xlabel('')
    # axs[i].set_ylabel(first)
    bar_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}_per_dataset""".format(datasets),
             x_column=x_column, y_column=second, title="""Average loss""", y_max=5, y_lim=True,
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
        """{}solutions_{}_clients_bar_per_dataset.png""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_bar_per_dataset.svg""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)

def line(df, base_dir, x_column, first, second, hue, ci=None, style=None):
    titles = ["Average accuracy", "Average loss"]
    y_columns = [first, second]
    y_maxs = [100, 6]
    y_lims = [True, True]
    datasets = df['Dataset'].unique().tolist()

    for i in range(2):
        y_column = y_columns[i]
        y_max = y_maxs[i]
        y_lim = y_lims[i]
        fig, axs = plt.subplots(2,  sharex='all', figsize=(6, 5))
        for j in range(2):
            title = datasets[j]
            line_plot(df=df.query("""Dataset == '{}'""".format(title)), base_dir=base_dir, ax=axs[j],
                      file_name="""solutions_{}""".format(datasets), x_column=x_column, y_column=y_column,
                      hue=hue, ci=ci, title=title, tipo=None, y_lim=y_lim, y_min=0, y_max=y_max)
            if j != 1:
                axs[j].get_legend().remove()

            if j == 0:
                axs[j].set_xlabel('')

        # axs[i].get_legend().remove()
        # axs[1].set_ylabel(y_column, labelpad=16)
        axs[1].legend(fontsize=5)

    # fig.suptitle("", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.07, hspace=0.14)
        fig.savefig(
            """{}solutions_{}_clients_line_{}.png""".format(base_dir,
                                                    num_clients, y_column), bbox_inches='tight',
            dpi=400)
        fig.savefig(
            """{}solutions_{}_clients_line_{}.svg""".format(base_dir,
                                                    num_clients, y_column), bbox_inches='tight',
            dpi=400)

if __name__ == "__main__":

    alphas = ['0.1', '5.0']
    # alphas = ['5.0', '0.1']
    configuration = {"dataset": ["WISDM-W", "Tiny-ImageNet"], "alpha": [float(i) for i in alphas]}
    models_names = ["gru", "cnn_a"]
    datasets = configuration["dataset"]
    # solutions = ["FedNome",  "MultiFedAvgRR", "FedFairMMFL", "MultiFedAvg"]
    solutions = ["Proposta",  "MultiFedAvgRR", "FedFairMMFL", "MultiFedAvg"]
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

    d = """results/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(num_clients, alphas, datasets, models_names, fc, rounds, epochs)
    read_solutions = []
    read_accs = []
    read_loss = []
    read_round = []
    read_datasets = []
    for solution in solutions:
        acc = []
        loss = []
        for dataset in datasets:
            df = pd.read_csv("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            acc += df["Accuracy"].tolist()
            loss += df["Loss"].tolist()
            read_round += df["Round"].tolist()
            read_datasets += [dataset] * len(df)
        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss

    first = 'Balanced accuracy'
    second = 'Loss'
    df = pd.DataFrame({'Solution': read_solutions, first: np.array(read_accs) * 100, second: read_loss, "Round (t)": read_round,
                       "Dataset": read_datasets})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})

    print(df)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order,
                                                                                    models_names, rounds)

    bar(df, base_dir, "Solution", first, second, x_order, hue_order)
    plt.plot()
    bar_auc(df, base_dir, "Solution", first, second, x_order, hue_order)
    plt.plot()
    line(df, base_dir, "Round (t)", first, second, "Solution", ci=None, style="Dataset")

