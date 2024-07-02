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
             file_name="""solutions_{}""".format(datasets), x_column=x_column, y_column=first, y_lim=True,
             title="""Average balanced accuracy""", tipo=None, y_max=10)
    i = 0
    axs[i].get_legend().remove()

    axs[i].set_xlabel('')
    axs[i].set_ylim(0, 10)
    # axs[i].set_ylabel(first)
    bar_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}""".format(datasets),
             x_column=x_column, y_column=second, title="""Average loss""", y_max=10, y_lim=True,
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
        """{}solutions_{}_clients_bar_efficiency.png""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_bar_efficiency.svg""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)

def line(df, base_dir, x_column, first, second, hue, ci=None):
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 5))
    line_plot(df=df, base_dir=base_dir, ax=axs[0],
             file_name="""solutions_{}""".format(datasets), x_column=x_column, y_column=first,
             hue=hue, ci=ci, title="""Average accuracy""", tipo=None, y_lim=True, y_max=40)
    i = 0
    axs[i].get_legend().remove()

    axs[i].set_xlabel('')
    line_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}""".format(datasets),
             x_column=x_column, y_column=second, title="""Average loss""", y_lim=True, y_max=1,
             hue=hue, ci=ci, tipo=None)
    i = 1
    # axs[i].get_legend().remove()
    axs[i].set_ylabel(second, labelpad=16)
    axs[i].legend(fontsize=6)

    print("""{}solutions_{}_clients_line_efficiency.png""".format(base_dir,
                                                num_clients))

    # fig.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    fig.savefig(
        """{}solutions_{}_clients_line_efficiency.png""".format(base_dir,
                                                num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_line_efficiency.svg""".format(base_dir,
                                                num_clients), bbox_inches='tight',
        dpi=400)

def m(df, first, second, third):

    first_efficiency = df[first].mean() / df[third].sum()
    second_efficiency = df[second].mean() / df[third].sum()
    print(first_efficiency, second_efficiency)
    return pd.DataFrame({first + " efficiency": [first_efficiency], second + " efficiency": [second_efficiency]})

if __name__ == "__main__":

    alphas = ['0.1', '5.0']
    # alphas = ['5.0', '0.1']
    configuration = {"dataset": ["WISDM-W", "ImageNet100"], "alpha": [float(i) for i in alphas]}
    models_names = ["gru", "cnn_a"]
    datasets = configuration["dataset"]
    # solutions = ["FedNome",  "MultiFedAvgRR", "FedFairMMFL", "MultiFedAvg", "Propostav1", "Propostav0", ]
    solutions = ["Proposta", "Propostav4", "MultiFedAvg", "FedFairMMFL"]
    num_classes = {"EMNIST": 47, "Cifar10": 10, "GTSRB": 43}
    num_clients = 40
    fc = 0.3
    rounds = 100
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
    read_training_clients = []
    read_round = []
    read_datasets = []
    for solution in solutions:
        acc = []
        loss = []
        training_clients = []
        for dataset in datasets:
            df = pd.read_csv("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            print("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            print(df["Accuracy"].to_numpy())
            print(df["# training clients"].to_numpy())
            acc += df["Balanced accuracy"].tolist()
            loss += df["Loss"].tolist()
            training_clients += df["# training clients"].tolist()
            read_round += df["Round"].tolist()
            read_datasets += [dataset] * len(df)
        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss
        read_training_clients += training_clients

    first = 'Balanced accuracy'
    second = 'Loss'
    df = pd.DataFrame(
        {'Solution': read_solutions, first: np.array(read_accs) * 100, second: read_loss, "Round (t)": read_round,
         "Dataset": read_datasets, '# training clients': read_training_clients})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})
    # df['Accuracy efficiency'] = df['Accuracy'] / df['# training clients']
    # df['Loss efficiency'] = df['Loss'] / df['# training clients']
    df = df.groupby(["Solution", "Round (t)"]).apply(lambda e: m(e, first, second, '# training clients')).reset_index()
    third = 'Balanced accuracy efficiency'
    fourth = 'Loss efficiency'

    print(df)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, models_names, rounds)

    bar(df, base_dir, "Solution", third, fourth, x_order, hue_order)
    plt.plot()
    line(df, base_dir, "Round (t)", third, fourth, "Solution", None)

