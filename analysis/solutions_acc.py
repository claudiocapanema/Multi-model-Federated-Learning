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
    bar_plot(df=df, base_dir=base_dir, palette=sns.color_palette(),
             file_name="""solutions_{}_acc""".format(datasets), x_column=x_column, y_column=first, y_lim=True,
             title="""""", tipo=None, y_max=100)
    # i = 0
    # axs[i].get_legend().remove()
    #
    # axs[i].set_xlabel('')
    # # axs[i].set_ylabel(first)
    # bar_plot(df=df, base_dir=base_dir, ax=axs[1],
    #          file_name="""solutions_{}""".format(datasets),
    #          x_column=x_column, y_column=second, title="""Average loss""", y_max=5, y_lim=True,
    #          tipo=None)
    # i = 1
    # axs[i].get_legend().remove()
    # axs[i].set_ylabel(second, labelpad=16)
    # axs[i].set_ylim(0, 5)
    # # axs[i].legend(fontsize=10)
    # # fig.suptitle("", fontsize=16)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.07, hspace=0.14)
    # fig.savefig(
    #     """{}solutions_{}_clients_bar.png""".format(base_dir,
    #                                                          num_clients), bbox_inches='tight',
    #     dpi=400)
    # fig.savefig(
    #     """{}solutions_{}_clients_bar.svg""".format(base_dir,
    #                                                          num_clients), bbox_inches='tight',
    #     dpi=400)

def line(df, base_dir, x_column, first, second, hue, ci=None):
    line_plot(df=df, base_dir=base_dir,
             file_name="""solutions_{}_acc""".format(datasets), x_column=x_column, y_column=first,
             hue=hue, ci=ci, title="""""", tipo=None, y_lim=True, y_max=100)
    # i = 0
    # # axs[i].get_legend().remove()
    # axs[i].legend(fontsize=7)
    #
    # axs[i].set_xlabel('')
    # line_plot(df=df, base_dir=base_dir, ax=axs[1],
    #          file_name="""solutions_{}""".format(datasets),
    #          x_column=x_column, y_column=second, title="""Average loss""", y_lim=True, y_max=5,
    #          hue=hue, ci=ci, tipo=None)
    # i = 1
    # axs[i].get_legend().remove()
    # axs[i].set_ylabel(second, labelpad=16)
    #
    #
    # # fig.suptitle("", fontsize=16)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.07, hspace=0.14)
    # fig.savefig(
    #     """{}solutions_{}_clients_line.png""".format(base_dir,
    #                                             num_clients), bbox_inches='tight',
    #     dpi=400)
    # fig.savefig(
    #     """{}solutions_{}_clients_line.svg""".format(base_dir,
    #                                             num_clients), bbox_inches='tight',
    #     dpi=400)

if __name__ == "__main__":

    alphas = ['0.1', '5.0']
    # alphas = ['5.0', '0.1']
    configuration = {"dataset": ["WISDM-W", "GTSRB"], "alpha": [float(i) for i in alphas]}
    models_names = ["gru", "cnn_a"]
    datasets = configuration["dataset"]
    # solutions = ["FedNome",  "MultiFedAvgRR", "FedFairMMFL", "MultiFedAvg"]
    solutions = ["FedFairMMFL",  "MultiFedAvgRR", "MultiFedAvg"]
    num_classes = {"EMNIST": 47, "CIFAR10": 10, "GTSRB": 43}
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
    read_round = []
    for solution in solutions:
        acc = []
        loss = []
        for dataset in datasets:
            df = pd.read_csv("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            acc += df["Accuracy"].tolist()
            loss += df["Loss"].tolist()
            read_round += df["Round"].tolist()
        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss

    first = 'Accuracy'
    second = 'Loss'
    df = pd.DataFrame({'Solution': read_solutions, first: np.array(read_accs) * 100, second: read_loss, "Round (t)": read_round})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})

    print(df)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, models_names, rounds)

    bar(df, base_dir, "Solution", first, second, x_order, hue_order)
    plt.plot()
    line(df, base_dir, "Round (t)", first, second, "Solution", None)

