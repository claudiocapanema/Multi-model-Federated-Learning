import copy
import json
import numpy as np
import pandas as pd

from base_plots import heatmap_plot
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
             file_name="""solutions_{}_per_dataset""".format(datasets), x_column=x_column, y_column=first, y_lim=True,
             title="""Average accuracy""", tipo=None, y_max=5)
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
    plt.subplots_adjust(wspace=0.07, hspace=0.12)
    fig.savefig(
        """{}solutions_{}_clients_bar_per_dataset_efficiency.png""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_bar_per_dataset_efficiency.svg""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)

def heatmap(df, base_dir, training_clients_models):

    rows = len(list(training_clients_models.keys()))
    fig, axs = plt.subplots(rows, 2, sharex='all', figsize=(6, 12))

    for j in range(2):
        for i in range(rows):


            solution = list(training_clients_models.keys())[i]
            dataset = list(training_clients_models[solution].keys())[j]

            heatmap_plot(df=training_clients_models[solution][dataset], base_dir=base_dir, ax=axs[j, i], title=solution, file_name="")

            print("""{} {}: {}""".format(solution, dataset, np.sum(training_clients_models[solution][dataset], axis=1)))

    # fig, axs = plt.subplots(2, sharex='all', figsize=(6, 12))
    #
    # for j in range(2):
    #
    #     solution = list(training_clients_models.keys())[0]
    #     dataset = list(training_clients_models[solution].keys())[j]
    #
    #     heatmap_plot(df=training_clients_models[solution][dataset], base_dir=base_dir, ax=axs[j], title=solution, file_name="")
    #
    #     print("""{} {}: {}""".format(solution, dataset, np.sum(training_clients_models[solution][dataset], axis=1)))




    plt.tight_layout()

    plt.subplots_adjust(wspace=0.23, hspace=0.25)

    fig.savefig(
        """{}solutions_{}_{}_training_clients_rounds.png""".format(base_dir, datasets,
                                                     num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_{}_training_clients_rounds.svg""".format(base_dir, datasets,
                                                     num_clients), bbox_inches='tight',
        dpi=400)

    print(""""{}solutions_{}_{}_training_clients_rounds.png""".format(base_dir, datasets,
                                                     num_clients))

def m(df, first, second, third):

    first_efficiency = df[first].mean() / df[third].sum()
    second_efficiency = df[second].mean() / df[third].sum()
    training_clients = df[third].sum()
    kb_transmitted = df["# training clients"].to_numpy() * df["Model size"].to_numpy()
    acc = df[first].mean()
    loss = df["Loss"].mean()
    print(first_efficiency, second_efficiency)
    return pd.DataFrame({"Efficiency": [first_efficiency], second + " efficiency": [second_efficiency], third: [training_clients], first: [acc], "Loss": [loss], "Communication cost (KB)": [kb_transmitted]})

if __name__ == "__main__":
    #
    rounds_semi_convergence = []

    # alphas = ['100.0', '100.0']
    # models_names = ["cnn_a", "cnn_a"]
    # configuration = {"dataset": ["CIFAR10", "ImageNet"], "alpha": [float(i) for i in alphas]}
    # solutions = ["MultiFedSpeed@3", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    # solutions = ["MultiFedSpeed@3", "MultiFedSpeed@2", "MultiFedSpeed@1", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]

    alphas = ['1.0', '100.0']
    models_names = ["gru", "cnn_a"]
    configuration = {"dataset": ["WISDM-W", "ImageNet"], "alpha": [float(i) for i in alphas]}
    # solutions = ["MultiFedSpeed@3", "MultiFedSpeed@2", "MultiFedSpeed@1", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    # rounds_semi_convergence = [5, 9, 16, 30, 33, 39, 49, 55, 59]

    # alphas = ['100.0', '100.0']
    # models_names = ["gru", "cnn_a"]
    # configuration = {"dataset": ["WISDM-W", "ImageNet"], "alpha": [float(i) for i in alphas]}
    # solutions = ["MultiFedSpeed@3", "MultiFedSpeed@2", "MultiFedSpeed@1", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    # rounds_semi_convergence = [5, 9, 26, 30, 35, 39, 43, 49]
    #
    # alphas = ['100.0', '100.0']
    # models_names = ["cnn_a", "cnn_a"]
    # configuration = {"dataset": ["ImageNet", "ImageNet_v2"], "alpha": [float(i) for i in alphas]}
    solutions = ["MultiFedSpeedRelative@3", "MultiFedSpeedDynamic@3"]
    rounds_semi_convergence = [22, 27, 33, 35, 38, 44, 45, 57]


    # models_names = ["cnn_a", "cnn_a"]
    datasets = configuration["dataset"]
    models_size = {"WISDM-P": 0.039024, "WISDM-W": 0.039024, "CIFAR10": 3.514152, "ImageNet": 3.524412, "ImageNet_v2": 3.524412}

    num_classes = {"EMNIST": 47, "CIFAR10": 10, "GTSRB": 43}
    num_clients = 40
    fc = 0.3
    rounds = 54
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
    read_training_clients_and_models = {}
    read_round = []
    read_datasets = []
    read_models = []
    read_sizes = []
    for solution in solutions:
        acc = []
        loss = []
        training_clients = []

        size = []
        for dataset in datasets:
            training_clients_and_models = np.zeros((num_clients, rounds))
            df = pd.read_csv("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            print("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            print(df["Accuracy"].to_numpy())
            print(df["# training clients"].to_numpy())
            acc += df["Balanced accuracy"].tolist()
            loss += df["Loss"].tolist()
            training_clients += df["# training clients"].tolist()
            training_clients_rounds = [json.loads(i) for i in df["training clients and models"].tolist()]
            for i in range(len(training_clients_rounds)):
                for j in range(len(training_clients_rounds[i])):
                    training_clients_and_models[training_clients_rounds[i][j]][i] = 1

            if solution not in read_training_clients_and_models:
                read_training_clients_and_models[solution] = {}
            read_training_clients_and_models[solution][dataset] = training_clients_and_models
            read_round += df["Round"].tolist()
            read_datasets += [dataset] * len(df)
            read_models += [{"EMNIST": "CNN-A", "CIFAR10": "CNN-B", "WISDM-P": "GRU", "WISDM-W": "GRU", "ImageNet": "CNN-A", "ImageNet_v2": "CNN-B"}[
                                dataset]] * len(df)
            model_size = models_size[dataset]
            size += [model_size] * len(df)
        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss
        read_training_clients += training_clients
        read_sizes += size

    # print("Heatmap para relative imagenet")
    # print(read_training_clients_and_models["MultiFedSpeedRelative@3"]["ImageNet"])
    # print("Heatmap para relative wisdm")
    # print(read_training_clients_and_models["MultiFedSpeedRelative@3"]["WISDM-W"])
    # exit()

    first = 'Avg. $acc_b$'
    second = '# training clients'
    df = pd.DataFrame({'Solution': read_solutions, first: np.array(read_accs) * 100, "Loss": read_loss, "Round (t)": read_round,
                       "Model": read_models, "Dataset": read_datasets, '# training clients': read_training_clients, "Model size": read_sizes})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})
    # df['Accuracy efficiency'] = df['Accuracy'] / df['# training clients']
    # df['Loss efficiency'] = df['Loss'] / df['# training clients']
    print(len(read_models), len(read_training_clients), len(read_sizes))
    df = df.groupby(["Solution", "Model", "Round (t)"]).apply(lambda e: m(e, first, second, '# training clients')).reset_index()
    third = first
    fourth = 'Loss efficiency'

    print(read_training_clients_and_models)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, models_names, rounds)

    heatmap(df, base_dir, read_training_clients_and_models)
    plt.plot()
    # # bar(df, base_dir, "Solution", third, second, x_order, hue_order)
    # line(df, base_dir, "Round (t)", first, second, "Solution", datasets, rounds_semi_covnergence=rounds_semi_convergence, hue_order=solutions, ci=None)



