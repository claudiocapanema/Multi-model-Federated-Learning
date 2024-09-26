import copy
import numpy as np
import pandas as pd
from data_utils import read_data
import json

import copy

import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot, heatmap_plot
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
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    fig.savefig(
        """{}solutions_{}_clients_bar_per_dataset_efficiency.png""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_bar_per_dataset_efficiency.svg""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)

def line(dfs, base_dir, x_column, y_column, solutions, models):
    fig, axs = plt.subplots(2, 2, sharex='all', figsize=(8, 8))
    print("aa: ", df)

    i = 0

    for i in range(len(solutions)):
        solution = solutions[i]
        for j in range(len(models)):
            model = models[j]
            # print(dfs[i][j])
            # exit()
            heatmap_plot(df=dfs[solution][model], base_dir=base_dir, ax=axs[i,j], file_name="""solutions_{}_""".format(datasets), x_column=x_column, y_column=y_column, title="""{} ; {}""".format(solution, model))


    fig.savefig(
        """{}solutions_{}_clients_rounds_heatmap.png""".format(base_dir,
                                                     num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_rounds_heatmap.svg""".format(base_dir,
                                                     num_clients), bbox_inches='tight',
        dpi=400)

    print(""""{}solutions_{}_clients_rounds_heatmap.png""".format(base_dir,
                                                     num_clients))

def m(df, first, second, third):

    first_efficiency = df[first].mean() / df[third].sum()
    second_efficiency = df[second].mean() / df[third].sum()
    training_clients = df[third].sum()
    acc = df[first].mean()
    loss = df["Loss"].mean()
    print(first_efficiency, second_efficiency)
    return pd.DataFrame({"Efficiency": [first_efficiency], second + " efficiency": [second_efficiency], third: [training_clients], first: [acc], "Loss": [loss]})

if __name__ == "__main__":

    # alphas = ['0.1', '5.0']
    alphas = ['5.0', '1.0']
    models_names = ["gru", "cnn_a"]
    configuration = {"dataset": ["WISDM-P", "ImageNet"], "alpha": [float(i) for i in alphas]}
    models_names_formated = [{"gru": "GRU", "cnn_a": "CNN"}[model] for model in models_names]
    datasets = configuration["dataset"]
    # solutions = ["FedNome",  "MultiFedAvgRR", "FedFairMMFL", "MultiFedAvg", "MultiFedSpeedv1", "MultiFedSpeedv0", ]
    # solutions = ["MultiFedSpeed@1", "MultiFedSpeed@2", "MultiFedSpeed@3", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    solutions = ["MultiFedSpeed@3", "MultiFedAvg"]
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
    read_training_clients = []
    read_round = []
    read_datasets = []
    read_models = []
    read_rounds_training_clients = {solution: {{"gru": "GRU", "cnn_a": "CNN"}[model]: 0 for model in models_names} for solution in solutions}
    for solution in solutions:
        acc = []
        loss = []
        training_clients = []

        for dataset in datasets:
            rounds_training_clients = np.zeros((num_clients, rounds))
            df = pd.read_csv("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            print("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            print(df["Accuracy"].to_numpy())
            print(df["# training clients"].to_numpy())
            print(df.columns)
            acc += df["Balanced accuracy"].tolist()
            loss += df["Loss"].tolist()
            training_clients += df["# training clients"].tolist()
            read_round += df["Round"].tolist()
            read_datasets += [dataset] * len(df)
            i_model = {"WISDM-P": "GRU", "ImageNet": "CNN"}[dataset]
            read_models += [i_model] * len(df)
            df_rounds_training_client = df["training clients and models"].tolist()
            # Build heatmap matrix
            for k in range(len(df_rounds_training_client)):
                tra_clients = json.loads(df_rounds_training_client[k])
                round_ = df["Round"].tolist()[k]
                for client_ in tra_clients:
                    rounds_training_clients[client_][round_-1] = 1
            read_rounds_training_clients[solution][i_model] = copy.deepcopy(rounds_training_clients)

        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss
        read_training_clients += training_clients

    first = 'Avg. $acc_b$'
    second = '# training clients'
    dfs = {solution: {{"gru": "GRU", "cnn_a": "CNN"}[model]: 0 for model in models_names} for solution in solutions}
    for solution in solutions:
        for i_model in ["GRU", "CNN"]:
            df = pd.DataFrame(read_rounds_training_clients[solution][i_model], index=[i + 1 for i in range(num_clients)], columns=[i for i in range(rounds)])
            dfs[solution][i_model] = copy.deepcopy(df)


    print(dfs)
    # exit()

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, models_names, rounds)

    rounds_semi_convergence = [5, 13, 17, 34, 36, 44, 50, 54, 55]

    line(dfs, base_dir, x_column="Round", y_column="Client", solutions=solutions, models=models_names_formated)
    plt.plot()
    # bar(df, base_dir, "Solution", third, second, x_order, hue_order)
    line(dfs, base_dir, x_column="Round", y_column="Client", solutions=solutions, models=models_names_formated)



