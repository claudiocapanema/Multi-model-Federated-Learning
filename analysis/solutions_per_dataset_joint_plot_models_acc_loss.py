import copy
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

def line(df, base_dir, x_column, first, second, hue, datasets, rounds_semi_covnergence, ci=None, style=None, hue_order=None):
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(4, 4))
    titles = ["Average accuracy", "Average loss"]
    y_columns = [first, second]
    y_maxs = [100, 7]
    y_lims = [True, True]
    print("aa: ", df)

    i = 0
    y_column = y_columns[i]
    y_max = y_maxs[i]
    y_lim = y_lims[i]

    print(base_dir + """solutions_{}_""".format(datasets))
    print("""x column: {} first: {} second: {} hue: {} order: {}""".format(x_column, first, second, hue, hue_order))
    line_plot(df=df, base_dir=base_dir, ax=axs[i],
              file_name="""solutions_{}_""".format(datasets), x_column=x_column, y_column='Avg. $acc_b$',
              hue=hue, ci=ci, title="Balanced accuracy", tipo=None, hue_order=hue_order, style="Model", y_lim=y_lim, y_max=y_max)
    axs[i].get_legend().remove()
    axs[i].set_xlabel("")
    axs[i].set_ylabel("$Acc_B$", labelpad=-4)

    fig.text(0.575, 0, 'Round (t)', ha='center', fontsize=12)

    i = 1
    y_column = y_columns[i]
    y_max = y_maxs[i]
    y_lim = y_lims[i]

    print(base_dir + """solutions_{}_""".format(datasets))
    print("""x column: {} first: {} second: {} hue: {} order: {}""".format(x_column, first, second, hue, hue_order))
    line_plot(df=df, base_dir=base_dir, ax=axs[i],
              file_name="""solutions_{}_""".format(datasets), x_column=x_column, y_column='Loss',
              hue=hue, ci=ci, title="Loss", tipo=None,style="Model",  hue_order=hue_order, y_lim=y_lim, y_max=y_max)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 3, 2, 1, 4, 5, 6, 7, 9, 8]
    # axs[i].legend([handles[i] for i in order], [labels[i] for i in order], fontsize=6.5, ncols=3)
    axs[i].set_xlabel("")
    axs[i].legend(fontsize=6.5)



    # specify order


    # for r in rounds_semi_convergence:
    #     for i in range(2):
    #         for j in range(2):
    #             axs[i].grid(False)
    #             axs[i].vlines(x=[int(r)], ymin=0, ymax=100, colors='silver', ls='--', lw=1,
    #                           label='vline_multiple - full height')
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.23, hspace=0.25)

    fig.savefig(
        """{}solutions_{}_{}_models_acc_loss.png""".format(base_dir, datasets,
                                                     num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_{}_models_acc_loss.svg""".format(base_dir, datasets,
                                                     num_clients), bbox_inches='tight',
        dpi=400)

    print(""""{}solutions_{}_{}_models_acc_loss.png""".format(base_dir, datasets,
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
    solutions = ["MultiFedSpeed@3", "MultiFedSpeed@2", "MultiFedSpeed@1", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    rounds_semi_convergence = [5, 9, 16, 30, 33, 39, 49, 55, 59]

    alphas = ['100.0', '100.0']
    models_names = ["gru", "cnn_a"]
    configuration = {"dataset": ["WISDM-W", "ImageNet"], "alpha": [float(i) for i in alphas]}
    solutions = ["MultiFedSpeed@3", "MultiFedSpeed@2", "MultiFedSpeed@1", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    rounds_semi_convergence = [5, 9, 26, 30, 35, 39, 43, 49]
    #
    alphas = ['100.0', '100.0']
    models_names = ["cnn_a", "cnn_a"]
    configuration = {"dataset": ["ImageNet", "ImageNet_v2"], "alpha": [float(i) for i in alphas]}
    solutions = ["MultiFedSpeed@3", "MultiFedSpeed@2", "MultiFedSpeed@1", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    rounds_semi_convergence = [22, 27, 33, 35, 38, 44, 45, 57]

    concept_drift = True
    alphas = ['0.1', '0.1']
    models_names = ["cnn_a", "gru"]
    configuration = {"dataset": ["ImageNet", "WISDM-W"], "alpha": [float(i) for i in alphas]}
    solutions = ["MultiFedAvg", "MultiFedBalance"]
    rounds_semi_convergence = [22, 27, 33, 35, 38, 44, 45, 57]
    alphas_end = ['0.1', '100.0']
    rounds_concept_drift = [28, 28]


    # models_names = ["cnn_a", "cnn_a"]
    datasets = configuration["dataset"]
    models_size = {"WISDM-P": 0.039024, "WISDM-W": 0.039024, "CIFAR10": 3.514152, "ImageNet": 3.524412, "ImageNet_v2": 3.524412}

    num_classes = {"EMNIST": 47, "CIFAR10": 10, "GTSRB": 43}
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

    # d = """results/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(num_clients, alphas, datasets, models_names, fc, rounds, epochs)
    d = """results/concept_drift_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
        concept_drift, num_clients, alphas, alphas_end[0], alphas_end[1], datasets, rounds_concept_drift[0],
        rounds_concept_drift[1],
        models_names, fc, rounds, epochs)
    read_solutions = []
    read_accs = []
    read_loss = []
    read_training_clients = []
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
            df = pd.read_csv("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            print("""{}{}_{}_test_0.csv""".format(d, dataset, solution))
            print(df["Accuracy"].to_numpy())
            print(df["# training clients"].to_numpy())
            acc += df["Balanced accuracy"].tolist()
            loss += df["Loss"].tolist()
            training_clients += df["# training clients"].tolist()
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

    print(df)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, models_names, rounds)

    line(df, base_dir, "Round (t)", first, second, "Solution", datasets, rounds_semi_covnergence=rounds_semi_convergence, hue_order=solutions, ci=None)
    plt.plot()
    # bar(df, base_dir, "Solution", third, second, x_order, hue_order)
    line(df, base_dir, "Round (t)", first, second, "Solution", datasets, rounds_semi_covnergence=rounds_semi_convergence, hue_order=solutions, ci=None)



