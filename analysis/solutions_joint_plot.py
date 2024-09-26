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
from scipy.integrate import simpson
from numpy import trapz

def ci(data, f=False):

    mi, ma =  st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))

    mean_value = (mi + ma)/2
    diff = abs(ma) - abs(mean_value)
    mean_value = round(mean_value, 2)
    if f:
        print("loss ci: ", mi, ma)
        print("loss media: ", np.mean(data))
        print("calc: ", mean_value)
        print(data)
    if np.isnan(mean_value):
        mean_value = np.mean(data)
        return """{}""".format(int(mean_value))
    diff = round(diff, 2)
    return """{} $\pm$ {}""".format(mean_value, diff)

def group_by(df, first, second, third):

    area_first = trapz(df[first].to_numpy(), dx=1)
    area_second = trapz(df[second].to_numpy(), dx=1)
    area_first_efficiency = trapz(df.groupby("Round (t)").apply(lambda e: pd.DataFrame({"eff": [e[first].mean() / e["# training clients"].sum()]}))["eff"].to_numpy(), dx=1)
    area_third = trapz(df["# training clients"].to_numpy(), dx=1)
    df_2 = df[df["Round (t)"] == 100]
    acc_efficiency = df_2.groupby("Round (t)").apply(lambda e: pd.DataFrame({"eff": [e[first].mean() / e["# training clients"].sum()]})).reset_index()
    acc_efficiency = acc_efficiency["eff"].tolist()[0]
    # acc = ci(df_2[first].to_numpy())
    acc = df_2[first].mean()
    # training_clients = ci(df_2["# training clients"].to_numpy(), True)
    training_clients = df["# training clients"].mean()
    # loss = ci(df_2[second].to_numpy())
    loss = df_2[second].mean()

    kb_transmitted = (df["# training clients"].to_numpy() * df["Model size"].to_numpy()).sum()
    kb_transmitted_auc = trapz(df.groupby("Round (t)").apply(lambda e: pd.DataFrame({"kb": [(e["# training clients"].to_numpy() * e["Model size"].to_numpy()).sum()]}))["kb"].to_numpy(), dx=1)

    solution = df["Solution"].to_numpy()[0]
    if solution == "FedFairMMFL":
        print(df[first], first, df["# training clients"])

    return pd.DataFrame({"Solution": [solution], "Efficiency": [acc_efficiency], "Balanced accuracy": [acc], "# training clients": [training_clients], "Loss": [loss], "Efficiency AUC": area_first_efficiency, first + " AUC": [area_first], second + " AUC": [area_second], "# training clients AUC": [area_third], "Communication cost (KB)": [kb_transmitted], "Communication cost (KB) AUC": [kb_transmitted_auc]})

def aggregate_metrics(df):

    columns = list(df.columns)
    balanced_acc_variance = df["Balanced accuracy"].std()
    loss_variance = df["Loss"].std()
    aggregated_df = df.groupby(columns).mean().reset_index()
    aggregated_df["Balanced accuracy std"] = balanced_acc_variance
    aggregated_df["Loss std"] = loss_variance

    return aggregated_df[["Balanced accuracy", "Loss", "Balanced accuracy std", "Loss std"]]

def bar_auc(df, base_dir, x_column, first, second, third, x_order, hue_order):

    df_2 = df.groupby(["Solution", "Dataset", "Round (t)", "Model"]).apply(lambda x: x.mean()).reset_index()
    df_2 = df_2.groupby(["Solution"]).apply(lambda x: group_by(x, first, second, third)).round(
        5).reset_index(1)[["Efficiency AUC", "Balanced accuracy AUC", "# training clients AUC", "Loss AUC", "Communication cost (KB) AUC"]].round(2)
    # print(df.drop(index=["Dataset", "Solution"]))
    # exit()
    # df_2 = df.reset_index().groupby("Solution").apply(lambda x: group_by(x, first, second)).round(2).reset_index(1)[
    #     ["Balanced accuracy AUC", "Loss AUC"]].round(2)
    print(df_2)
    # exit()
    # fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 5))
    #
    # i = 0
    # axs[i].set_ylim(0, 12000)
    # axs[i].get_legend().remove()
    # bar_plot(df=df_2, base_dir=base_dir, ax=axs[0],
    #          file_name="""solutions_{}""".format(datasets), x_column=x_column, y_column=first + " AUC", y_lim=True,
    #          x_order=x_order, title="""Balanced accuracy""", tipo="auc", y_max=12000)
    #
    # axs[i].set_xlabel('')
    # # axs[i].set_ylabel(first)
    # bar_plot(df=df_2, base_dir=base_dir, ax=axs[1],
    #          file_name="""solutions_{}""".format(datasets),
    #          x_column=x_column, y_column=second + " AUC", title="""Average loss""", y_max=3000, y_lim=True,
    #          x_order=x_order, tipo="auc")
    # i = 1
    # axs[i].set_ylim(0, 3000)
    # axs[i].get_legend().remove()
    #
    # axs[i].set_ylabel(second + " AUC", labelpad=5)
    #
    # # axs[i].legend(fontsize=10)
    # # fig.suptitle("", fontsize=16)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.07, hspace=0.14)
    # fig.savefig(
    #     """{}solutions_{}_clients_bar_auc.png""".format(base_dir,
    #                                                          num_clients), bbox_inches='tight',
    #     dpi=400)
    # fig.savefig(
    #     """{}solutions_{}_clients_bar_auc.svg""".format(base_dir,
    #                                                          num_clients), bbox_inches='tight',
    #     dpi=400)

    df_2.to_latex("""{}auc.latex""".format(base_dir))
    print("""{}auc.latex""".format(base_dir))

def bar_performance(df, base_dir, x_column, first, second, third, x_order, hue_order):

    print(df)
    # df_2 = df[["Solution", "Round (t)", "Balanced accuracy", "# training clients", "Loss"]]
    df_2 = df.groupby(["Solution", "Dataset", "Round (t)", "Model"]).apply(lambda x: x.mean()).reset_index()
    df_2 = df_2.groupby(["Solution"]).apply(lambda x: group_by(x, first, second, third)).round(
        5).reset_index(1).round(2)[["Efficiency", "Balanced accuracy", "# training clients", "Loss", "Communication cost (KB)"]]
    print(df_2)
    # exit()


    df_2.to_latex("""{}global_metrics.latex""".format(base_dir))
    print("""{}auc.latex""".format(base_dir))

def bar_metric(df, base_dir, x_column, first, second, x_order, hue_order):
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 5))
    bar_plot(df=df, base_dir=base_dir, ax=axs[0],
             file_name="""solutions_{}""".format(datasets), x_column=x_column, y_column=first, y_lim=True,
             title="""Average balanced accuracy""", tipo=None, y_max=100)
    i = 0
    axs[i].get_legend().remove()

    axs[i].set_xlabel('')
    # axs[i].set_ylabel(first)
    bar_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}""".format(datasets),
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
        """{}solutions_{}_clients_bar.png""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_bar.svg""".format(base_dir,
                                                             num_clients), bbox_inches='tight',
        dpi=400)

def line(df, base_dir, x_column, first, second, hue, ci=None):
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 5))
    line_plot(df=df, base_dir=base_dir, ax=axs[0],
             file_name="""solutions_{}""".format(datasets), x_column=x_column, y_column=first,
             hue=hue, ci=ci, title="""Average balanced accuracy""", tipo=None, y_lim=True, y_max=100)
    i = 0
    # axs[i].get_legend().remove()
    axs[i].legend(fontsize=7)

    axs[i].set_xlabel('')
    line_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}""".format(datasets),
             x_column=x_column, y_column=second, title="""Average loss""", y_lim=True, y_max=5,
             hue=hue, ci=ci, tipo=None)
    i = 1
    axs[i].get_legend().remove()
    axs[i].set_ylabel(second, labelpad=16)


    # fig.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    fig.savefig(
        """{}solutions_{}_clients_line.png""".format(base_dir,
                                                num_clients), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}solutions_{}_clients_line.svg""".format(base_dir,
                                                num_clients), bbox_inches='tight',
        dpi=400)

if __name__ == "__main__":

    # alphas = ['0.1', '5.0']
    alphas = ['5.0', '5.0']
    models_names = ["cnn_a", "cnn_a"]
    configuration = {"dataset": ["Cifar10", "ImageNet"], "alpha": [float(i) for i in alphas]}
    models_names = ["cnn_a", "cnn_a"]
    datasets = configuration["dataset"]
    models_size = {"Cifar10": 3514.152, "ImageNet": 3524.412}
    solutions = ["FedNome", "MultiFedAvgRR", "FedFairMMFL", "MultiFedAvg", "MultiFedSpeedv1", "MultiFedSpeedv0"]
    # solutions = ["MultiFedSpeed@1", "MultiFedSpeed@2", "MultiFedSpeed@3", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    solutions = ["MultiFedSpeed@3", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
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

    d = """results/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(num_clients, alphas, datasets,
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
            read_models += [{"EMNIST": "CNN-A", "Cifar10": "CNN-A", "WISDM-W": "GRU", "ImageNet": "CNN-B"}[
                                dataset]] * len(df)
            model_size = models_size[dataset]
            size += [model_size] * len(df)
        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss
        read_training_clients += training_clients
        read_sizes += size

    first = 'Balanced accuracy'
    second = 'Loss'
    third = '# training clients'
    df = pd.DataFrame(
        {'Solution': read_solutions, first: np.array(read_accs) * 100, "Loss": read_loss, "Round (t)": read_round,
         "Model": read_models, "Dataset": read_datasets, '# training clients': read_training_clients,
         "Model size": read_sizes})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})
    # df['Accuracy efficiency'] = df['Accuracy'] / df['# training clients']
    # df['Loss efficiency'] = df['Loss'] / df['# training clients']

    print(df)

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, models_names, rounds)

    bar_metric(df, base_dir, "Solution", first, second, x_order, hue_order)
    plt.plot()
    bar_auc(df, base_dir, "Solution", first, second, third, solutions, solutions)
    plt.plot()
    bar_performance(df, base_dir, "Solution", first, second, third, solutions, solutions)
    plt.plot()
    line(df, base_dir, "Round (t)", first, second, "Solution", None)

