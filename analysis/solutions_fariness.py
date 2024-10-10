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
import numpy as np, scipy.stats as st
from numpy import trapz

custom_dict = {"MultiFedSpeed@3": 2, "MultiFedSpeed@2": 1, "MultiFedSpeed@1": 0, "MultiFedAvg": 3, "MultiFedAvgRR": 4, "FedFairMMFL": 5, "GRU": 0, "CNN-A": 1, "CNN-B": 2}

def ci(data):

    mi, ma =  st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))
    mean_value = (mi + ma)/2
    diff = ma - mean_value
    mean_value = round(mean_value, 2)
    diff = round(diff, 2)
    return """{} $\pm$ {}""".format(mean_value, diff)

def m(df, first, second, third):
    # area_first = trapz(df[first].to_numpy(), dx=1)
    # area_second = trapz(df[second].to_numpy(), dx=1)
    # area_first_efficiency = trapz(df.groupby("Round (t)").apply(
    #     lambda e: pd.DataFrame({"eff": [e[first].mean() / e["# training clients"].sum()]}))["eff"].to_numpy(), dx=1)
    # area_third = trapz(df["# training clients"].to_numpy(), dx=1)

    first_efficiency = df[first].mean() / df[third].mean()
    second_efficiency = df[second].mean() / df[third].mean()
    training_clients = df[third].mean()
    acc = df[first].mean()
    loss = df["Loss"].mean()
    acc_std = df[first].std()
    loss_std = df["Loss"].std()
    model_size = df["Model size"].mean()
    kb_transmitted = (df["# training clients"].to_numpy() * df["Model size"].to_numpy()).mean()
    print(first_efficiency, second_efficiency)
    return pd.DataFrame({"Efficiency": [first_efficiency], second + " efficiency": [second_efficiency], third: [int(training_clients)], first: [acc], "Loss": [loss], first + " std": [acc_std], "Loss std": [loss_std], "Communication cost (MB)": [kb_transmitted], "Model size": [model_size]})

def group_by(df, first, second, third):

    area_first = trapz(df[first].to_numpy(), dx=1)
    area_second = trapz(df[second].to_numpy(), dx=1)
    area_first_efficiency = trapz(df.groupby("Round (t)").apply(lambda e: pd.DataFrame({"eff": [e[first].mean() / e["# training clients"].sum()]}))["eff"].to_numpy(), dx=1)
    kb_transmitted_auc = trapz(df.groupby("Round (t)").apply(
        lambda e: pd.DataFrame({"kb": [(e["# training clients"].to_numpy() * e["Model size"].to_numpy()).sum()]}))[
                                   "kb"].to_numpy(), dx=1)
    area_third = trapz(df["# training clients"].to_numpy(), dx=1)

    return pd.DataFrame({"Efficiency AUC": area_first_efficiency, first + " AUC": [area_first], second + " AUC": [area_second], "# training clients AUC": [area_third], "Communication cost (MB) AUC": [kb_transmitted_auc]})

def latex(df, dir_path):
    df = df[df["Round (t)"] == 100]
    df = df[["Solution", "Model", "Efficiency", "Communication cost (MB)", "# training clients", "Avg. $acc_b$", "Loss"]].round(2)
    df = df.set_index(["Solution", "Model"]).sort_index(key=lambda x: x.map(custom_dict))
    print(df)
    latex = df.to_latex().replace("\multirow[t]", "\multirow").replace("\cline{1-7}", "\midrule").replace("0000", "")
    f = open(dir_path + "models_metrics.latex", "w")
    f.write(latex)
    f.close()

def auc_latex(df, dir_path):

    df = df.groupby(["Solution", "Model"]).apply(lambda e: group_by(e, "Avg. $acc_b$", "Loss", "# training clients")).reset_index().set_index(["Solution", "Model"])[["Efficiency AUC", "Communication cost (MB) AUC", "# training clients AUC", "Avg. $acc_b$ AUC", "Loss AUC"]].round(2).sort_index(key=lambda x: x.map(custom_dict))
    latex = df.to_latex().replace("\multirow[t]", "\multirow").replace("\cline{1-7}", "\midrule").replace("0000", "")
    f = open(dir_path + "models_metrics_auc.latex", "w")
    f.write(latex)
    f.close()

if __name__ == "__main__":

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

    # models_names = ["cnn_a", "cnn_a"]
    datasets = configuration["dataset"]
    models_size = {"WISDM-P": 0.039024, "WISDM-W": 0.039024, "CIFAR10": 3.514152, "ImageNet": 3.524412, "ImageNet_v2": 3.524412}

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
    df = pd.DataFrame(
        {'Solution': read_solutions, first: np.array(read_accs) * 100, "Loss": read_loss, "Round (t)": read_round,
         "Model": read_models, "Dataset": read_datasets, '# training clients': read_training_clients,
         "Model size": read_sizes})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})
    # df['Accuracy efficiency'] = df['Accuracy'] / df['# training clients']
    # df['Loss efficiency'] = df['Loss'] / df['# training clients']
    df = df.groupby(["Solution", "Dataset", "Model", "Round (t)"]).apply(lambda e: m(e, first, second, '# training clients')).reset_index()
    third = 'Efficiency'
    fourth = 'Loss efficiency'

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, models_names, rounds)

    rounds_semi_convergence = [13, 36, 40, 47, 53]
    latex(df, base_dir)
    auc_latex(df, base_dir)





