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
    print(first_efficiency, second_efficiency)
    return pd.DataFrame({"Efficiency": [first_efficiency], second + " efficiency": [second_efficiency], third: [int(training_clients)], first: [acc], "Loss": [loss], first + " std": [acc_std], "Loss std": [loss_std]})

def group_by(df, first, second, third):

    area_first = trapz(df[first].to_numpy(), dx=1)
    area_second = trapz(df[second].to_numpy(), dx=1)
    area_first_efficiency = trapz(df.groupby("Round (t)").apply(lambda e: pd.DataFrame({"eff": [e[first].mean() / e["# training clients"].sum()]}))["eff"].to_numpy(), dx=1)
    area_third = trapz(df["# training clients"].to_numpy(), dx=1)

    return pd.DataFrame({"Efficiency AUC": area_first_efficiency, first + " AUC": [area_first], second + " AUC": [area_second], "# training clients AUC": [area_third]})

def latex(df, dir_path):
    df = df[df["Round (t)"] == 100]
    df = df[["Solution", "Dataset", "Efficiency", "Avg. $acc_b$", "# training clients", "Loss"]].round(2)
    df = df.set_index(["Solution", "Dataset"])
    print(df)
    df.to_latex(dir_path + "dataset.latex")

def auc_latex(df, dir_path):

    df = df.groupby(["Solution", "Dataset"]).apply(lambda e: group_by(e, "Avg. $acc_b$", "Loss", "# training clients")).reset_index().set_index(["Solution", "Dataset"])[["Efficiency AUC", "Avg. $acc_b$ AUC", "Loss AUC", "# training clients AUC"]].round(2)
    df.to_latex(dir_path + "dataset_metrics.latex")
    print(df)

if __name__ == "__main__":

    alphas = ['0.1', '5.0']
    # alphas = ['5.0', '0.1']
    models_names = ["gru", "cnn_a"]
    configuration = {"dataset": ["WISDM-P", "ImageNet"], "alpha": [float(i) for i in alphas]}
    models_names = ["gru", "cnn_a"]
    datasets = configuration["dataset"]
    # solutions = ["FedNome",  "MultiFedAvgRR", "FedFairMMFL", "MultiFedAvg", "MultiFedSpeedv1", "MultiFedSpeedv0", ]
    solutions = ["MultiFedSpeed@1", "MultiFedSpeed@2", "MultiFedSpeed@3", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
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
            df = pd.read_csv("""{}{}_{}_test_0_clients.csv""".format(d, dataset, solution))
            print("""{}{}_{}_test_0_clients.csv""".format(d, dataset, solution))
            print(df["Accuracy"].to_numpy())
            print(df["# training clients"].to_numpy())
            acc += df["Balanced accuracy"].tolist()
            loss += df["Loss"].tolist()
            training_clients += df["# training clients"].tolist()
            read_round += df["Round"].tolist()
            read_datasets += [dataset.replace("ImageNet", "CNN").replace("WISDM-P", "GRU")] * len(df)
        read_solutions += [solution] * len(acc)
        read_accs += acc
        read_loss += loss
        read_training_clients += training_clients

    first = 'Avg. $acc_b$'
    second = '# training clients'
    df = pd.DataFrame({'Solution': read_solutions, first: np.array(read_accs) * 100, "Loss": read_loss, "Round (t)": read_round,
                       "Dataset": read_datasets, '# training clients': read_training_clients})
    # df_2 = pd.DataFrame({'\u03B1': read_std_alpha, 'Dataset': read_std_dataset, 'Samples (%) std': read_num_samples_std})
    # df['Accuracy efficiency'] = df['Accuracy'] / df['# training clients']
    # df['Loss efficiency'] = df['Loss'] / df['# training clients']
    df = df.groupby(["Solution", "Dataset", "Round (t)"]).apply(lambda e: m(e, first, second, '# training clients')).reset_index()
    third = 'Efficiency'
    fourth = 'Loss efficiency'

    x_order = alphas

    hue_order = datasets
    base_dir = """analysis/solutions/clients_{}/{}/fc_{}/{}/{}/rounds_{}/""".format(num_clients, x_order, fc, hue_order, models_names, rounds)

    rounds_semi_convergence = [13, 36, 40, 47, 53]
    latex(df, base_dir)
    auc_latex(df, base_dir)





