from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from numpy import trapz

import copy

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

def read_data(read_solutions, read_dataset_order):
    df_concat = None
    solution_strategy_version = {
        "FedAvg+FP": {"Strategy": "FedAvg", "Version": "FP", "Table": "FedAvg+FP"},
        "FedAvg": {"Strategy": "FedAvg", "Version": "Original", "Table": "FedAvg"},
        "FedYogi+FP": {"Strategy": "FedYogi", "Version": "FP", "Table": "FedYogi+FP"},
        "FedYogi": {"Strategy": "FedYogi", "Version": "Original", "Table": "FedYogi"},
        "FedPer": {"Strategy": "FedPer", "Version": "Original", "Table": "FedPer"},
        "FedKD": {"Strategy": "FedKD", "Version": "Original", "Table": "FedKD"},
        "FedKD+FP": {"Strategy": "FedKD", "Version": "FP", "Table": "FedKD+FP"},
        "MultiFedAvg+MFP": {"Strategy": "MultiFedAvg", "Version": "MFP", "Table": "MultiFedAvg+MFP"},
        "MultiFedAvg+FPD": {"Strategy": "MultiFedAvg", "Version": "FPD", "Table": "MultiFedAvg+FPD"},
        "MultiFedAvg+FP": {"Strategy": "MultiFedAvg", "Version": "FP", "Table": "MultiFedAvg+FP"},
        "MultiFedAvg": {"Strategy": "MultiFedAvg", "Version": "Original", "Table": "MultiFedAvg"},
        "MultiFedAvgRR": {"Strategy": "MultiFedAvgRR", "Version": "Original", "Table": "MultiFedAvgRR"},
        "MultiFedAvg-MDH": {"Strategy": "MultiFedAvg-MDH", "Version": "Original", "Table": "MultiFedAvg-MDH"},
        "FedFairMMFL": {"Strategy": "FedFairMMFL", "Version": "Original", "Table": "FedFairMMFL"},
    }
    hue_order = []
    for solution in read_solutions:

        paths = read_solutions[solution]
        for i in range(len(paths)):
            try:
                dataset = read_dataset_order[i]
                path = paths[i]
                df = pd.read_csv(path)
                df["Solution"] = np.array([solution] * len(df))
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Dataset"] = np.array(["All datasets"] * len(df))
                df["Table"] = np.array([solution_strategy_version[solution]["Table"]] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                df["Alpha"] = np.array([0.1] * len(df))

                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution_strategy_version[solution]["Strategy"]
                if strategy not in hue_order:
                    hue_order.append(strategy)
            except Exception as e:
                print("\n######### \nFaltando", paths[i])
                print(e)

    return df_concat, hue_order

def group_by(df, metric):

    area = trapz(df[metric].to_numpy(), dx=1)

    return area



def table(df, write_path, metric, t=None):
    datasets = df["Dataset"].unique().tolist()
    alphas = sorted(df["Alpha"].unique().tolist())
    fraction_fit = sorted(df["Fraction fit"].unique().tolist())
    columns = df["Table"].unique().tolist()
    n_strategies = str(len(columns))

    print(columns)

    model_report = {i: {} for i in fraction_fit}
    if t is not None:
        df = df[df['Round (t)'].isin(t)]

    df_test = df[
        ['Round (t)', 'Table', 'Balanced accuracy (%)', 'Accuracy (%)', 'Fraction fit', 'Dataset',
         'Alpha']]

    # df_test = df_test.query("""Round in [10, 100]""")
    print("agrupou table")
    experiment = 1
    print(df_test)

    arr = []
    for dt in datasets:
        arr += [dt] * len(columns)
    index = [np.array(arr),
             np.array(columns * len(datasets))]

    models_dict = {}
    ci = 0.95

    # for alpha in model_report:
    models_datasets_dict = {dt: {column: {} for column in columns} for dt in datasets}
    for ff in fraction_fit:
        for column in columns:
            for dt in datasets:
                # models_datasets_dict[dt][column] = t_distribution((filter(df_test, dt,
                #                                                           alpha=float(alpha), strategy=column)[
                #     metric]).tolist(), ci)
                filtered = df_test.query(f"Dataset == '{dt}' and Table == '{column}'")
                filtered = filtered[filtered['Fraction fit'] == ff]
                size = filtered.shape[0]
                re = group_by(filtered, metric=metric)
                models_datasets_dict[dt][column][ff] = re / size

        model_metrics = []

        for dt in datasets:
            for column in columns:
                model_metrics.append(str(round(models_datasets_dict[dt][column][ff], 2)))

        models_dict[ff] = model_metrics

    print(models_dict)
    print(index)
    # exit()

    df_table = pd.DataFrame(models_dict, index=index).round(4)
    print("df table: ", df_table)

    print(df_table.to_string())

    df_accuracy_improvements = accuracy_improvement(df_table, datasets)

    indexes = df_table.index.tolist()
    n_solutions = len(pd.Series([i[1] for i in indexes]).unique().tolist())
    max_values = idmax(df_table, n_solutions)
    print("max values", max_values)

    for max_value in max_values:
        row_index = max_value[0]
        column = max_value[1]
        column_values = df_accuracy_improvements[column].tolist()
        column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

        df_accuracy_improvements[column] = np.array(column_values)

    df_accuracy_improvements.columns = np.array(list(model_report.keys()))
    print("melhorias")
    print(df_accuracy_improvements)

    indexes = fraction_fit
    for i in range(df_accuracy_improvements.shape[0]):
        row = df_accuracy_improvements.iloc[i]
        for index in indexes:
            value_string = row[index]
            add_textbf = False
            if "textbf{" in value_string:
                value_string = value_string.replace("textbf{", "").replace("}", "")
                add_textbf = True

            if ")" in value_string:
                value_string = value_string.replace("(", "").split(")")
                gain = value_string[0]
                acc = value_string[1]
            else:
                gain = ""
                acc = value_string

            if add_textbf:
                if gain != "":
                    gain = "textbf{" + gain + "}"
                acc = "textbf{" + acc + "}"

            row[index] = acc + " & " + gain

        df_accuracy_improvements.iloc[i] = row

    # print(df_accuracy_improvements.T)
    # print(df_accuracy_improvements.T.columns)
    # exit()
    df_accuracy_improvements = df_accuracy_improvements.T

    latex = df_accuracy_improvements.to_latex().replace("\\\nEMNIST", "\\\n\hline\nEMNIST").replace("\\\nGTSRB",
                                                                                                    "\\\n\hline\nGTSRB").replace(
        "\\\nCIFAR-10", "\\\n\hline\nCIFAR-10").replace("\\bottomrule", "\\hline\n\\bottomrule").replace("\\midrule",
                                                                                                         "\\hline\n\\midrule").replace(
        "\\toprule", "\\hline\n\\toprule").replace("textbf", r"\textbf").replace("\}", "}").replace("\{", "{").replace(
        "\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}").replace("\$", "$").replace("\&",
                                                                                                       "&").replace(
        "&  &", "& - &").replace("\_", "_").replace(
        "&  \\", "& - \\").replace(" - " + r"\textbf", " " + r"\textbf").replace("_{dc}", r"_{\text{dc}}").replace(
        "\multirow[t]{" + n_strategies + "}{*}{EMNIST}", "EMNIST").replace(
        "\multirow[t]{" + n_strategies + "}{*}{CIFAR10}", "CIFAR10").replace(
        "\multirow[t]{" + n_strategies + "}{*}{GTSRB}", "GTSRB").replace("\cline{1-4}", "\hline").replace("\cline{1-5}", "\hline").replace("\multirow[t]", "\multirow").replace("0.200000", str(int(30*0.2))).replace("0.300000", str(int(30*0.3))).replace("0.400000", str(int(30*0.4))).replace("0.500000", str(int(30*0.5))).replace("0.600000", str(int(30*0.6))).replace("0.700000", str(int(30*0.7))).replace("0.800000", str(int(30*0.8))).replace("0.900000", str(int(30*0.9)))

    Path(write_path).mkdir(parents=True, exist_ok=True)
    if t is not None:
        filename = """{}latex_round_auc_general_joint_datasets_{}_{}_fraction_fit.txt""".format(write_path, t, metric)
    else:
        filename = """{}latex_auc_general_joint_datasets_{}_fraction_fit.txt""".format(write_path, metric)
    pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    improvements(df_table, datasets, metric)

    #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")


def improvements(df, datasets, metric):
    # , "FedKD+FP": "FedKD"
    strategies = {"MultiFedAvg-MDH": "MultiFedAvg", "MultiFedAvgRR": "MultiFedAvg", "FedFairMMFL": "MultiFedAvg"}
    # strategies = {r"MultiFedAvg+FP": "MultiFedAvg"}
    columns = df.columns.tolist()
    improvements_dict = {'Dataset': [], 'Table': [], 'Original strategy': [], 'Alpha': [], metric: []}
    df_improvements = pd.DataFrame(improvements_dict)

    for dataset in datasets:
        for strategy in strategies:
            original_strategy = strategies[strategy]

            for j in range(len(columns)):
                index = (dataset, strategy)
                index_original = (dataset, original_strategy)
                print(df)
                print("indice: ", index)
                acc = float(df.loc[index].tolist()[j].replace("textbf{", "").replace(u"\u00B1", "")[:4])
                acc_original = float(
                    df.loc[index_original].tolist()[j].replace("textbf{", "")[:4].replace(u"\u00B1", ""))

                row = {'Dataset': [dataset], 'Table': [strategy], 'Original strategy': [original_strategy],
                       'Alpha': [columns[j]], metric: [acc - acc_original]}
                row = pd.DataFrame(row)

                print(row)

                if len(df_improvements) == 0:
                    df_improvements = row
                else:
                    df_improvements = pd.concat([df_improvements, row], ignore_index=True)

    print(df_improvements)


def groupb_by_plot(self, df, metric):
    accuracy = float(df[metric].mean())
    loss = float(df['Loss'].mean())

    return pd.DataFrame({metric: [accuracy], 'Loss': [loss]})


def filter(df, dataset, alpha, strategy=None):
    # df['Balanced accuracy (%)'] = df['Balanced accuracy (%)']*100
    if strategy is not None:
        df = df.query(
            """ Dataset=='{}' and Table=='{}'""".format(str(dataset), strategy))
        df = df[df['Alpha'] == alpha]
    else:
        df = df.query(
            """and Dataset=='{}'""".format((dataset)))
        df = df[df['Alpha'] == alpha]

    print("filtrou: ", df, dataset, alpha, strategy)

    return df


def t_distribution(data, ci):
    if len(data) > 1:
        min_ = st.t.interval(confidence=ci, df=len(data) - 1,
                             loc=np.mean(data),
                             scale=st.sem(data))[0]

        mean = np.mean(data)
        average_variation = (mean - min_).round(1)
        mean = mean.round(1)

        return str(mean) + u"\u00B1" + str(average_variation)
    else:
        return str(round(data[0], 1)) + u"\u00B1" + str(0.0)


def accuracy_improvement(df, datasets):
    df_difference = copy.deepcopy(df)
    columns = df.columns.tolist()
    indexes = df.index.tolist()
    solutions = pd.Series([i[1] for i in indexes]).unique().tolist()
    # reference_solutions = {"MultiFedAvg+FP": "MultiFedAvg", "MultiFedYogi+FP": "MultiFedYogi", "FedAvgGlobalModelEval+FP": "FedAvgGlobalModelEval", "MultiFedKD+FP": "FedKD"}
    # reference_solutions = {"MultiFedAvg+FP": "MultiFedAvg", "MultiFedAvgGlobalModelEval+FP": "MultiFedAvgGlobalModelEval"}
    # ,
    #                            "FedKD+FP": "FedKD"
    reference_solutions = {"MultiFedAvg-MDH": "MultiFedAvg", "MultiFedAvgRR": "MultiFedAvg", "FedFairMMFL": "MultiFedAvg"}

    print(df_difference)
    # exit()

    for dataset in datasets:
        for solution in reference_solutions:
            reference_index = (dataset, solution)
            target_index = (dataset, reference_solutions[solution])

            for column in columns:
                difference = str(round(float(df.loc[reference_index, column].replace(u"\u00B1", "")[:4]) - float(
                    df.loc[target_index, column].replace(u"\u00B1", "")[:4]), 2))
                difference = str(
                    round(float(difference) * 100 / float(df.loc[target_index, column][:4].replace(u"\u00B1", "")), 2))
                if difference[0] != "-":
                    difference = r"\textuparrow" + difference
                else:
                    difference = r"\textdownarrow" + difference.replace("-", "")
                df_difference.loc[reference_index, column] = "(" + difference + "\%)" + df.loc[reference_index, column]

    return df_difference


def select_mean(index, column_values, columns, n_solutions):
    list_of_means = []
    indexes = []
    print("ola: ", column_values, "ola0")

    for i in range(len(column_values)):
        print("valor: ", column_values[i])
        value = float(str(str(column_values[i])[:4]).replace(u"\u00B1", ""))
        interval = 0
        minimum = value - interval
        maximum = value + interval
        list_of_means.append((value, minimum, maximum))

    for i in range(0, len(list_of_means), n_solutions):

        dataset_values = list_of_means[i: i + n_solutions]
        max_tuple = max(dataset_values, key=lambda e: e[0])
        column_min_value = max_tuple[1]
        column_max_value = max_tuple[2]
        print("maximo: ", column_max_value)
        for j in range(len(list_of_means)):
            value_tuple = list_of_means[j]
            min_value = value_tuple[1]
            max_value = value_tuple[2]
            if j >= i and j < i + n_solutions:
                if not (max_value < column_min_value or min_value > column_max_value):
                    indexes.append([j, columns[index]])

    return indexes


def idmax(df, n_solutions):
    df_indexes = []
    columns = df.columns.tolist()
    print("colunas", columns)
    for i in range(len(columns)):
        column = columns[i]
        column_values = df[column].tolist()
        print("ddd", column_values)
        indexes = select_mean(i, column_values, columns, n_solutions)
        df_indexes += indexes

    return df_indexes


if __name__ == "__main__":
    experiment_id = 2
    total_clients = 30
    # alphas = [10.0, 10.0]
    alphas = [0.1, 0.1, 0.1]
    # alphas = [0.1, 0.1]
    # alphas = [1.0, 1.0]
    # alphas = [10.0, 0.1]
    # dataset = ["WISDM-W", "CIFAR10"]
    dataset = ["WISDM-W", "ImageNet10", "Gowalla"]
    # dataset = ["WISDM-W", "ImageNet10"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    model_name = ["gru", "CNN", "lstm"]
    # model_name = ["gru", "CNN"]
    fraction_fit = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    number_of_rounds = 100
    local_epochs = 1
    round_new_clients = 0
    train_test = "test"
    # solutions = ["MultiFedAvg+MFP", "MultiFedAvg+FPD", "MultiFedAvg+FP", "MultiFedAvg", "MultiFedAvgRR"]
    # solutions = ["MultiFedAvg-MDH", "MultiFedAvg"]
    solutions = ["MultiFedAvg-MDH", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]

    read_solutions = {solution: [] for solution in solutions}
    read_dataset_order = []
    for solution in solutions:
        for dt in dataset:
            for ff in fraction_fit:
                algo = dt + "_" + solution

                read_path = """../system/results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                    experiment_id,
                    total_clients,
                    alphas,
                    dataset,
                    model_name,
                    ff,
                    number_of_rounds,
                    local_epochs,
                    train_test)
                read_dataset_order.append(dt)

                read_solutions[solution].append("""{}{}_{}.csv""".format(read_path, dt, solution))

    write_path = """plots/MEFL/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
        experiment_id,
        total_clients,
        alphas,
        dataset,
        model_name,
        fraction_fit,
        number_of_rounds,
        local_epochs)

    print(read_solutions)

    df, hue_order = read_data(read_solutions, read_dataset_order)

    table(df, write_path, "Balanced accuracy (%)", t=None)
    table(df, write_path, "Accuracy (%)", t=None)
    # table(df, write_path, "Accuracy (%)", t=[39,40])