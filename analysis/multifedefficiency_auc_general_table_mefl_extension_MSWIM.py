from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from numpy import trapz
import sys

import copy

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
import json

def get_std(df):

    df2 = df[["Accuracy (%)", "training clients and models", "Round (t)"]]
    df2["n"] = [len(json.loads(i)) for i in df["training clients and models"].tolist()]
    df2 = df2.query("n > 0")
    accs = df2["Accuracy (%)"].tolist()
    rounds = df2["Round (t)"].tolist()
    negative_oscillations = []
    for i in range(1,len(accs)):
        diff = accs[i] - accs[i-1]
        if diff < 0:
            negative_oscillations.append(diff)

    return pd.DataFrame({"# negative oscillations": [len(negative_oscillations)] * len(df), "Amplitude": [sum(negative_oscillations)] * len(df)})

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
        "MultiFedAvg+MFP_v2": {"Strategy": "MultiFedAvg", "Version": "MFP_v2", "Table": "$MultiFedAvg+MFP_{v2}$"},
        "MultiFedAvg+MFP_v2_dh": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_dh", "Table": "$MultiFedAvg+MFP_{v2\_dh}$"},
        "MultiFedAvg+MFP_v2_iti": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_iti", "Table": "$MultiFedAvg+MFP_{v2\_iti}$"},
        "MultiFedAvg+FPD": {"Strategy": "MultiFedAvg", "Version": "FPD", "Table": "MultiFedAvg+FPD"},
        "MultiFedAvg+FP": {"Strategy": "MultiFedAvg", "Version": "FP", "Table": "MultiFedAvg+FP"},
        "MultiFedAvg": {"Strategy": "MultiFedAvg", "Version": "Original", "Table": "MultiFedAvg"},
        "MultiFedAvgRR": {"Strategy": "MultiFedAvgRR", "Version": "Original", "Table": "MultiFedAvgRR"},
        "FedFairMMFL": {"Strategy": "FedFairMMFL", "Version": "Original", "Table": "FedFairMMFL"},
        "MultiFedAvg-MDH": {"Strategy": "MultiFedAvg-MDH", "Version": "Original", "Table": "MultiFedAvg-MDH"}
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
                df["Dataset"] = np.array([dataset.replace("WISDM-W", "WISDM").replace("ImageNet10", "ImageNet-10")] * len(df))
                df["Table"] = np.array([solution_strategy_version[solution]["Table"]] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                df["Alpha"] = np.array([0.1] * len(df))
                # df["Efficiency"] = np.array(df["# training clients"])
                std = get_std(df)
                df["# negative oscillations"] = std["# negative oscillations"].to_numpy()
                df["Amplitude"] = std["Amplitude"].to_numpy()

                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution_strategy_version[solution]["Strategy"]
                if strategy not in hue_order:
                    hue_order.append(strategy)
            except Exception as e:
                print("read_data error")
                print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    return df_concat, hue_order

def group_by(df, metric):

    area = trapz(df[metric].to_numpy(), dx=1)

    return area



def table(df, write_path, metric, t=None):
    datasets = df["Dataset"].unique().tolist()
    alphas = sorted(df["Alpha"].unique().tolist())
    columns = df["Table"].unique().tolist()
    n_strategies = str(len(columns))

    print(columns)

    model_report = {i: {} for i in alphas}
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
    models_datasets_dict = {dt: {} for dt in datasets}
    for column in columns:
        for dt in datasets:
            # models_datasets_dict[dt][column] = t_distribution((filter(df_test, dt,
            #                                                           alpha=float(alpha), strategy=column)[
            #     metric]).tolist(), ci)
            filtered = df_test.query(f"Dataset == '{dt}' and Table == '{column}'")
            size = filtered.shape[0]
            re = group_by(filtered, metric=metric)
            models_datasets_dict[dt][column] = re / size

    model_metrics = []

    for dt in datasets:
        for column in columns:
            model_metrics.append(str(round(models_datasets_dict[dt][column], 2)))

    models_dict[0.1] = model_metrics

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

    indexes = alphas
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
        "\multirow[t]{" + n_strategies + "}{*}{GTSRB}", "GTSRB").replace("\cline{1-4}", "\hline").replace("\cline{1-5}", "\hline").replace("\multirow[t]", "\multirow").replace("MultiFedAvg-MDH", "MultiFedAvg-MDH").replace("\cline{1-3}", "\hline").replace("WISDM", "GRU").replace("ImageNet-10", "CNN"). replace("Gowalla", "LSTM").replace(r"\textuparrow0.0\%", "0.0\%")

    Path(write_path).mkdir(parents=True, exist_ok=True)
    if t is not None:
        filename = """{}latex_round_auc_general_novo_{}_{}.txt""".format(write_path, [min(t), max(t)], metric)
    else:
        filename = """{}latex_auc_general_noovo_{}.txt""".format(write_path, metric)
    pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    improvements(df_table, datasets, metric)

    #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")


def improvements(df, datasets, metric):
    # , "FedKD+FP": "FedKD"
    # strategies = {"MultiFedAvg-MDH": "MultiFedAvg", "FedFairMMFL": "MultiFedAvg", "MultiFedAvgRR": "MultiFedAvg"}
    strategies = {"$MultiFedAvg+MFP_{v2}$": "MultiFedAvg", "$MultiFedAvg+MFP_{v2\_dh}$": "MultiFedAvg", "$MultiFedAvg+MFP_{v2\_iti}$": "MultiFedAvg", "MultiFedAvg+MFP": "MultiFedAvg", "MultiFedAvg+FP": "MultiFedAvg"}
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
    reference_solutions = {"$MultiFedAvg+MFP_{v2}$": "MultiFedAvg", "$MultiFedAvg+MFP_{v2\_dh}$": "MultiFedAvg", "$MultiFedAvg+MFP_{v2\_iti}$": "MultiFedAvg", "MultiFedAvg+MFP": "MultiFedAvg", "MultiFedAvg+FP": "MultiFedAvg"}
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
    # experiment_id = "label_shift#1"
    # experiment_id = "label_shift#2"
    # experiment_id = "label_shift#3"
    # experiment_id = "label_shift#3_gradual"
    # experiment_id = "label_shift#4"
    # experiment_id = "label_shift#4_gradual"
    experiment_id = "label_shift#5"
    # experiment_id = "label_shift#6"
    # experiment_id = "concept_drift#1"
    # experiment_id = "concept_drift#2"
    # experiment_id = "concept_drift#1_gradual"
    # experiment_id = "concept_drift#2_gradual"
    # experiment_id = "concept_drift#1_recurrent"
    # experiment_id = "concept_drift#2_recurrent"
    total_clients = 40
    # alphas = [10.0, 10.0]
    alphas = [0.1, 0.1, 0.1]
    # alphas = [10.0, 10.0, 10.0]
    # alphas = [1.0, 0.1, 0.1]
    # alphas = [0.1, 0.1]
    # alphas = [10.0]
    # alphas = [1.0, 1.0]

    # alphas = [10.0, 0.1]
    # dataset = ["CIFAR10", "WISDM-W"]
    # dataset = ["WISDM-W"]
    # dataset = ["ImageNet10", "WISDM-W", "Gowalla"]
    dataset = ["ImageNet10", "WISDM-W", "wikitext"]
    # dataset = ["WISDM-W", "ImageNet10"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    model_name = ["CNN", "gru", "lstm"]
    # model_name = ["gru"]
    # model_name = ["CNN", "gru"]
    fraction_fit = 0.3
    number_of_rounds = 100
    local_epochs = 1
    round_new_clients = 0
    train_test = "test"
    # solutions = ["MultiFedAvg+MFP", "MultiFedAvg+FPD", "MultiFedAvg+FP", "MultiFedAvg", "MultiFedAvgRR"]
    # solutions = ["MultiFedAvg+MFP_v2", "MultiFedAvg+MFP", "MultiFedAvg+FPD", "MultiFedAvg+FP", "MultiFedAvg"]
    # solutions = ["MultiFedAvg+MFP", "MultiFedAvg+FPD", "MultiFedAvg+FP", "MultiFedAvg", "MultiFedAvgRR"]
    solutions = ["MultiFedAvg+MFP_v2", "MultiFedAvg+MFP_v2_dh", "MultiFedAvg+MFP_v2_iti", "MultiFedAvg+MFP", "MultiFedAvg+FP", "MultiFedAvg"]

    read_solutions = {solution: [] for solution in solutions}
    read_dataset_order = []
    for solution in solutions:
        for dt in dataset:
            algo = dt + "_" + solution

            read_path = """../system/results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                experiment_id,
                total_clients,
                alphas,
                dataset,
                model_name,
                fraction_fit,
                number_of_rounds,
                local_epochs,
                train_test)
            read_dataset_order.append(dt)

            # read_solutions[solution].append("""{}{}_{}.csv""".format(read_path, dt, solution.replace("MultiFedAvg-MDH", "HMultiFedAvg")))
            read_solutions[solution].append(
                """{}{}_{}.csv""".format(read_path, dt, solution))

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

    pd.set_option('display.max_rows', None)
    print(df[["Strategy", "Dataset", "# negative oscillations", "Amplitude"]].drop_duplicates())

    # table(df, write_path, "Balanced accuracy (%)", t=None)
    table(df, write_path, "Accuracy (%)", t=None)
    # table(df, write_path, "Accuracy (%)", t=[i for i in range(1, 31)])
    # table(df, write_path, "Accuracy (%)", t=[i for i in range(1, 51)])
    # table(df, write_path, "Accuracy (%)", t=[i for i in range(70, 101)])