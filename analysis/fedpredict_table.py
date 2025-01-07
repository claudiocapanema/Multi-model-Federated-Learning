from pathlib import Path
import copy
import numpy as np
import pandas as pd
import scipy.stats as st

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

def read_data(read_solutions, read_dataset_order):

    df_concat = None
    solution_strategy_version = {"MultiFedAvgWithFedPredict": {"Strategy_separated": "MultiFedAvg", "Version": "FP", "Strategy": "MultiFedAvg+FP"},
                                 "MultiFedAvg": {"Strategy_separated": "MultiFedAvg", "Version": "Original", "Strategy": "MultiFedAvg"},
                                 "MultiFedAvgGlobalModelEval": {"Strategy_separated": "MultiFedAvgGlobalModelEval", "Version": "Original", "Strategy": "MultiFedAvgGlobalModelEval"},
                                 "MultiFedAvgGlobalModelEvalWithFedPredict": {"Strategy_separated": "MultiFedAvgGlobalModelEval", "Version": "FP", "Strategy": "MultiFedAvgGlobalModelEval+FP"},
                                 "MultiFedPer": {"Strategy_separated": "MultiFedPer", "Version": "Original", "Strategy": "MultiFedPer"},
                                 "MultiFedYogi": {"Strategy_separated": "MultiFedYogi", "Version": "Original", "Strategy": "MultiFedYogi"}, "MultiFedYogiWithFedPredict": {"Strategy_separated": "MultiFedYogi", "Version": "FP", "Strategy": "MultiFedYogiWithFedPredict"}}
    for solution in read_solutions:

        paths = read_solutions[solution]
        for i in range(len(paths)):
            dataset = read_dataset_order[i]
            path = paths[i]
            df = pd.read_csv(path)
            df["Solution"] = np.array([solution] * len(df))
            df["Accuracy (%)"] = df["Accuracy"] * 100
            df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
            df["Round (t)"] = df["Round"]
            df["Dataset"] = np.array([dataset] * len(df))
            df["Strategy_separated"] = np.array([solution_strategy_version[solution]["Strategy_separated"]] * len(df))
            df["Strategy"] = np.array(
                [solution_strategy_version[solution]["Strategy"]] * len(df))
            df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))

            if df_concat is None:
                df_concat = df
            else:
                df_concat = pd.concat([df_concat, df])

    print(df_concat.columns)

    return df_concat


def table(df, write_path, t=None):

    datasets = df["Dataset"].unique().tolist()
    alphas = df["Alpha"].unique().tolist()
    columns = df["Strategy"].unique().tolist()


    model_report = {i: {} for i in alphas}
    if t is not None:
        df = df[df['Round (t)'] == t]
    
    df_test = df[
        ['Round (t)', 'Strategy', 'Balanced accuracy (%)', 'Accuracy (%)', 'Fraction fit', 'Dataset',
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
    
    for alpha in model_report:
        models_datasets_dict = {dt: {} for dt in datasets}
        for column in columns:
            for dt in datasets:
                models_datasets_dict[dt][column] = t_distribution((filter(df_test, dt,
                                                                     alpha=float(alpha), strategy=column)[
                    'Balanced accuracy (%)']).tolist(), ci)

        model_metrics = []

        for dt in datasets:
            for column in columns:
                model_metrics.append(models_datasets_dict[dt][column])

        models_dict[alpha] = model_metrics

    print(models_dict)
    print(index)


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
        "\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}").replace("\$", "$").replace("\&", "&").replace("&  &", "& - &").replace("\_", "_").replace(
        "&  \\", "& - \\").replace(" - " + r"\textbf", " " + r"\textbf").replace("_{dc}", r"_{\text{dc}}").replace("\multirow[t]{5}{*}{EMNIST}", "EMNIST").replace("\multirow[t]{5}{*}{CIFAR10}", "CIFAR10").replace("\multirow[t]{5}{*}{GTSRB}", "GTSRB").replace("\cline{1-5}", "\hline")

    if t is not None:
        filename = """{}latex_round_{}.txt""".format(write_path, t)
    else:
        filename = """{}latex.txt""".format(write_path)
    pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    improvements(df_table, datasets)

    #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")


def improvements(df, datasets):
    # strategies = {r"MultiFedAvg+FP": "MultiFedAvg", r"MultiFedYogi+FP": "MultiFedYogi"}
    strategies = {r"MultiFedAvg+FP": "MultiFedAvg"}
    columns = df.columns.tolist()
    improvements_dict = {'Dataset': [], 'Strategy': [], 'Original strategy': [], 'Alpha': [], 'Balanced accuracy (%)': []}
    df_improvements = pd.DataFrame(improvements_dict)

    for dataset in datasets:
        for strategy in strategies:
            original_strategy = strategies[strategy]

            for j in range(len(columns)):
                index = (dataset, strategy)
                index_original = (dataset, original_strategy)
                print(df)
                print("indice: ", index)
                acc = float(df.loc[index].tolist()[j].replace("textbf{", "")[:4])
                acc_original = float(df.loc[index_original].tolist()[j].replace("textbf{", "")[:4])

                row = {'Dataset': [dataset], 'Strategy': [strategy], 'Original strategy': [original_strategy],
                       'Alpha': [columns[j]], 'Balanced accuracy (%)': [acc - acc_original]}
                row = pd.DataFrame(row)

                print(row)

                if len(df_improvements) == 0:
                    df_improvements = row
                else:
                    df_improvements = pd.concat([df_improvements, row], ignore_index=True)

    print(df_improvements)


def groupb_by_plot(self, df):
    accuracy = float(df['Balanced accuracy (%)'].mean())
    loss = float(df['Loss'].mean())

    return pd.DataFrame({'Balanced accuracy (%)': [accuracy], 'Loss': [loss]})


def filter(df, dataset, alpha, strategy=None):
    # df['Balanced accuracy (%)'] = df['Balanced accuracy (%)']*100
    if strategy is not None:
        df = df.query(
            """ Dataset=='{}' and Strategy=='{}'""".format(str(dataset), strategy))
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
        return str(round(data[0],1)) + u"\u00B1" + str(0.0)

def accuracy_improvement(df, datasets):

    df_difference = copy.deepcopy(df)
    columns = df.columns.tolist()
    indexes = df.index.tolist()
    solutions = pd.Series([i[1] for i in indexes]).unique().tolist()
    # reference_solutions = {"MultiFedAvg+FP": "MultiFedAvg", "MultiFedYogi+FP": "MultiFedYogi", "MultiFedAvgGlobalModelEval+FP": "MultiFedAvgGlobalModelEval"}
    reference_solutions = {"MultiFedAvg+FP": "MultiFedAvg", "MultiFedAvgGlobalModelEval+FP": "MultiFedAvgGlobalModelEval"}


    for dataset in datasets:
        for solution in reference_solutions:
            reference_index = (dataset, solution)
            target_index = (dataset, reference_solutions[solution])

            for column in columns:
                difference = str(round(float(df.loc[reference_index, column][:4]) - float(df.loc[target_index, column][:4]), 1))
                difference = str(round(float(difference)*100/float(df.loc[target_index, column][:4]), 1))
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
        interval = float(str(column_values[i])[5:8])
        minimum = value - interval
        maximum = value + interval
        list_of_means.append((value, minimum, maximum))


    for i in range(0, len(list_of_means), n_solutions):

        dataset_values = list_of_means[i: i+n_solutions]
        max_tuple = max(dataset_values, key=lambda e: e[0])
        column_min_value = max_tuple[1]
        column_max_value = max_tuple[2]
        print("maximo: ", column_max_value)
        for j in range(len(list_of_means)):
            value_tuple = list_of_means[j]
            min_value = value_tuple[1]
            max_value = value_tuple[2]
            if j >= i and j < i+n_solutions:
                if not(max_value < column_min_value or min_value > column_max_value):
                    indexes.append([j, columns[index]])

    return indexes

def idmax( df, n_solutions):

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
    cd = "False"
    num_clients = 20
    alphas = [0.1, 1.0, 10.0]
    dataset = ["EMNIST", "CIFAR10", "GTSRB"]
    # dataset = ["EMNIST", "CIFAR10"]
    models_names = ["cnn_a"]
    join_ratio = 0.3
    global_rounds = 100
    local_epochs = 1
    fraction_new_clients = 0
    round_new_clients = 0
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg", "MultiFedAvgGlobalModelEval", "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedPer", "MultiFedYogi", "MultiFedYogiWithFedPredict"]
    solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg", "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedAvgGlobalModelEval",
                 "MultiFedPer"]

    read_solutions = {solution: [] for solution in solutions}
    read_dataset_order = []
    for solution in solutions:
        for alpha in alphas:
            for dt in dataset:
                read_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
                    cd,
                    fraction_new_clients,
                    round_new_clients,
                    num_clients,
                    [str(alpha)],
                    alpha,
                    alpha,
                    [str(dt)],
                    0,
                    0,
                    models_names,
                    join_ratio,
                    global_rounds,
                    local_epochs)
                read_dataset_order.append(dt)

                read_solutions[solution].append("""{}{}_{}_test_0.csv""".format(read_path, dt, solution))

    write_path = """plots/single_model/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
        cd,
        fraction_new_clients,
        round_new_clients,
        num_clients,
        [str(alpha)],
        alpha,
        alpha,
        dataset,
        0,
        0,
        models_names,
        join_ratio,
        global_rounds,
        local_epochs)

    df = read_data(read_solutions, read_dataset_order)
    table(df, write_path, t=None)
    table(df, write_path, t=100)