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
        "MultiFedAvg+MFP_v2": {"Strategy": "MultiFedAvg", "Version": "MFP_v2", "Table": "$MultiFedAvg+MFP_{v2}$"},
        "MultiFedAvg+MFP_v2_dh": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_dh", "Table": "$MultiFedAvg+MFP_{v2dh}$"},
        "MultiFedAvg+MFP_v2_iti": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_iti", "Table": "$MultiFedAvg+MFP_{v2iti}$"},
        "MultiFedAvg+FPD": {"Strategy": "MultiFedAvg", "Version": "FPD", "Table": "MultiFedAvg+FPD"},
        "MultiFedAvg+FP": {"Strategy": "MultiFedAvg", "Version": "FP", "Table": "MultiFedAvg+FP"},
        "MultiFedAvg": {"Strategy": "MultiFedAvg", "Version": "Original", "Table": "MultiFedAvg"},
        "MultiFedAvgRR": {"Strategy": "MultiFedAvgRR", "Version": "Original", "Table": "MultiFedAvgRR"},
        "FedFairMMFL": {"Strategy": "FedFairMMFL", "Version": "Original", "Table": "FedFairMMFL"},
        "MultiFedAvg-MDH": {"Strategy": "MultiFedAvg-MDH", "Version": "Original", "Table": "MultiFedAvg-MDH"},
        "DMA-FL": {"Strategy": "DMA-FL", "Version": "Original", "Table": "DMA-FL"},
        "AdaptiveFedAvg": {"Strategy": "AdaptiveFedAvg", "Version": "Original", "Table": "AdaptiveFedAvg"}
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

def mean_ci(values, ci=0.95):
    values = values.dropna().to_numpy()

    if len(values) == 0:
        return 0.0, 0.0

    mean = np.mean(values)

    if len(values) > 1:
        interval = st.t.interval(
            confidence=ci,
            df=len(values) - 1,
            loc=mean,
            scale=st.sem(values)
        )
        margin = mean - interval[0]
    else:
        margin = 0.0

    return round(mean, 2), round(margin, 2)


def group_by(df, metric, ci=0.95):
    values = df[metric].dropna().to_numpy()

    if len(values) == 0:
        return "0.0±0.0"

    mean = np.mean(values)

    if len(values) > 1:
        interval = st.t.interval(
            confidence=ci,
            df=len(values) - 1,
            loc=mean,
            scale=st.sem(values)
        )
        margin = mean - interval[0]
    else:
        margin = 0.0

    return f"{round(mean,2)}\u00B1{round(margin,2)}"

def table(df, write_path, metric, t=None):

    if t is not None:
        df = df[df['Round (t)'].isin(t)]

    datasets = df["Dataset"].unique().tolist()
    solutions = df["Table"].unique().tolist()
    alphas = sorted(df["Alpha"].unique().tolist())

    rows = []

    for dataset in datasets:
        for solution in solutions:

            row = {"Dataset": dataset, "Solution": solution}

            for alpha in alphas:

                filtered = df.query(
                    f"Dataset == '{dataset}' and Table == '{solution}' and Alpha == {alpha}"
                )

                mean, ci_margin = mean_ci(filtered[metric])

                row[f"{alpha}_mean"] = mean
                row[f"{alpha}_ci"] = ci_margin

            rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results.set_index(["Dataset", "Solution"], inplace=True)

    # ===============================
    # 🔹 GANHO VS BASELINE
    # ===============================

    baseline = "MultiFedAvg"

    for alpha in alphas:

        for dataset in datasets:

            base_value = df_results.loc[(dataset, baseline)][f"{alpha}_mean"]

            for solution in solutions:

                mean_val = df_results.loc[(dataset, solution)][f"{alpha}_mean"]

                gain = ((mean_val - base_value) / base_value) * 100
                df_results.loc[(dataset, solution), f"{alpha}_gain"] = round(gain, 2)

    # ===============================
    # 🔹 MARCAR MELHORES (COM IC)
    # ===============================

    for alpha in alphas:

        for dataset in datasets:

            subset = df_results.loc[dataset]

            means = subset[f"{alpha}_mean"]
            cis = subset[f"{alpha}_ci"]

            max_idx = means.idxmax()
            max_mean = means[max_idx]
            max_ci = cis[max_idx]

            max_lower = max_mean - max_ci
            max_upper = max_mean + max_ci

            for solution in subset.index:

                mean_val = subset.loc[solution][f"{alpha}_mean"]
                ci_val = subset.loc[solution][f"{alpha}_ci"]

                lower = mean_val - ci_val
                upper = mean_val + ci_val

                overlap = not (upper < max_lower or lower > max_upper)

                acc_str = f"{mean_val}±{ci_val}"
                gain_val = df_results.loc[(dataset, solution)][f"{alpha}_gain"]

                if gain_val >= 0:
                    gain_str = f"\\textuparrow{gain_val}\\%"
                else:
                    gain_str = f"\\textdownarrow{abs(gain_val)}\\%"

                if overlap:
                    acc_str = f"\\textbf{{{acc_str}}}"

                df_results.loc[(dataset, solution), f"{alpha}"] = acc_str
                df_results.loc[(dataset, solution), f"{alpha}_gain_str"] = gain_str

    # ===============================
    # 🔹 MONTAR TABELA FINAL
    # ===============================

    final_columns = []

    for alpha in alphas:
        final_columns.append(f"{alpha}")
        final_columns.append(f"{alpha}_gain_str")

    df_final = df_results[final_columns]

    Path(write_path).mkdir(parents=True, exist_ok=True)

    filename = f"{write_path}latex_table_{metric}.txt"

    latex = df_final.to_latex(escape=False)

    pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    print("\nTabela final:")
    print(df_final)
    print("\nSalvo em:", filename)

def improvements(df, datasets, metric):
    # , "FedKD+FP": "FedKD"
    indexes = df.index.tolist()
    solutions = pd.Series([i[1] for i in indexes]).unique().tolist()
    strategies = {solution: "MultiFedAvg" for solution in solutions}
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
    reference_solutions = {solution: "MultiFedAvg" for solution in solutions}

    for dataset in datasets:
        for solution in reference_solutions:

            reference_index = (dataset, solution)
            target_index = (dataset, reference_solutions[solution])

            for column in columns:

                mean_ref = float(df.loc[reference_index, column].split("±")[0])
                mean_base = float(df.loc[target_index, column].split("±")[0])

                difference = mean_ref - mean_base
                percentage_gain = round((difference / mean_base) * 100, 2)

                if percentage_gain >= 0:
                    diff_str = r"\textuparrow" + str(percentage_gain)
                else:
                    diff_str = r"\textdownarrow" + str(abs(percentage_gain))

                df_difference.loc[reference_index, column] = \
                    "(" + diff_str + "\%) " + df.loc[reference_index, column]

    return df_difference

def select_mean(index, column_values, columns, n_solutions):

    list_of_means = []
    indexes = []

    for i in range(len(column_values)):
        value = float(column_values[i].split("±")[0])
        list_of_means.append(value)

    for i in range(0, len(list_of_means), n_solutions):

        dataset_values = list_of_means[i: i + n_solutions]
        max_value = max(dataset_values)

        for j in range(i, i + n_solutions):
            if list_of_means[j] == max_value:
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
    # experiment_id = "label_shift#1_sudden"
    # experiment_id = "label_shift#1_recurrent"
    # experiment_id = "label_shift#2_sudden"
    # experiment_id = "label_shift#2_recurrent"
    experiment_id = "label_shift#3_sudden"
    # experiment_id = "label_shift#4_sudden"
    experiment_id = ""
    # experiment_id = "label_shift#4"
    # experiment_id = "label_shift#4_gradual"
    # experiment_id = "label_shift#5"
    # experiment_id = "label_shift#6"
    # experiment_id = "concept_drift#2"
    # experiment_id = "concept_drift#1_sudden"
    # experiment_id = "concept_drift#2_sudden"
    # experiment_id = "concept_drift#1_gradual"
    # experiment_id = "concept_drift#2_gradual"
    # experiment_id = "concept_drift#1_recurrent"
    # experiment_id = "concept_drift#2_recurrent"
    # experiment_id = "concept_drift#2_sudden"
    total_clients = 40
    # alphas = [10.0, 10.0]
    # alphas = [0.1, 0.1, 0.1]
    alphas = [10.0, 10.0, 10.0]
    # alphas = [1.0, 0.1, 0.1]
    # alphas = [0.1, 0.1]
    # alphas = [10.0]
    # alphas = [1.0, 1.0]

    # alphas = [10.0, 0.1]
    # dataset = ["CIFAR10", "WISDM-W"]
    # dataset = ["WISDM-W"]
    dataset = ["WISDM-W", "ImageNet10", "Foursquare"]
    # dataset = ["WISDM-W", "ImageNet10", "wikitext"]
    # dataset = ["WISDM-W", "ImageNet10"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    # model_name = [ "CNN", "gru", "lstm"]
    model_name = ["gru", "CNN", "lstm"]
    # model_name = ["gru"]
    # model_name = ["CNN", "gru"]
    fraction_fit = 0.3
    number_of_rounds = 100
    local_epochs = 1
    round_new_clients = 0
    train_test = "test"
    solutions = ["MultiFedAvg+MFP_v2", "MultiFedAvg+MFP_v2_dh", "MultiFedAvg+MFP_v2_iti", "MultiFedAvg+MFP", "MultiFedAvg+FPD",
                 "MultiFedAvg+FP", "DMA-FL", "AdaptiveFedAvg", "MultiFedAvg"]

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
    # table(df, write_path, "Accuracy (%)", t=[30,31,32,33,34,50,51,52,53,54,70,71,72,73,74])