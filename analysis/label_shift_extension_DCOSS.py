from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import copy


# ===============================
# 🔹 EXTRAIR ALPHA DO NOME
# ===============================

def extract_alpha_from_experiment(experiment_id):
    alpha_part = experiment_id.split("#")[1]
    first_alpha = alpha_part.split("-")[0]
    return float(first_alpha)


# ===============================
# 🔹 MÉDIA + IC
# ===============================

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

def read_data(read_solutions, read_dataset_order):

    df_concat = None

    solution_strategy_version = {
        "MultiFedAvg+MFP_v2": {"Strategy": "MultiFedAvg", "Version": "MFP_v2", "Table": "$MultiFedAvg+MFP_{v2}$"},
        "MultiFedAvg+MFP_v2_dh": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_dh", "Table": "$MultiFedAvg+MFP_{v2dh}$"},
        "MultiFedAvg+MFP_v2_iti": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_iti", "Table": "$MultiFedAvg+MFP_{v2iti}$"},
        "MultiFedAvg+MFP": {"Strategy": "MultiFedAvg", "Version": "MFP", "Table": "MultiFedAvg+MFP"},
        "MultiFedAvg+FPD": {"Strategy": "MultiFedAvg", "Version": "FPD", "Table": "MultiFedAvg+FPD"},
        "MultiFedAvg+FP": {"Strategy": "MultiFedAvg", "Version": "FP", "Table": "MultiFedAvg+FP"},
        "DMA-FL": {"Strategy": "DMA-FL", "Version": "Original", "Table": "DMA-FL"},
        "AdaptiveFedAvg": {"Strategy": "AdaptiveFedAvg", "Version": "Original", "Table": "AdaptiveFedAvg"},
        "MultiFedAvg": {"Strategy": "MultiFedAvg", "Version": "Original", "Table": "MultiFedAvg"},
    }

    for solution in read_solutions:

        paths = read_solutions[solution]

        for i in range(len(paths)):

            try:
                dataset = read_dataset_order[i]
                path = paths[i]

                df = pd.read_csv(path)

                df["Solution"] = solution
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Dataset"] = dataset
                df["Table"] = solution_strategy_version[solution]["Table"]

                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

            except Exception as e:
                print("Arquivo faltando:", paths[i])
                print(e)

    return df_concat

def read_data_multi_experiments(
    experiment_ids,
    solutions,
    datasets,
    total_clients,
    model_name,
    fraction_fit,
    number_of_rounds,
    local_epochs,
    train_test
):

    df_concat = None

    for experiment_id in experiment_ids:

        alpha_value = extract_alpha_from_experiment(experiment_id)
        alphas = [alpha_value] * len(datasets)

        read_solutions = {solution: [] for solution in solutions}
        read_dataset_order = []

        for solution in solutions:
            for dt in datasets:

                read_path = """../system/results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                    experiment_id,
                    total_clients,
                    alphas,
                    datasets,
                    model_name,
                    fraction_fit,
                    number_of_rounds,
                    local_epochs,
                    train_test
                )

                read_dataset_order.append(dt)

                read_solutions[solution].append(
                    "{}{}_{}.csv".format(read_path, dt, solution)
                )

        df_exp = read_data(read_solutions, read_dataset_order)

        if df_exp is None:
            continue

        df_exp["Scenario"] = experiment_id

        if df_concat is None:
            df_concat = df_exp
        else:
            df_concat = pd.concat([df_concat, df_exp])

    return df_concat

def table_multi_scenarios(df, write_path, metric, t=None):

    if t is not None:
        df = df[df['Round (t)'].isin(t)]

    datasets = df["Dataset"].unique().tolist()
    solutions = df["Table"].unique().tolist()
    scenarios = df["Scenario"].unique().tolist()

    rows = []

    for dataset in datasets:
        for solution in solutions:

            row = {"Dataset": dataset, "Solution": solution}

            for scenario in scenarios:

                filtered = df.query(
                    f"Dataset == '{dataset}' and Table == '{solution}' and Scenario == '{scenario}'"
                )

                mean, ci_margin = mean_ci(filtered[metric])

                row[f"{scenario}_mean"] = mean
                row[f"{scenario}_ci"] = ci_margin

            rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results.set_index(["Dataset", "Solution"], inplace=True)

    baseline = "MultiFedAvg"

    # ===============================
    # 🔹 GANHO
    # ===============================

    for scenario in scenarios:
        for dataset in datasets:

            base_value = df_results.loc[(dataset, baseline)][f"{scenario}_mean"]

            for solution in solutions:

                mean_val = df_results.loc[(dataset, solution)][f"{scenario}_mean"]

                gain = ((mean_val - base_value) / base_value) * 100
                df_results.loc[(dataset, solution), f"{scenario}_gain"] = round(gain, 2)

    # ===============================
    # 🔹 NEGRITO COM IC
    # ===============================

    for scenario in scenarios:
        for dataset in datasets:

            subset = df_results.loc[dataset]

            means = subset[f"{scenario}_mean"]
            cis = subset[f"{scenario}_ci"]

            max_idx = means.idxmax()
            max_mean = means[max_idx]
            max_ci = cis[max_idx]

            max_lower = max_mean - max_ci
            max_upper = max_mean + max_ci

            for solution in subset.index:

                mean_val = subset.loc[solution][f"{scenario}_mean"]
                ci_val = subset.loc[solution][f"{scenario}_ci"]

                lower = mean_val - ci_val
                upper = mean_val + ci_val

                overlap = not (upper < max_lower or lower > max_upper)

                acc_str = f"{mean_val}±{ci_val}"

                gain_val = df_results.loc[(dataset, solution)][f"{scenario}_gain"]

                if gain_val >= 0:
                    gain_str = f"\\textuparrow{gain_val}\\%"
                else:
                    gain_str = f"\\textdownarrow{abs(gain_val)}\\%"

                if overlap:
                    acc_str = f"\\textbf{{{acc_str}}}"

                df_results.loc[(dataset, solution), scenario] = acc_str
                df_results.loc[(dataset, solution), f"{scenario}_gain_str"] = gain_str

    final_columns = []

    for scenario in scenarios:
        final_columns.append(scenario)
        final_columns.append(f"{scenario}_gain_str")

    df_final = df_results[final_columns]

    Path(write_path).mkdir(parents=True, exist_ok=True)

    filename = f"{write_path}latex_table_multi_{metric}.txt"

    latex = df_final.to_latex(escape=False)

    pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    print(df_final)
    print("Salvo em:", filename)

if __name__ == "__main__":

    experiment_ids = [
        "label_shift#0.1-1.0_sudden",
        "label_shift#0.1-10.0_sudden",
        "label_shift#1.0-0.1_sudden",
        "label_shift#1.0-10.0_sudden",
        "label_shift#10.0-0.1_sudden",
        "label_shift#10.0-1.0_sudden"
    ]

    total_clients = 40
    datasets = ["WISDM-W", "ImageNet10", "Foursquare"]
    model_name = ["gru", "CNN", "lstm"]
    fraction_fit = 0.3
    number_of_rounds = 100
    local_epochs = 1
    train_test = "test"

    solutions = [
        "MultiFedAvg+MFP_v2",
        "MultiFedAvg+MFP_v2_dh",
        "MultiFedAvg+MFP_v2_iti",
        "MultiFedAvg+MFP",
        "MultiFedAvg+FPD",
        "MultiFedAvg+FP",
        "DMA-FL",
        "AdaptiveFedAvg",
        "MultiFedAvg"
    ]

    write_path = "plots/MEFL/multi_experiments/"

    df = read_data_multi_experiments(
        experiment_ids,
        solutions,
        datasets,
        total_clients,
        model_name,
        fraction_fit,
        number_of_rounds,
        local_epochs,
        train_test
    )

    table_multi_scenarios(df, write_path, "Balanced accuracy (%)")
    table_multi_scenarios(df, write_path, "Accuracy (%)")

