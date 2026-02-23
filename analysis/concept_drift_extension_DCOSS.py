from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from numpy import trapz

import copy

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

def read_data(read_solutions, read_dataset_order, experiment_id, alpha_value):
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
                df["Concept Drift"] = experiment_id
                df["Alpha"] = alpha_value
                df["Solution"] = np.array([solution] * len(df))
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Dataset"] = np.array([dataset] * len(df))
                table_name = solution_strategy_version[solution]["Table"]

                # remover "MultiFedAvg+" apenas se não for exatamente "MultiFedAvg"
                if solution.startswith("MultiFedAvg+") and solution != "MultiFedAvg":
                    table_name = solution.replace("MultiFedAvg+", "")

                df["Table"] = np.array([table_name] * len(df))
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

def table_per_dataset(df, write_path, metric, solutions_order, ci=0.95):

    datasets = sorted(df["Dataset"].unique().tolist())
    alphas = sorted(df["Alpha"].unique().tolist())
    solutions = [
        df[df["Solution"] == s]["Table"].iloc[0]
        for s in solutions_order
        if s in df["Solution"].values
    ]

    Path(write_path).mkdir(parents=True, exist_ok=True)

    for dataset in datasets:

        rows_raw = {}

        # ==============================
        # 1️⃣ CALCULAR MÉDIA E CI
        # ==============================

        for solution in solutions:
            rows_raw[solution] = {}

            for alpha in alphas:

                filtered = df.query(
                    f"Dataset == '{dataset}' and Table == '{solution}' and Alpha == {alpha}"
                )

                mean, ci_margin = mean_ci(filtered[metric], ci=ci)

                rows_raw[solution][alpha] = {
                    "mean": mean,
                    "ci": ci_margin
                }

        # ==============================
        # 2️⃣ IDENTIFICAR MELHORES COM IC
        # ==============================

        for alpha in alphas:

            # coletar valores da coluna
            col_values = {
                sol: rows_raw[sol][alpha]
                for sol in solutions
            }

            # encontrar maior média
            best_sol = max(col_values, key=lambda x: col_values[x]["mean"])
            best_mean = col_values[best_sol]["mean"]
            best_ci = col_values[best_sol]["ci"]

            best_lower = best_mean - best_ci
            best_upper = best_mean + best_ci

            # verificar sobreposição
            for sol in solutions:
                mean_val = col_values[sol]["mean"]
                ci_val = col_values[sol]["ci"]

                lower = mean_val - ci_val
                upper = mean_val + ci_val

                overlap = not (upper < best_lower or lower > best_upper)

                rows_raw[sol][alpha]["bold"] = overlap

        # ==============================
        # 3️⃣ FORMATAR TABELA FINAL
        # ==============================

        rows_final = []

        for solution in solutions:
            safe_solution = solution.replace("_", r"\_")
            row = {"Solution": safe_solution}

            for alpha in alphas:

                mean_val = rows_raw[solution][alpha]["mean"]
                ci_val = rows_raw[solution][alpha]["ci"]
                bold = rows_raw[solution][alpha]["bold"]

                # CORREÇÃO AQUI
                value_str = f"{mean_val:.2f}$\\pm${ci_val:.2f}"

                if bold:
                    value_str = f"\\textbf{{{value_str}}}"

                row[f"$\\alpha={alpha}$"] = value_str

            rows_final.append(row)

        df_dataset = pd.DataFrame(rows_final)
        df_dataset.set_index("Solution", inplace=True)

        # ==============================
        # 4️⃣ GERAR LATEX
        # ==============================

        latex = df_dataset.to_latex(
            escape=False,
            column_format="l" + "c" * len(alphas),
            index_names=False
        )

        latex_complete = f"""
\\begin{{table}}[t]
\\centering
\\caption{{Concept Drift -- {dataset} - {metric.replace(" (%)", "(\%)")}}}
\\label{{tab:concept_drift_{dataset}_{metric.replace(' ','_').replace("_(%)", "")}}}
\\resizebox{{\\columnwidth}}{{!}}{{%
{latex}}}
\\end{{table}}
"""

        filename = f"{write_path}/latex_table_concept_dirft_{dataset}_{metric.replace(' ','_')}.tex".replace("_(%)", "")

        with open(filename, "w") as f:
            f.write(latex_complete)

        print(f"\nTabela salva para {dataset} em:")
        print(filename)

def mean_ci(values, ci=0.95):
    values = values.dropna().to_numpy()

    if len(values) == 0:
        return 0.00, 0.00

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
        margin = 0.00

    return round(mean, 2), round(margin, 2)

def extract_alpha_from_experiment(experiment_id):
    # concept_drift#0.1_sudden
    alpha_str = experiment_id.split("#")[1].split("_")[0]
    return float(alpha_str)



if __name__ == "__main__":

    concept_experiments = [
        "concept_drift#0.1_sudden",
        "concept_drift#1.0_sudden",
        "concept_drift#10.0_sudden"
    ]

    total_clients = 40
    dataset = ["WISDM-W", "ImageNet10", "Foursquare"]
    model_name = ["gru", "CNN", "lstm"]
    fraction_fit = 0.3
    number_of_rounds = 100
    local_epochs = 1
    train_test = "test"

    solutions = [
        "MultiFedAvg+MFP_v2", "MultiFedAvg+MFP_v2_dh",
        "MultiFedAvg+MFP_v2_iti", "MultiFedAvg+MFP",
        "MultiFedAvg+FPD", "MultiFedAvg+FP",
        "DMA-FL", "MultiFedAvg"
    ]

    # "AdaptiveFedAvg",

    df_all = None

    for experiment_id in concept_experiments:

        alpha_value = extract_alpha_from_experiment(experiment_id)
        alphas = [alpha_value] * len(dataset)

        read_solutions = {solution: [] for solution in solutions}
        read_dataset_order = []

        for solution in solutions:
            for dt in dataset:

                read_path = """../system/results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                    experiment_id,
                    total_clients,
                    alphas,
                    dataset,
                    model_name,
                    fraction_fit,
                    number_of_rounds,
                    local_epochs,
                    train_test
                )

                read_dataset_order.append(dt)
                read_solutions[solution].append(f"{read_path}{dt}_{solution}.csv")

        df, _ = read_data(read_solutions, read_dataset_order, experiment_id, alpha_value)

        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    write_path = "plots/MEFL/multi_experiments/"

    # table_per_dataset(df_all, write_path, "Balanced accuracy (%)")
    table_per_dataset(df_all, write_path, "Accuracy (%)", solutions)