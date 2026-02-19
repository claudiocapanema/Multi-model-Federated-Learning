from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base_plots import line_plot


# ===============================
# 🔹 EXTRAIR ALPHA DO NOME
# ===============================

def extract_alpha_from_experiment(experiment_id):
    """
    Extrai o primeiro alpha do experiment_id.
    Exemplo:
    label_shift#0.1-1.0_sudden -> 0.1
    """
    alpha_part = experiment_id.split("#")[1]
    first_alpha = alpha_part.split("-")[0]
    return float(first_alpha)

def read_data(read_solutions, read_dataset_order):

    df_concat = None

    solution_strategy_version = {
        "MultiFedAvg+MFP_v2": {"Strategy": "MultiFedAvg", "Version": "MFP_v2", "Table": "MultiFedAvg+MFP_v2"},
        "MultiFedAvg+MFP_v2_dh": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_dh", "Table": "MultiFedAvg+MFP_v2_dh"},
        "MultiFedAvg+MFP_v2_iti": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_iti", "Table": "MultiFedAvg+MFP_v2_iti"},
        "MultiFedAvg+MFP": {"Strategy": "MultiFedAvg", "Version": "MFP", "Table": "MultiFedAvg+MFP"},
        "MultiFedAvg+FPD": {"Strategy": "MultiFedAvg", "Version": "FPD", "Table": "MultiFedAvg+FPD"},
        "MultiFedAvg+FP": {"Strategy": "MultiFedAvg", "Version": "FP", "Table": "MultiFedAvg+FP"},
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
                df["Dataset"] = dataset.replace("WISDM-W", "WISDM").replace("ImageNet10", "ImageNet-10")
                df["Strategy"] = solution_strategy_version[solution]["Strategy"]
                df["Version"] = solution_strategy_version[solution]["Version"]
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

def line_subplots_by_scenario(df,
                              base_dir,
                              x,
                              y,
                              hue="Table",
                              hue_order=None,
                              y_max=100,
                              ci=None):

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    datasets = df["Dataset"].unique().tolist()
    scenarios = df["Scenario"].unique().tolist()

    n_scenarios = len(scenarios)

    for dataset in datasets:

        df_dataset = df[df["Dataset"] == dataset]

        fig, axs = plt.subplots(
            n_scenarios,
            1,
            figsize=(8, 3 * n_scenarios),
            sharex=True
        )

        if n_scenarios == 1:
            axs = [axs]

        for i, scenario in enumerate(scenarios):

            df_plot = df_dataset[df_dataset["Scenario"] == scenario]

            line_plot(
                df=df_plot,
                base_dir=base_dir,
                ax=axs[i],
                file_name=f"{dataset}_{scenario}_{y}",
                x_column=x,
                y_column=y,
                hue=hue,
                hue_order=hue_order,
                title="",
                tipo=None,
                y_lim=True,
                y_max=y_max,
                ci=ci
            )

            axs[i].set_title(f"Scenario: {scenario}", fontsize=9)

            if i != 0:
                axs[i].get_legend().remove()

        fig.suptitle(f"{dataset} - {y}", fontsize=12)

        plt.tight_layout()

        fig.savefig(
            f"{base_dir}/{dataset}_{y}_subplots.png",
            dpi=400,
            bbox_inches="tight"
        )

        fig.savefig(
            f"{base_dir}/{dataset}_{y}_subplots.svg",
            dpi=400,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Salvo: {base_dir}/{dataset}_{y}_subplots.png")

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

    df = df[['Round (t)', 'Fraction fit', 'Alpha',
             'Solution', 'Accuracy (%)',
             'Balanced accuracy (%)',
             'Dataset', 'Strategy',
             'Version', 'Table', 'Scenario']]

    line_subplots_by_scenario(
        df,
        write_path,
        x="Round (t)",
        y="Accuracy (%)",
        hue="Table",
        hue_order=solutions,
        ci=None
    )

    line_subplots_by_scenario(
        df,
        write_path,
        x="Round (t)",
        y="Balanced accuracy (%)",
        hue="Table",
        hue_order=solutions,
        ci=None
    )

    print("Plots multi-experimentos finalizados.")
