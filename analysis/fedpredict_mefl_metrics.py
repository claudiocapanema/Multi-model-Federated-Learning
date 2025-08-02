from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns

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
        "MultiFedAvgRR": {"Strategy": "MultiFedAvgRR", "Version": "Original", "Table": "MultiFedAvgRR"}
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
                df["Dataset"] = np.array([dataset.replace("WISDM-W", "WISDM").replace("ImageNet", "ImageNet-15")] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                try:
                    df["\u03B1"] = df["Alpha"]
                except Exception as e:
                    print(e)
                try:
                    df["\u03B1"] = df["alpha"]
                except Exception as e:
                    print(e)
                try:
                    df["ps"] = df["Similarity"]
                except Exception as e:
                    print(e)
                try:
                    df["ps"] = df["similarity"]
                except Exception as e:
                    print(e)

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


def bar(df_metrics, df, base_dir, x, y_list, hue=None, style=None, ci=None, hue_order=None):

    datasets = df_metrics["Dataset"].unique().tolist()

    fig, axs = plt.subplots(2, 2, sharex='all', figsize=(7, 7))
    hue_order = ["MultiFedAvg", "MultiFedAvgRR"]

    for k in range(len(y_list)):

        if k <= 2:
         df_plot = df_metrics
        else:
            df_plot = df
        y = y_list[k]
        tipo = ""
        if k == 0:
            i, j = 0, 0
        elif k == 1:
            i, j = 0, 1
        elif k == 2:
            i, j = 1, 0
        elif k == 3:
            i, j = 1, 1
            tipo = "original"
        # , style=style, ci=ci
        bar_plot(df=df_plot, base_dir=base_dir, ax=axs[i, j],
                  file_name="""solutions_{}""".format(y_list), x_column=x, y_column=y,
                 hue="Dataset", title="", tipo=tipo, y_lim=True, y_max=1, palette=sns.color_palette())
        axs[i, j].set_ylim(0, 1.15)
        # axs[j].set_title(r"""Dataset: {}""".format(datasets[j]), size=10)

        if not (i == 0 and j == 1):
            axs[i, j].get_legend().remove()

    n_solutions = len(df_metrics["Version"].unique())
    print(n_solutions)

    # axs[1].legend(handles, labels, fontsize=9)

    # fig.suptitle("", fontsize=16)

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig(
        """{}alpha_dataset_round_{}_metrics.png""".format(base_dir, y_list), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}alpha_dataset_round_{}_metrics.svg""".format(base_dir, y_list), bbox_inches='tight',
        dpi=400)
    print("""{}alpha_dataset_round_{}_metrics.png""".format(base_dir, y_list))

def files(cd, total_clients, fraction_fit, number_of_rounds, local_epochs, train_test, dataset, model_name, fraction_new_clients, round_new_clients, alphas, version=""):
    read_solutions = {solution: [] for solution in solutions}
    read_dataset_order = []
    for solution in solutions:
        for dt in dataset:
            algo = dt + "_" + solution

            read_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                cd,
                0.1,
                0.1,
                total_clients,
                alphas,
                dataset,
                model_name,
                fraction_fit,
                number_of_rounds,
                local_epochs,
                train_test)
            read_dataset_order.append(dt)

            if version == "original":
                read_solutions[solution].append("""{}{}_{}.csv""".format(read_path, dt, solution))
            else:
                read_solutions[solution].append("""{}{}_{}_metrics.csv""".format(read_path, dt, solution))

    write_path = """plots/MEFL/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/concept_drift_experiment_id_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
        cd,
        fraction_new_clients,
        round_new_clients,
        total_clients,
        alphas,
        concept_drift_experiment_id,
        dataset,
        model_name,
        fraction_fit,
        number_of_rounds,
        local_epochs)

    print(read_solutions)

    return read_solutions, write_path, read_dataset_order


if __name__ == "__main__":
    concept_drift_experiment_id = 6
    cd = "false" if concept_drift_experiment_id == 0 else f"true_experiment_id_{concept_drift_experiment_id}"
    total_clients = 20
    # alphas = [0.1, 10.0]
    alphas = {6: [10.0, 10.0], 7: [0.1, 0.1], 8: [10.0, 10.0], 9: [0.1, 0.1], 10: [1.0, 1.0]}[concept_drift_experiment_id]
    # dataset = ["WISDM-W", "CIFAR10"]
    dataset = ["WISDM-W", "ImageNet"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    model_name = ["gru", "CNN"]
    fraction_fit = 0.3
    number_of_rounds = 100
    local_epochs = 1
    fraction_new_clients = alphas[0]
    round_new_clients = 0
    train_test = "test"
    # solutions = ["MultiFedAvg+MFP", "MultiFedAvg+FPD", "MultiFedAvg+FP", "MultiFedAvg", "MultiFedAvgRR"]
    solutions = ["MultiFedAvg+MFP"]

    read_solutions, write_path, read_dataset_order = files(cd, total_clients, fraction_fit, number_of_rounds,
                                                           local_epochs, train_test, dataset, model_name,
                                                           fraction_new_clients, round_new_clients, alphas)

    df_metrics, hue_order = read_data(read_solutions, read_dataset_order)

    read_solutions, write_path, read_dataset_order = files(cd, total_clients, fraction_fit, number_of_rounds,
                                                           local_epochs, train_test, dataset, model_name,
                                                           fraction_new_clients, round_new_clients, alphas, "original")

    df, hue_order = read_data(read_solutions, read_dataset_order)
    cp_rounds = [20, 50, 80]
    cp_window = []
    window = 5
    for i in range(len(cp_rounds)):
        cp_round = cp_rounds[i]
        cp_window += [round_ for round_ in range(cp_round, cp_round + window + 1)]
    df = df[df["Round (t)"].isin(cp_window)]
    print(df)

    y_list = ["fc", "il", "dh", "ps"]

    bar(df_metrics, df, write_path, x="\u03B1", y_list=y_list)
    bar(df_metrics, df, write_path, x="\u03B1", y_list=y_list)