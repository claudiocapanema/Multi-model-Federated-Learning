from pathlib import Path
import numpy as np
import pandas as pd

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
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Dataset"] = np.array([dataset.replace("WISDM-W", "WISDM").replace("ImageNet", "ImageNet-15")] * len(df))
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


def line(df, base_dir, x, y, hue=None, style=None, ci=None, hue_order=None):

    datasets = df["Dataset"].unique().tolist()

    fig, axs = plt.subplots(len(datasets), sharex='all', figsize=(9, 6))
    hue_order = ["MultiFedAvg", "MultiFedAvgRR"]

    for j in range(len(datasets)):

        df_plot = df[df["Dataset"] == datasets[j]]

        line_plot(df=df_plot, base_dir=base_dir, ax=axs[j],
                  file_name="""solutions_{}""".format(datasets), x_column=x, y_column=y,
                  hue=hue, hue_order=hue_order, style=style, ci=ci, title="", tipo=None, y_lim=True, y_max=100)
        axs[j].set_title(r"""Dataset: {}""".format(datasets[j]), size=10)

        if j == 1:
            axs[j].get_legend().remove()

    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    colors = []
    for i in range(len(lines)):
        color = lines[i].get_color()
        colors.append(color)
        ls = lines[i].get_ls()
        if ls not in ["o"]:
            ls = "o"

    n_solutions = len(df["Version"].unique())
    print(n_solutions)
    # exit()
    markers = {3: ["", "-", "--", "dotted"], 4: ["", "-", "--", "-.", "dotted"]}[n_solutions]

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
    handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(len(markers))]
    axs[0].legend(handles, labels, fontsize=9)
    # axs[1].legend(handles, labels, fontsize=9)

    # fig.suptitle("", fontsize=16)

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig(
        """{}alpha_dataset_round_{}.png""".format(base_dir, y), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}alpha_dataset_round_{}.svg""".format(base_dir, y), bbox_inches='tight',
        dpi=400)
    print("""{}alpha_dataset_round_{}.png""".format(base_dir, y))


if __name__ == "__main__":

    concept_drift_experiment_id = 8
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
    solutions = ["MultiFedAvg+MFP", "MultiFedAvg+FPD", "MultiFedAvg"]

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

            read_solutions[solution].append("""{}{}_{}.csv""".format(read_path, dt, solution))

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

    df, hue_order = read_data(read_solutions, read_dataset_order)
    print(df)

    line(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)
    line(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)
    line(df, write_path, x="Round (t)", y="Balanced accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)
    line(df, write_path, x="Round (t)", y="Balanced accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)