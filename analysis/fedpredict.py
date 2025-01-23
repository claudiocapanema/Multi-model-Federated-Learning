from pathlib import Path
import numpy as np
import pandas as pd

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

def read_data(read_solutions, read_dataset_order):

    df_concat = None
    solution_strategy_version = {
        "MultiFedAvgWithFedPredict": {"Strategy": "MultiFedAvg", "Version": "FP", "Table": "FedAvg+FP"},
        "MultiFedAvg": {"Strategy": "MultiFedAvg", "Version": "Original", "Table": "FedAvg"},
        "MultiFedAvgGlobalModelEval": {"Strategy": "MultiFedAvgGlobalModelEval", "Version": "Original",
                                       "Table": "FedAvgGlobalModelEval"},
        "MultiFedAvgGlobalModelEvalWithFedPredict": {"Strategy": "MultiFedAvgGlobalModelEval", "Version": "FP",
                                                     "Table": "MultiFedAvgGlobalModelEvalWithFedPredict"},
        "MultiFedPer": {"Strategy": "MultiFedPer", "Version": "Original", "Table": "FedPer"},
        "MultiFedYogi": {"Strategy": "MultiFedYogi", "Version": "Original", "Table": "FedYogi"},
        "MultiFedYogiWithFedPredict": {"Strategy": "MultiFedYogi", "Version": "FP", "Table": "FedYogi+FP"},
        "MultiFedYogiGlobalModelEval": {"Strategy": "MultiFedYogiGlobalModelEval", "Version": "Original",
                                        "Table": "FedYogiGlobalModelEval"},
        "MultiFedYogiGlobalModelEvalWithFedPredict": {"Strategy": "MultiFedYogiGlobalModelEval", "Version": "FP",
                                                      "Table": "FedYogiGlobalModelEvalWithFedPredict"},
        "MultiFedKD": {"Strategy": "MultiFedKD", "Version": "Original", "Table": "FedKD"},
        "MultiFedKDWithFedPredict": {"Strategy": "MultiFedKD", "Version": "FP", "Table": "FedKD+FP"},
        "FedProto": {"Strategy": "FedProto", "Version": "Original", "Table": "FedProto"}}
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
                df["Round (t)"] = df["Round"]
                df["Dataset"] = np.array([dataset] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))

                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution_strategy_version[solution]["Strategy"]
                if strategy not in hue_order:
                    hue_order.append(strategy)
            except:
                print("\n######### \nFaltando", paths[i])

    return df_concat, hue_order


def line(df, base_dir, x, y, hue=None, style=None, ci=None, hue_order=None):

    datasets = df["Dataset"].unique().tolist()
    # datasets = ["ImageNet", "ImageNet"]
    alphas = df["Alpha"].unique().tolist()
    df["Strategy"] = np.array([i.replace("Multi", "") for i in df["Strategy"].tolist()])

    fig, axs = plt.subplots(len(alphas), len(datasets), sharex='all', figsize=(12, 9))

    for i in range(len(alphas)):
        for j in range(len(datasets)):

            df_plot = df[df["Dataset"] == datasets[j]]
            df_plot = df_plot[df_plot["Alpha"] == alphas[i]]

            line_plot(df=df_plot, base_dir=base_dir, ax=axs[i, j],
                      file_name="""solutions_{}""".format(datasets), x_column=x, y_column=y,
                      hue=hue, style=style, ci=ci, title="", tipo=None, y_lim=True, y_max=100)
            axs[i, j].set_title(r"""Dataset: {}; $\alpha$={}""".format(datasets[j], alphas[i]), fontweight="bold", size=9)

            if [i, j] != [0, 1]:
                axs[i, j].get_legend().remove()

    lines_labels = [axs[0, 0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    colors = []
    for i in range(len(lines)):
        color = lines[i].get_color()
        colors.append(color)
        ls = lines[i].get_ls()
        if ls not in ["o"]:
            ls = "o"
    markers = ["", "-", "--"]

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
    handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(3)]
    for i in range(len(alphas)):
        for j in range(len(datasets)):
            axs[i, j].legend(handles, labels, fontsize=7)

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
    cd = "False"
    num_clients = 20
    alphas = [0.1, 1.0, 10.0]
    dataset = ["EMNIST", "CIFAR10", "GTSRB"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    models_names = ["cnn_a"]
    join_ratio = 0.3
    global_rounds = 100
    local_epochs = 1
    fraction_new_clients = 0.3
    round_new_clients = 70
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg",
    #              "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedAvgGlobalModelEval",
    #              "MultiFedYogiWithFedPredict", "MultiFedYogi", "MultiFedYogiGlobalModelEval", "MultiFedPer"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg", "MultiFedAvgGlobalModelEval",
    #              "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedPer"]
    solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg", "MultiFedYogiWithFedPredict", "MultiFedYogi", "MultiFedKD","MultiFedKDWithFedPredict", "MultiFedPer"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg"]

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

    df, hue_order = read_data(read_solutions, read_dataset_order)
    line(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)
    line(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)
    line(df, write_path, x="Round (t)", y="Balanced accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)
    line(df, write_path, x="Round (t)", y="Balanced accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)