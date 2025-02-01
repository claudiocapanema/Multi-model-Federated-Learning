from pathlib import Path
import numpy as np
import pandas as pd

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

def read_data(read_solutions, read_dataset_order, CM):

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
        "FedProto": {"Strategy": "FedProto", "Version": "Original", "Table": "FedProto"},
        "MultiFedEfficiency": {"Strategy": "MultiFedEfficiency", "Version": "Original", "Table": "MultiFedEfficiency"}}
    hue_order = []

    ref = r"results/concept_drift_False/new_clients_fraction_0_round_0/clients_40/alpha_['0.1', '10.0']/alpha_end_0.1_10.0/['ImageNet', 'ImageNet_v2']/concept_drift_rounds_0_0/['cnn_a', 'cnn_a']/fc_0.3/rounds_100/epochs_1"
    for solution in read_solutions:

        paths = read_solutions[solution]
        for i in range(len(paths)):
            # print(paths[i])

            try:
                dataset = read_dataset_order[i]
                path = paths[i]
                df = pd.read_csv(path).query("Round <= 100")
                metric = "Balanced accuracy (%)"
                df["Solution"] = np.array([solution] * len(df))
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Round (t)"] = df["Round"]
                df["Dataset"] = np.array([dataset] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                df["Round efficiency"] = np.array(((df[metric]/100) * (1-(df["# training clients"]).to_numpy()/CM)))
                df["Cumulative efficiency"] = np.cumsum(np.array(((df[metric]/100) * (1-(df["# training clients"]).to_numpy()/CM))))/100
                print(df["Round efficiency"])
                # exit()
                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution_strategy_version[solution]["Strategy"]
                if strategy not in hue_order:
                    hue_order.append(strategy)

                    # print("\n####### \nLido\n", path.replace("../", ""), "\n")
                    # print(ref, "\n")
            except Exception as e:
                print("\n####### \nFaltando\n", paths[i].replace("../", ""), "\n")
                print(e)
                # print(ref,  "\n")

    return df_concat, hue_order


def line_datasets(df, base_dir, x, y, hue=None, style=None, ci=None, hue_order=None, y_max=100):

    datasets = df["Dataset"].unique().tolist()
    # datasets = ["ImageNet", "ImageNet"]
    alphas = df["Alpha"].unique().tolist()
    # df["Strategy"] = np.array([i.replace("Multi", "") for i in df["Strategy"].tolist()])

    fig, axs = plt.subplots(len(datasets), sharex='all', figsize=(12, 9))

    # for i in range(len(alphas)):
    for i in range(len(datasets)):

        df_plot = df[df["Dataset"] == datasets[i]]
        # df_plot = df_plot[df_plot["Alpha"] == alphas[i]]
        # print(df_plot)
        # exit()

        line_plot(df=df_plot, base_dir=base_dir, ax=axs[i],
                  file_name="""solutions_{}""".format(datasets), x_column=x, y_column=y,
                  hue=hue, style=style, ci=ci, title="", tipo=None, y_lim=True, y_max=y_max)
        axs[i].set_title(r"""Dataset: {}""".format(datasets[i]), fontweight="bold", size=9)

            # if [i] != [0, 1]:
            #     axs[i].get_legend().remove()

    # lines_labels = [axs[0, 0].get_legend_handles_labels()]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # colors = []
    # for i in range(len(lines)):
    #     color = lines[i].get_color()
    #     colors.append(color)
    #     ls = lines[i].get_ls()
    #     if ls not in ["o"]:
    #         ls = "o"
    # markers = ["", "-", "--"]
    #
    # f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    # handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
    # handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(3)]
    # for i in range(len(alphas)):
    #     for j in range(len(datasets)):
    #         axs[i, j].legend(handles, labels, fontsize=7)
    #
    # # fig.suptitle("", fontsize=16)

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

def line(df, base_dir, x, y, hue=None, style=None, ci=None, hue_order=None, y_max=100):

    datasets = df["Dataset"].unique().tolist()
    # datasets = ["ImageNet", "ImageNet"]
    alphas = df["Alpha"].unique().tolist()
    # df["Strategy"] = np.array([i.replace("Multi", "") for i in df["Strategy"].tolist()])

    plt.figure()
    df_plot = df[["Strategy", "Round (t)", "Accuracy (%)", "Balanced accuracy (%)", "# training clients", "Round efficiency", "Cumulative efficiency"]].groupby(["Round (t)", "Strategy"]).mean().reset_index()
    # df_plot = df_plot[df_plot["Alpha"] == alphas[i]]
    # print(df_plot)
    # exit()
    print("agg")
    print(df_plot)

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    line_plot(df=df_plot, base_dir=base_dir,
              file_name="""alpha_dataset_round_global_{}""".format(y), x_column=x, y_column=y,
              hue=hue, hue_order=["MultiFedEfficiency", "MultiFedAvg"], style=style, ci=ci, title="", tipo=None, y_lim=True, y_max=y_max)


if __name__ == "__main__":
    cd = "False"
    num_clients = 60
    # dataset = ["EMNIST", "CIFAR10", "GTSRB"]
    # dataset = ["Gowalla", "ImageNet", "WISDM-W"]
    dataset = ["ImageNet", "WISDM-W", "Gowalla"]
    dataset_alpha = {"Gowalla": 0.1, "ImageNet": 1.0, "WISDM-W": 10.0}
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    models_names = [{"ImageNet": "cnn_a", "Gowalla": "lstm", "WISDM-W": "gru"}[i] for i in dataset]
    join_ratio = 0.3
    global_rounds = 100
    local_epochs = 1
    fraction_new_clients = 0
    round_new_clients = 0
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg",
    #              "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedAvgGlobalModelEval",
    #              "MultiFedYogiWithFedPredict", "MultiFedYogi", "MultiFedYogiGlobalModelEval", "MultiFedPer"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg", "MultiFedAvgGlobalModelEval",
    #              "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedPer"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg", "MultiFedYogiWithFedPredict", "MultiFedYogi", "MultiFedKD","MultiFedKDWithFedPredict", "MultiFedPer"]
    solutions = ["MultiFedEfficiency", "MultiFedAvg"]
    tw = {"MultiFedEfficiency": "_tw_12_df_0.5", "MultiFedAvg": ""}

    read_solutions = {solution: [] for solution in solutions}
    read_dataset_order = []
    alphas = [str(i) for i in dataset_alpha.values()]
    alphas_end = alphas
    alphas_end_str = ""
    for alpha in alphas_end:
        alphas_end_str += str(alpha) + "_"
    alphas_end_str = alphas_end_str[:-1]
    for solution in solutions:
        for dt in dataset:
            read_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/test/""".format(
                cd,
                fraction_new_clients,
                round_new_clients,
                num_clients,
                str(alphas),
                alphas_end,
                dataset,
                0,
                0,
                models_names,
                join_ratio,
                global_rounds,
                local_epochs)
            read_dataset_order.append(dt)

            read_solutions[solution].append("""{}{}_{}_test_0{}.csv""".format(read_path, dt, solution, tw[solution]))

    write_path = """plots/multimodel/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
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
        local_epochs,
        tw["MultiFedEfficiency"][1:])

    df, hue_order = read_data(read_solutions, read_dataset_order, int(join_ratio * num_clients))
    print(df)

    # line(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)
    # line(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy", style="Version", hue_order=hue_order)
    line_datasets(df, write_path, x="Round (t)", y="Round efficiency", hue="Strategy", y_max=1)
    line_datasets(df, write_path, x="Round (t)", y="Round efficiency", hue="Strategy", y_max=1)
    line_datasets(df, write_path, x="Round (t)", y="Cumulative efficiency", hue="Strategy", y_max=0.7)
    line_datasets(df, write_path, x="Round (t)", y="Cumulative efficiency", hue="Strategy", y_max=0.7)
    line_datasets(df, write_path, x="Round (t)", y="Balanced accuracy (%)", hue="Strategy")
    line_datasets(df, write_path, x="Round (t)", y="Balanced accuracy (%)", hue="Strategy")
    line_datasets(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy")
    line_datasets(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy")
    line_datasets(df, write_path, x="Round (t)", y="# training clients", hue="Strategy", y_max=15)
    line_datasets(df, write_path, x="Round (t)", y="# training clients", hue="Strategy", y_max=15)

    line(df, write_path, x="Round (t)", y="Round efficiency", hue="Strategy", y_max=1)
    line(df, write_path, x="Round (t)", y="Round efficiency", hue="Strategy", y_max=1)
    line(df, write_path, x="Round (t)", y="Cumulative efficiency", hue="Strategy", y_max=0.7)
    line(df, write_path, x="Round (t)", y="Cumulative efficiency", hue="Strategy", y_max=0.7)
    line(df, write_path, x="Round (t)", y="Balanced accuracy (%)", hue="Strategy")
    line(df, write_path, x="Round (t)", y="Balanced accuracy (%)", hue="Strategy")
    line(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy")
    line(df, write_path, x="Round (t)", y="Accuracy (%)", hue="Strategy")
    line(df, write_path, x="Round (t)", y="# training clients", hue="Strategy", y_max=15)
    line(df, write_path, x="Round (t)", y="# training clients", hue="Strategy", y_max=15)