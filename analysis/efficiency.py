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
        "MultiFedEfficiency": {"Strategy": "MultiFedEfficiency", "Version": "Original", "Table": "MultiFedEfficiency"},
        "MultiFedAvgRR": {"Strategy": "MultiFedAvgRR", "Version": "Original", "Table": "MultiFedAvgRR"},
        "FedFairMMFL": {"Strategy": "FedFairMMFL", "Version": "Original", "Table": "FedFairMMFL"}
    }
    hue_order = []

    ref = r"results/concept_drift_False/new_clients_fraction_0_round_0/clients_40/alpha_['0.1', '10.0']/alpha_end_0.1_10.0/['ImageNet', 'ImageNet_v2']/concept_drift_rounds_0_0/['cnn_a', 'cnn_a']/fc_0.3/rounds_100/epochs_1"
    for solution in read_solutions:

        paths = read_solutions[solution]["file"]
        tws = read_solutions[solution]["tw"]
        rfs = read_solutions[solution]["rf"]
        # if solution == "MultiFedAvg":
        #     print(paths)
        #     exit()
        for i in range(len(paths)):
            # print(paths[i])
            tw = tws[i]
            rf = rfs[i]
            config = ""
            if solution == "MultiFedEfficiency":
                if len(tw) == 0:
                    tw = "0"
                if len(rf) == 0:
                    rf = "0"
                config = " TW={};RF={}".format(tw, rf)

            try:
                dataset = read_dataset_order[i]
                path = paths[i]
                df = pd.read_csv(path).query("Round <= 100")
                metric = "Balanced accuracy (%)"
                df["Solution"] = np.array([solution] * len(df))
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Round (t)"] = df["Round"]
                df["Dataset"] = np.array([dataset.replace("WISDM-W", "WISDM")] * len(df))
                df["Strategy"] = np.array([solution + config] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                df["Round efficiency"] = np.array(((df[metric]/100) * (1-(df["# training clients"]).to_numpy()/CM)))
                df["Cumulative efficiency"] = np.cumsum(((df[metric]/100) * (1-(df["# training clients"]).to_numpy()/CM)))/100
                df["Comulative communication cost (MB)"] = np.cumsum(((df["model size"] * df["# training clients"]) / 1000000).to_numpy())
                print(df["Round efficiency"])
                # exit()
                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution + config
                if strategy not in hue_order:
                    hue_order.append(strategy)

                    # print("\n####### \nLido\n", path.replace("../", ""), "\n")
                    # print(ref, "\n")
            except Exception as e:
                print("\n####### \nFaltando\n", paths[i].replace("../", ""), "\n")
                print(e)
                # print(ref,  "\n")

    # print(df_concat.query("Strategy == 'MultiFedAvg' and Dataset == 'WISDM'"))
    # exit()


    return df_concat, hue_order


def line_datasets(df, base_dir, x, y, hue=None, style=None, ci=None, hue_order=None, y_max=100):

    datasets = df["Dataset"].unique().tolist()
    # datasets = ["ImageNet", "ImageNet"]
    alphas = df["Alpha"].unique().tolist()
    # df["Strategy"] = np.array([i.replace("Multi", "") for i in df["Strategy"].tolist()])

    plt.plot()
    fig, axs = plt.subplots(len(datasets), sharex='all', figsize=(6, 9))

    # for i in range(len(alphas)):
    for i in range(len(datasets)):

        df_plot = df[df["Dataset"] == datasets[i]]
        # if y == "Comulative communication cost (MB)":
        #     if datasets[i] in ["WISDM"]:
        #         y_max = 7
        #     elif datasets[i] in ["Gowalla"]:
        #         y_max = 0.5
        #     elif datasets[i] in ["ImageNet"]:
        #         y_max = 2500
        keep_legend_idxs = [0]
        if "ccuracy" in y:
            y_max = 100
            y_lim = True
        else:
            y_lim = False

        # df_plot = df_plot[df_plot["Alpha"] == alphas[i]]
        # print(df_plot)
        # exit()

        model = datasets[i].replace("ImageNet", "CNN").replace("WISDM", "GRU").replace("Gowalla", "LSTM")
        alpha = datasets[i].replace("ImageNet", "0.1").replace("WISDM", "1.0").replace("Gowalla", "10.0")

        line_plot(df=df_plot, base_dir=base_dir, ax=axs[i],
                  file_name="""solutions_{}""".format(datasets), x_column=x, y_column=y,
                  hue=hue, style=style, ci=ci, title="", tipo=None, y_lim=y_lim, y_max=y_max)
        axs[i].set_title(r"""Model: {}; Dataset: {} ($\alpha$: {})""".format(model, datasets[i], alpha), size=10)
        if i not in keep_legend_idxs:
            axs[i].get_legend().remove()
        else:
            axs[i].legend(fontsize=10)

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
    df_plot = df[["Strategy", "Round (t)", "Accuracy (%)", "Balanced accuracy (%)", "# training clients", "Round efficiency", "Cumulative efficiency", "Comulative communication cost (MB)"]].groupby(["Round (t)", "Strategy"]).mean().reset_index()
    # df_plot = df_plot[df_plot["Alpha"] == alphas[i]]
    # print(df_plot)
    # exit()
    print("agg")
    print(df_plot)

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    line_plot(df=df_plot, base_dir=base_dir,
              file_name="""alpha_dataset_round_global_{}""".format(y), x_column=x, y_column=y,
              hue=hue, hue_order=None, style=style, ci=ci, title="", tipo=None, y_lim=True, y_max=y_max)


if __name__ == "__main__":
    cd = "False"
    num_clients = 60
    # dataset = ["EMNIST", "CIFAR10", "GTSRB"]
    # dataset = ["Gowalla", "ImageNet", "WISDM-W"]
    dataset = ["ImageNet", "WISDM-W", "Gowalla"]
    # dataset = ["WISDM-W", "Gowalla", "ImageNet"]
    # dataset_alpha = {"Gowalla": 0.1, "ImageNet": 1.0, "WISDM-W": 10.0}
    dataset_alpha = [str({"Gowalla": 10.0, "ImageNet": 0.1, "WISDM-W": 1.0}[i]) for i in dataset]
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
    solutions = ["MultiFedEfficiency", "MultiFedAvg", "MultiFedAvgRR", "FedFairMMFL"]
    tw = {"MultiFedEfficiency": ["8", "15", "20"], "MultiFedAvg": [""], "MultiFedAvgRR": [""], "FedFairMMFL": [""]}
    rf = {"MultiFedEfficiency": ["0.0"], "MultiFedAvg": [""], "MultiFedAvgRR": [""], "FedFairMMFL": [""]}
    include_tw = True
    include_rf = False
    # tw = {"MultiFedEfficiency": ["15"], "MultiFedAvg": [""], "MultiFedAvgRR": [""], "FedFairMMFL": [""]}
    # rf = {"MultiFedEfficiency": ["0.0", "0.5", "1.0"], "MultiFedAvg": [""], "MultiFedAvgRR": [""], "FedFairMMFL": [""]}
    # include_tw = False
    # include_rf = True

    read_solutions = {solution: {"file": [], "tw": [], "rf": []} for solution in solutions}
    read_dataset_order = []
    alphas = [str(i) for i in dataset_alpha]
    alphas_end = alphas
    alphas_end_str = ""
    for alpha in alphas_end:
        alphas_end_str += str(alpha) + "_"
    alphas_end_str = alphas_end_str[:-1]
    for solution in solutions:
        for tw_config in tw[solution]:
            for rf_config in rf[solution]:
                for dt in dataset:
                    config = ""
                    if len(tw_config) > 0:
                        config += """_tw_{}""".format(tw_config)
                    if len(rf_config) > 0:
                        config += """_df_{}""".format(rf_config)

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

                    read_solutions[solution]["file"].append("""{}{}_{}_test_0{}_weighted.csv""".format(read_path, dt, solution, config))
                    read_solutions[solution]["tw"].append(tw_config)
                    read_solutions[solution]["rf"].append(rf_config)

    # for solution in solutions:
    #     for f in read_solutions[solution]["file"]:
    #         print(f)
    # exit()

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
        """_tw_{}_rf{}""".format(tw["MultiFedEfficiency"], rf["MultiFedEfficiency"]))

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
    line_datasets(df, write_path, x="Round (t)", y="Comulative communication cost (MB)", hue="Strategy", y_max=2500)
    line_datasets(df, write_path, x="Round (t)", y="Comulative communication cost (MB)", hue="Strategy", y_max=2500)

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
    line(df, write_path, x="Round (t)", y="Comulative communication cost (MB)", hue="Strategy", y_max=2500)
    line(df, write_path, x="Round (t)", y="Comulative communication cost (MB)", hue="Strategy", y_max=2500)