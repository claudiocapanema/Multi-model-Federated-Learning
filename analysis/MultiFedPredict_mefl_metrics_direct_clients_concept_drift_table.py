import copy
from pathlib import Path
import numpy as np
import sys
import pandas as pd
import seaborn as sns
from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
import torch
from numpy.linalg import norm
import csv
import os

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

def read_data(alphas, datasets, total_clients):

    filename = f"clients_{total_clients}_datasets_{datasets}_alphas_{alphas}_metrics_clients_concept_drift.csv"

    if os.path.exists(filename):
        print("O arquivo existe!")
        df = pd.read_csv(filename)
        return df

    print("O arquivo não existe!")

    n_classes = [
        {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43,
         'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
         "ImageNet10": 10, "ImageNet_v2": 15,
         "Gowalla": 7, "wikitext": 30,
         "Foursquare": 10}[dataset]
        for dataset in datasets
    ]

    ME = len(datasets)

    # TRANSIÇÕES ENTRE ALPHAS (igual ao primeiro algoritmo)
    alpha_pairs = [(0.1, 1.0), (0.1, 10.0), (1.0, 10.0)]
    alpha_pairs_str = [f"{a}<->{b}" for a, b in alpha_pairs]

    clients_train_loader = {
        cid: {alpha: {me: None for me in range(ME)} for alpha in alphas}
        for cid in range(1, total_clients + 1)
    }

    rows = []

    # -----------------------
    # Carrega todos os dados
    # -----------------------
    for cid in range(1, total_clients + 1):

        for alpha in alphas:

            for me in range(ME):

                clients_train_loader[cid][alpha][me], _ = load_data(
                    dataset_name=datasets[me],
                    alpha=alpha,
                    data_sampling_percentage=0.8,
                    partition_id=cid,
                    num_partitions=total_clients + 1,
                    batch_size=32,
                )

    # -----------------------
    # Calcula concept drift
    # -----------------------
    for me in range(ME):

        for cid in range(1, total_clients + 1):

            drift_dict = {}

            for alpha_a, alpha_b in alpha_pairs:

                # janela t0
                p_ME_a, _, _ = get_datasets_metrics(
                    clients_train_loader[cid][alpha_a],
                    ME,
                    n_classes,
                    concept_drift_window=[0] * ME
                )

                # janela t1
                p_ME_b, _, _ = get_datasets_metrics(
                    clients_train_loader[cid][alpha_b],
                    ME,
                    n_classes,
                    concept_drift_window=[1] * ME
                )

                drift_value = 1 - cosine_similarity(
                    p_ME_a[me],
                    p_ME_b[me]
                )

                drift_dict[f"{alpha_a}<->{alpha_b}"] = round(drift_value, 4)

            row = [
                cid,
                me,
                datasets[me].replace("WISDM-W", "WISDM")
                            .replace("ImageNet10", "ImageNet-10"),
                drift_dict["0.1<->1.0"],
                drift_dict["0.1<->10.0"],
                drift_dict["1.0<->10.0"],
            ]

            rows.append(row)

    df = pd.DataFrame(
        data=rows,
        columns=[
            "cid",
            "me",
            "Dataset",
            "0.1<->1.0",
            "0.1<->10.0",
            "1.0<->10.0"
        ]
    )

    df.to_csv(filename, index=False)

    return df

def get_datasets_metrics(trainloader, ME, n_classes, concept_drift_window=None):

    try:
        p_ME = []
        fc_ME = []
        il_ME = []
        for me in range(ME):
            labels_me = []
            n_classes_me = n_classes[me]
            p_me = {i: 0 for i in range(n_classes_me)}
            with (torch.no_grad()):
                for batch in trainloader[me]:
                    labels = batch["label"]
                    labels = labels.to("cuda:0")

                    if concept_drift_window is not None:
                        labels = (labels + concept_drift_window[me])
                        labels = labels % n_classes[me]
                    labels = labels.detach().cpu().numpy()
                    labels_me += labels.tolist()
                unique, count = np.unique(labels_me, return_counts=True)
                data_unique_count_dict = dict(zip(np.array(unique).tolist(), np.array(count).tolist()))
                for label in data_unique_count_dict:
                    p_me[label] = data_unique_count_dict[label]
                p_me = np.array(list(p_me.values()))
                fc_me = len(np.argwhere(p_me > 0)) / n_classes_me
                print("fc: ", fc_me)
                il_me = len(np.argwhere(p_me < np.sum(p_me) / n_classes_me)) / n_classes_me
                p_me = p_me / np.sum(p_me)
                p_ME.append(p_me)
                fc_ME.append(fc_me)
                il_ME.append(il_me)
                # print(f"p_me {p_me} fc_me {fc_me} il_me {il_me} model {me} client {client_id}")
        return p_ME, fc_ME, il_ME
    except Exception as e:
       print("_get_datasets_metrics error")
       print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def cosine_similarity(p_1, p_2):

    # compute cosine similarity
    try:
        p_1_size = np.array(p_1).shape
        p_2_size = np.array(p_2).shape
        if p_1_size != p_2_size:
            raise Exception(f"Input sizes have different shapes: {p_1_size} and {p_2_size}. Please check your input data.")

        return np.dot(p_1, p_2) / (norm(p_1) * norm(p_2))
    except Exception as e:
        print("cosine_similairty error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
        
def write_header(self, filename, header, mode):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode) as server_log_file:
            writer = csv.writer(server_log_file)
            writer.writerow(header)
    except Exception as e:
        print("_write_header error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def write_outputs(self, filename, data, mode='a'):
    try:
        for i in range(len(data)):
            for j in range(len(data[i])):
                element = data[i][j]
                if type(element) == float:
                    element = round(element, 6)
                    data[i][j] = element
        with open(filename, 'a') as server_log_file:
            writer = csv.writer(server_log_file)
            writer.writerows(data)
    except Exception as e:
        print("_write_outputs error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def latex_general_metrics_table(df, base_dir):

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["Dataset"].unique())
    alphas = [0.1, 1.0, 10.0]
    metrics = ["fc", "il", "dh"]

    grouped = (
        df.groupby(["Dataset", "α"])[metrics]
        .agg(["mean", "std", "count"])
    )

    tex_path = f"{base_dir}/general_metrics_concept_drift.tex"

    with open(tex_path, "w") as f:

        f.write("\\begin{figure}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{General Metrics under Concept Drift (mean $\\pm$ 95\\% CI)}\n")
        f.write("\\label{tab:general_metrics_cd}\n")

        col_format = "ll" + "c" * len(datasets)
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\toprule\n")

        header = ["$\\alpha$", "Metric"] + datasets
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")

        for alpha in alphas:
            for metric in metrics:

                row = [str(alpha), metric]

                for dataset in datasets:

                    try:
                        mean = grouped.loc[(dataset, alpha)][(metric, "mean")]
                        std = grouped.loc[(dataset, alpha)][(metric, "std")]
                        n = grouped.loc[(dataset, alpha)][(metric, "count")]

                        ci = 1.96 * (std / np.sqrt(n))

                        value = f"{mean:.2f} $\\pm$ {ci:.2f}"
                        row.append(value)

                    except:
                        row.append("-")

                f.write(" & ".join(row) + " \\\\\n")

            f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{figure}\n")

    print(f"Tabela salva em {tex_path}")

def latex_concept_drift_table(df, base_dir):

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["Dataset"].unique())
    alpha_pairs = ["0.1<->1.0", "0.1<->10.0", "1.0<->10.0"]

    # Reorganiza no mesmo formato da label shift
    rows = []

    for _, row in df.iterrows():
        for pair in alpha_pairs:
            rows.append([
                row["Dataset"],
                pair,
                row[pair]
            ])

    drift_df = pd.DataFrame(rows, columns=["Dataset", "Pair", "Drift"])

    grouped = (
        drift_df.groupby(["Dataset", "Pair"])["Drift"]
        .agg(["mean", "std", "count"])
    )

    tex_path = f"{base_dir}/concept_drift_table.tex"

    with open(tex_path, "w") as f:

        f.write("\\begin{figure}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Concept Drift (mean $\\pm$ 95\\% CI)}\n")
        f.write("\\label{tab:ps_concept_drift}\n")

        # Mesmo formato da label shift
        col_format = "l" + "c" * len(datasets)
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\toprule\n")

        header = ["Pair"] + datasets
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")

        for pair in alpha_pairs:

            row = [pair]

            for dataset in datasets:

                try:
                    mean = grouped.loc[(dataset, pair)]["mean"]
                    std = grouped.loc[(dataset, pair)]["std"]
                    n = grouped.loc[(dataset, pair)]["count"]

                    ci = 1.96 * (std / np.sqrt(n))
                    value = f"{mean:.2f} $\\pm$ {ci:.2f}"

                except:
                    value = "-"

                row.append(value)

            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{figure}\n")

    print(f"Tabela salva em {tex_path}")

if __name__ == "__main__":

    total_clients = 40
    alphas = [0.1, 1.0, 10.0]
    alpha_tuples = [(0.1, 1.0), (0.1, 10.0), (1.0, 10.0)]
    dataset = ["ImageNet10", "WISDM-W", "Foursquare"]

    write_path = f"plots/MEFL/clients_{total_clients}_datasets_{dataset}_alphas_{alphas}/"

    df = read_data(alphas, dataset, total_clients)

    latex_concept_drift_table(df, write_path)