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

def read_data(alphas,
              datasets,
              total_clients):

    filename = (
        f"clients_{total_clients}_datasets_{datasets}"
        f"_alphas_{alphas}_metrics_clients.csv"
    )

    if os.path.exists(filename):
        print("O arquivo existe!")
        df = pd.read_csv(filename)

    else:
        print("O arquivo não existe!")

        n_classes = [
            {
                'EMNIST': 47,
                'MNIST': 10,
                'CIFAR10': 10,
                'GTSRB': 43,
                'WISDM-W': 12,
                'WISDM-P': 12,
                'ImageNet': 15,
                "ImageNet10": 10,
                "ImageNet_v2": 15,
                "Gowalla": 7,
                "wikitext": 30,
                "Foursquare": 10
            }[dataset]
            for dataset in datasets
        ]

        ME = len(datasets)

        client_metrics = {
            cid: {
                me: {
                    alpha: {
                        "fc": None,
                        "il": None,
                        "similarity": None
                    }
                    for alpha in [0.1, 1.0, 10.0]
                }
                for me in range(ME)
            }
            for cid in range(1, total_clients + 1)
        }

        clients_train_loader = {
            cid: {
                alpha: {
                    me: None
                    for me in range(ME)
                }
                for alpha in alphas
            }
            for cid in range(1, total_clients + 1)
        }

        rows = []

        for client_id in range(1, total_clients + 1):

            for i in range(len(alphas)):

                alpha = alphas[i]

                if i > 0:
                    p_ME_old = copy.deepcopy(p_ME)

                for me in range(ME):

                    clients_train_loader[client_id][alpha][me], a = load_data(
                        dataset_name=datasets[me],
                        alpha=alpha,
                        data_sampling_percentage=0.8,
                        partition_id=client_id,
                        num_partitions=total_clients + 1,
                        batch_size=32,
                    )

                    print(
                        """leu dados cid: {} dataset: {} size: {}""".format(
                            client_id,
                            datasets[me],
                            len(
                                clients_train_loader[client_id][alpha][me].dataset
                            )
                        )
                    )

                p_ME, fc_ME, il_ME = get_datasets_metrics(
                    clients_train_loader[client_id][alpha],
                    ME,
                    n_classes
                )

                for me in range(ME):

                    client_metrics[client_id][me][alpha]["fc"] = fc_ME[me]

                    client_metrics[client_id][me][alpha]["il"] = il_ME[me]

        alpha_tuples = [
            (0.1, 1.0),
            (0.1, 10.0),
            (1.0, 10.0)
        ]

        alpha_tuples_string = [
            f"{alpha_tuple[0]}<->{alpha_tuple[1]}"
            for alpha_tuple in alpha_tuples
        ]

        general_metrics_dict = {
            alpha: {
                "fc": None,
                "il": None,
                "dh": None
            }
            for alpha in [0.1, 1.0, 10.0]
        }

        for me in range(ME):

            for cid in range(1, total_clients + 1):

                for alpha in [0.1, 1.0, 10.0]:

                    fc = client_metrics[cid][me][alpha]["fc"]

                    il = client_metrics[cid][me][alpha]["il"]

                    if fc is not None and il is not None:
                        dh = ((1 - fc) + il) / 2
                    else:
                        dh = None

                    general_metrics_dict[alpha] = {
                        "fc": round(fc, 2),
                        "il": round(il, 2),
                        "dh": round(dh, 2)
                    }

                similarity_ALPHA = {
                    alpha_tuple: None
                    for alpha_tuple in alpha_tuples_string
                }

                for alpha_tuple in alpha_tuples:

                    alpha_a = alpha_tuple[0]

                    alpha_b = alpha_tuple[1]

                    p_ME_a, fc_ME, il_ME = get_datasets_metrics(
                        clients_train_loader[cid][alpha_a],
                        ME,
                        n_classes
                    )

                    p_ME_b, fc_ME, il_ME = get_datasets_metrics(
                        clients_train_loader[cid][alpha_b],
                        ME,
                        n_classes
                    )

                    similarity_me = (
                        1 -
                        cosine_similarity(
                            p_ME_a[me],
                            p_ME_b[me]
                        )
                    )

                    similarity_ALPHA[
                        f"{alpha_tuple[0]}<->{alpha_tuple[1]}"
                    ] = round(similarity_me, 2)

                for alpha in [0.1, 1.0, 10.0]:

                    dataset_size = len(
                        clients_train_loader[cid][alpha][me].dataset
                    )

                    row = [
                        cid,
                        me,
                        datasets[me]
                        .replace("WISDM-W", "WISDM")
                        .replace("ImageNet10", "ImageNet-10"),
                        alpha,
                        dataset_size,
                        general_metrics_dict[alpha]["fc"],
                        general_metrics_dict[alpha]["il"],
                        general_metrics_dict[alpha]["dh"],
                        similarity_ALPHA["0.1<->1.0"],
                        similarity_ALPHA["0.1<->10.0"],
                        similarity_ALPHA["1.0<->10.0"]
                    ]

                    rows.append(row)

        df = pd.DataFrame(
            data=rows,
            columns=[
                "cid",
                "me",
                "Dataset",
                "\u03B1",
                "dataset_size",
                "fc",
                "il",
                "dh",
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


def latex_general_metrics_table(
        df,
        base_dir,
        selected_clients_fraction=0.4,
        seed=42):

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    datasets = ["WISDM", "ImageNet-10", "Foursquare"]

    alphas = [0.1, 1.0, 10.0]

    metrics = ["fc", "il", "dh"]

    selected_clients = sorted(
        rng.choice(
            df["cid"].unique(),
            size=max(
                1,
                int(
                    len(df["cid"].unique()) *
                    selected_clients_fraction
                )
            ),
            replace=False
        )
    )

    print("Selected clients table:", selected_clients)

    df = df[df["cid"].isin(selected_clients)]

    tex_path = f"{base_dir}/general_metrics_table.tex"

    with open(tex_path, "w") as f:

        f.write("\\begin{table}[t]\n")

        f.write("\\centering\n")

        f.write(
            "\\caption{General Metrics "
            "(weighted mean $\\pm$ 95\\% CI)}\n"
        )

        f.write("\\label{tab:general_metrics}\n")

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

                    subset = df[
                        (df["Dataset"] == dataset) &
                        (df["α"] == alpha)
                    ]

                    if len(subset) == 0:

                        row.append("-")

                        continue

                    values = subset[metric].values.astype(float)

                    weights = subset["dataset_size"].values.astype(float)

                    weights = weights / np.sum(weights)

                    weighted_mean = np.sum(values * weights)

                    weighted_var = np.sum(
                        weights *
                        (values - weighted_mean) ** 2
                    )

                    weighted_std = np.sqrt(weighted_var)

                    n = len(values)

                    ci = 1.96 * (
                        weighted_std / np.sqrt(n)
                    )

                    value = (
                        f"{weighted_mean:.2f} "
                        f"$\\pm$ {ci:.2f}"
                    )

                    row.append(value)

                f.write(" & ".join(row) + " \\\\\n")

            f.write("\\midrule\n")

        f.write("\\bottomrule\n")

        f.write("\\end{tabular}\n")

        f.write("\\end{table}\n")

    print(f"Tabela salva em {tex_path}")

def latex_ps_table(
        df,
        base_dir,
        selected_clients_fraction=0.4,
        seed=42):

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    datasets = ["WISDM", "ImageNet-10", "Foursquare"]

    alpha_pairs = [
        "0.1<->1.0",
        "0.1<->10.0",
        "1.0<->10.0"
    ]

    selected_clients = sorted(
        rng.choice(
            df["cid"].unique(),
            size=max(
                1,
                int(
                    len(df["cid"].unique()) *
                    selected_clients_fraction
                )
            ),
            replace=False
        )
    )

    print("Selected clients table:", selected_clients)

    df = df[df["cid"].isin(selected_clients)]

    tex_path = f"{base_dir}/label_shift_table.tex"

    with open(tex_path, "w") as f:

        f.write("\\begin{table}[t]\n")

        f.write("\\centering\n")

        f.write(
            "\\caption{Label Shift "
            "(weighted mean $\\pm$ 95\\% CI and correlated min--max)}\n"
        )

        f.write("\\label{tab:ps_label_shift}\n")

        f.write("\\begin{tabular}{lccc}\n")

        f.write("\\toprule\n")

        f.write(
            "Dataset & Pair & "
            "Mean $\\pm$ CI & Min--Max \\\\\n"
        )

        f.write("\\midrule\n")

        for dataset in datasets:

            subset_dataset = df[
                df["Dataset"] == dataset
            ]

            if len(subset_dataset) == 0:
                continue

            for idx, pair in enumerate(alpha_pairs):

                values = subset_dataset[pair].values.astype(float)

                weights = (
                    subset_dataset["dataset_size"]
                    .values.astype(float)
                )

                weights = weights / np.sum(weights)

                weighted_mean = np.sum(values * weights)

                weighted_var = np.sum(
                    weights *
                    (values - weighted_mean) ** 2
                )

                weighted_std = np.sqrt(weighted_var)

                n = len(values)

                ci = 1.96 * (
                    weighted_std / np.sqrt(n)
                )

                mean_ci_value = (
                    f"{weighted_mean:.2f} "
                    f"$\\pm$ {ci:.2f}"
                )

                min_value = np.min(values)

                max_value = np.max(values)

                min_max_value = (
                    f"{min_value:.2f}--{max_value:.2f}"
                )

                if idx == 0:

                    dataset_col = (
                        f"\\multirow{{{len(alpha_pairs)}}}{{*}}"
                        f"{{{dataset}}}"
                    )

                else:

                    dataset_col = ""

                row = [
                    dataset_col,
                    pair,
                    mean_ci_value,
                    min_max_value
                ]

                f.write(
                    " & ".join(row) + " \\\\\n"
                )

            f.write("\\midrule\n")

        f.write("\\bottomrule\n")

        f.write("\\end{tabular}\n")

        f.write("\\end{table}\n")

    print(f"Tabela salva em {tex_path}")

if __name__ == "__main__":

    total_clients = 40
    fraction_fit = 0.375

    alphas = [0.1, 1.0, 10.0]

    dataset = [
        "WISDM-W",
        "ImageNet10",
        "Foursquare"
    ]

    write_path = (
        f"plots/MEFL/"
        f"clients_{total_clients}_datasets_{dataset}_alphas_{alphas}/"
    )

    df = read_data(
        alphas=alphas,
        datasets=dataset,
        total_clients=total_clients
    )

    latex_general_metrics_table(
        df,
        write_path,
        selected_clients_fraction=fraction_fit,
        seed=42
    )

    latex_ps_table(
        df,
        write_path,
        selected_clients_fraction=fraction_fit,
        seed=42
    )