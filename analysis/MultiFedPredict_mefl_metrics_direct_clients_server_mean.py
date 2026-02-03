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

    filename = f"clients_{total_clients}_datasets_{datasets}_alphas_{alphas}_metrics_clients_server_mean.csv"

    if os.path.exists(filename):
        print("O arquivo existe!")
        df = pd.read_csv(filename)
    else:
        print("O arquivo não existe!")

        n_classes = [
            {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
             "ImageNet10": 10, "ImageNet_v2": 15, "Gowalla": 7, "wikitext": 30, "Foursquare": 100}[dataset] for dataset in
            datasets]
        ME = len(datasets)
        client_metrics = {
                cid: {me: {alpha: {"fc": None, "il": None, "similarity": None, "size": None} for alpha in [0.1, 1.0, 10.0]} for me in
                      range(ME)} for cid in range(1, total_clients + 1)}
        clients_train_loader = {cid:  {alpha: {me: None for me in range(ME)} for alpha in alphas} for cid in range(1, total_clients + 1)}
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
                    print("""leu dados cid: {} dataset: {} size:  {}""".format(client_id, datasets[me],
                                                                               len(clients_train_loader[client_id][alpha][me].dataset)))


                p_ME, fc_ME, il_ME = get_datasets_metrics(clients_train_loader[client_id][alpha], ME, n_classes)
                # similarity_ME = []
                #
                # for me in range(ME):
                #     if i>0:
                #         similarity_me = cosine_similarity(p_ME[me], p_ME_old[me])
                #     else:
                #         similarity_me = 1
                #     similarity_ME.append(similarity_me)

                for me in range(ME):
                    client_metrics[client_id][me][alpha]["fc"] = fc_ME[me]
                    client_metrics[client_id][me][alpha]["il"] = il_ME[me]
                    client_metrics[client_id][me][alpha]["size"] = len(clients_train_loader[client_id][alpha][me].dataset)
                    # client_metrics[client_id][me][alpha]["similarity"] = similarity_ME[me]


        alpha_tuples = [(0.1, 1.0), (0.1, 10.0), (1.0, 10.0)]
        alpha_tuples_string = [f"{alpha_tuple[0]}<->{alpha_tuple[1]}" for alpha_tuple in alpha_tuples]
        general_metrics_dict = {alpha: {"fc": None, "il": None, "dh": None} for alpha in [0.1, 1.0, 10.0]}
        for me in range(ME):
            for cid in range(1, total_clients + 1):
                for alpha in [0.1, 1.0, 10.0]:
                    fc = client_metrics[cid][me][alpha]["fc"]
                    il = client_metrics[cid][me][alpha]["il"]
                    size = client_metrics[cid][me][alpha]["size"]
                    # if fc is not None and il is not None:
                    #     dh = ((1 - fc) + il) / 2
                    # else:
                    #     dh = None
                    general_metrics_dict[alpha] = {"fc": round(fc, 2), "il": round(il, 2), "size": size}

                similarity_ALPHA = {alpha_tuple: None for alpha_tuple in alpha_tuples_string}
                for alpha_tuple in alpha_tuples:
                    alpha_a = alpha_tuple[0]
                    alpha_b = alpha_tuple[1]

                    p_ME_a, fc_ME, il_ME = get_datasets_metrics(clients_train_loader[cid][alpha_a], ME, n_classes)
                    p_ME_b, fc_ME, il_ME = get_datasets_metrics(clients_train_loader[cid][alpha_b], ME, n_classes)
                    similarity_me = 1 - cosine_similarity(p_ME_a[me], p_ME_b[me])
                    similarity_ALPHA[f"{alpha_tuple[0]}<->{alpha_tuple[1]}"] = round(similarity_me, 2)


                for alpha in [0.1, 1.0, 10.0]:
                    row = [cid, me, datasets[me].replace("WISDM-W", "WISDM").replace("ImageNet10", "ImageNet-10"),
                           alpha, general_metrics_dict[alpha]["fc"], general_metrics_dict[alpha]["il"],
                           general_metrics_dict[alpha]["size"], similarity_ALPHA["0.1<->1.0"],
                           similarity_ALPHA["0.1<->10.0"], similarity_ALPHA["1.0<->10.0"]]
                    rows.append(row)

        df = pd.DataFrame(data=rows,
                          columns=["cid", "me", "Dataset", "\u03B1", "fc", "il", "size", "0.1<->1.0",
                                   "0.1<->10.0", "1.0<->10.0"])

        df_weighted_mean = df.groupby(["me", "Dataset", "\u03B1"]).apply(lambda e: weighted_average(e)).reset_index()[["Dataset", "me", "\u03B1", "fc", "il", "dh", "0.1<->1.0", "0.1<->10.0", "1.0<->10.0"]]

        print(df_weighted_mean)

        df_weighted_mean.to_csv(filename, index=False)
        df = df_weighted_mean

    return df

def weighted_average(df):

    columns = ["fc", "il", "0.1<->1.0", "0.1<->10.0", "1.0<->10.0"]
    total_samples = df["size"].sum()
    results = {column: 0 for column in columns}
    for column in columns:
        result = df[column].to_numpy() * df["size"].to_numpy()
        result = np.sum(result, axis=0) / total_samples
        results[column] = result

    dh = ((1 - results["fc"]) + results["il"]) / 2

    df = pd.DataFrame({"fc": [results["fc"]], "il": [results["il"]], "dh": [dh], "0.1<->1.0": [results["0.1<->1.0"]], "0.1<->10.0": [results["0.1<->10.0"]], "1.0<->10.0": [results["1.0<->10.0"]]})

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


def bar_general_metrics(df, base_dir, x, y_list, hue=None, style=None, ci=None, hue_order=None):

    datasets = df["Dataset"].unique().tolist()

    fig, axs = plt.subplots(3, sharex='all', figsize=(7, 7))
    hue_order = ["MultiFedAvg", "MultiFedAvgRR"]

    # print(df)
    # exit()

    for i in range(len(y_list)):

        bar_plot(df=df, base_dir=base_dir, ax=axs[i],
                  file_name="""solutions_{}_server_mean""".format(y_list), x_column=x, y_column=y_list[i],
                 hue="Dataset", title="", tipo="", y_lim=True, y_max=1.15, palette=sns.color_palette())
        axs[i].set_ylim(0, 1.24)
        # axs[j].set_title(r"""Dataset: {}""".format(datasets[j]), size=10)

        if not (i == 2):
            axs[i].get_legend().remove()


    # axs[1].legend(handles, labels, fontsize=9)

    # fig.suptitle("", fontsize=16)

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig(
        """{}{}_general_metrics_server_mean.png""".format(base_dir, y_list), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}{}_general_metrics_server_mean.svg""".format(base_dir, y_list), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}{}_general_metrics_server_mean.pdf""".format(base_dir, y_list), bbox_inches='tight',
        dpi=400)
    print("""{}{}_general_metrics_server_mean.png""".format(base_dir, y_list))

def bar_ps(df, base_dir, x, y_list, hue=None, style=None, ci=None, hue_order=None):

    datasets = df["Dataset"].unique().tolist()
    alpha_tuples = [(0.1, 1.0), (0.1, 10.0), (1.0, 10.0)]

    alpha_tuples_string = [f"{alpha_tuple[0]}<->{alpha_tuple[1]}" for alpha_tuple in alpha_tuples]

    columns = ["me", "Dataset", "ps", "Type"]
    new_df = []

    for i in range(len(df)):

        row = df.iloc[i]
        print(row)
        print(row["Dataset"])
        print(row["dh"])
        for alpha_tuple in alpha_tuples_string:
            new_row = [row["me"], row["Dataset"], row[alpha_tuple], alpha_tuple]
            new_df.append(new_row)


    df = pd.DataFrame(new_df, columns=columns)

    print(df)

    fig = plt.figure(figsize=(10, 7))

    bar_plot(df=df, base_dir=base_dir,
              file_name="""{}_ps_server_mean""".format(y_list), x_column="Type", y_column="ps",
             hue="Dataset", title="", tipo="", y_lim=False, y_max=1.1, sci=True, palette=sns.color_palette(), padding=26)


if __name__ == "__main__":
    total_clients = 10
    alphas = [0.1, 1.0, 10.0]
    dataset = ["ImageNet10", "WISDM-W", "wikitext"]
    # dataset = ["WISDM-W"]
    write_path = f"plots/MEFL/clients_{total_clients}_datasets_{dataset}_alphas_{alphas}/"

    df = read_data(alphas, dataset, total_clients)

    y_list = ["fc", "il", "dh"]

    bar_general_metrics(df, write_path, x="\u03B1", y_list=y_list)
    bar_general_metrics(df, write_path, x="\u03B1", y_list=y_list)

    y_list = ["0.1<->1.0", "0.1<->10.0", "1.0<->10.0"]
    bar_ps(df, write_path, x="\u03B1", y_list=y_list)
    bar_ps(df, write_path, x="\u03B1", y_list=y_list)