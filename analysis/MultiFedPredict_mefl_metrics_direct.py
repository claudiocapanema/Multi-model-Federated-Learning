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

    filename = f"clients_{total_clients}_datasets_{datasets}_alphas_{alphas}_metrics.csv"

    if os.path.exists(filename):
        print("O arquivo existe!")
        df = pd.read_csv(filename)
    else:
        print("O arquivo não existe!")

        n_classes = [
            {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
             "ImageNet10": 10, "ImageNet_v2": 15, "Gowalla": 7, "wikitext": 30}[dataset] for dataset in
            datasets]
        ME = len(datasets)
        client_metrics = {
                cid: {me: {alpha: {"fc": None, "il": None, "similarity": None} for alpha in [0.1, 1.0, 10.0]} for me in
                      range(ME)} for cid in range(1, total_clients + 1)}
        for client_id in range(1, total_clients + 1):

            trainloader = {alpha: {me: None for me in range(ME)} for alpha in alphas}
            valloader = {alpha: {me: None for me in range(ME)} for alpha in alphas}
            for i in range(len(alphas)):
                alpha = alphas[i]
                if i > 0:
                    p_ME_old = copy.deepcopy(p_ME)
                for me in range(ME):
                    trainloader[alpha][me], valloader[alpha][me] = load_data(
                        dataset_name=datasets[me],
                        alpha=alpha,
                        data_sampling_percentage=0.8,
                        partition_id=client_id,
                        num_partitions=total_clients + 1,
                        batch_size=32,
                    )
                    print("""leu dados cid: {} dataset: {} size:  {}""".format(client_id, datasets[me],
                                                                               len(trainloader[alpha][me].dataset)))


                p_ME, fc_ME, il_ME = get_datasets_metrics(trainloader[alpha], ME, n_classes)
                similarity_ME = []

                for me in range(ME):
                    if i>0:
                        similarity_me = cosine_similarity(p_ME[me], p_ME_old[me])
                    else:
                        similarity_me = 1
                    similarity_ME.append(similarity_me)

                for me in range(ME):
                    client_metrics[client_id][me][alpha]["fc"] = fc_ME[me]
                    client_metrics[client_id][me][alpha]["il"] = il_ME[me]
                    client_metrics[client_id][me][alpha]["similarity"] = similarity_ME[me]

        rows = []
        for me in range(ME):
            for cid in range(1, total_clients + 1):
                for alpha in [0.1, 1.0, 10.0]:
                    fc = client_metrics[cid][me][alpha]["fc"]
                    il = client_metrics[cid][me][alpha]["il"]
                    if fc is not None and il is not None:
                        dh = (fc + (1 - il)) / 2
                    else:
                        dh = None
                    row = [cid, me, datasets[me], alpha, round(fc, 2), round(il, 2), round(dh, 2), round(client_metrics[cid][me][alpha]["similarity"], 2)]
                    rows.append(row)

        df = pd.DataFrame(data=rows, columns = ["cid", "me", "Dataset", "\u03B1", "fc", "il", "dh", "similarity"])

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


def bar(df, base_dir, x, y_list, hue=None, style=None, ci=None, hue_order=None):

    datasets = df["Dataset"].unique().tolist()

    fig, axs = plt.subplots(2, 2, sharex='all', figsize=(10, 7))
    hue_order = ["MultiFedAvg", "MultiFedAvgRR"]

    for k in range(len(y_list)):

        if k <= 2:
         df_plot = df
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
        print(df_plot)
        bar_plot(df=df_plot, base_dir=base_dir, ax=axs[i, j],
                  file_name="""solutions_{}""".format(y_list), x_column=x, y_column=y,
                 hue="Dataset", title="", tipo=tipo, y_lim=True, y_max=1, palette=sns.color_palette())
        axs[i, j].set_ylim(0, 1.15)
        # axs[j].set_title(r"""Dataset: {}""".format(datasets[j]), size=10)

        if not (i == 0 and j == 1):
            axs[i, j].get_legend().remove()


    # axs[1].legend(handles, labels, fontsize=9)

    # fig.suptitle("", fontsize=16)

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig(
        """{}{}_metrics.png""".format(base_dir, y_list), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}{}_metrics.svg""".format(base_dir, y_list), bbox_inches='tight',
        dpi=400)
    print("""{}{}_metrics.png""".format(base_dir, y_list))


if __name__ == "__main__":
    total_clients = 10
    alphas = [0.1, 1.0, 10.0]
    dataset = ["ImageNet10", "WISDM-W", "wikitext"]
    # dataset = ["WISDM-W"]
    write_path = f"plots/MEFL/clients_{total_clients}_datasets_{dataset}_alphas_{alphas}/"

    df = read_data(alphas, dataset, total_clients)

    y_list = ["fc", "il", "dh", "similarity"]

    bar(df, write_path, x="\u03B1", y_list=y_list)
    bar(df, write_path, x="\u03B1", y_list=y_list)