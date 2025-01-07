# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import pandas as pd
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import os
import numpy as np
import csv
import copy
import time
import random
from utils.data_utils import read_client_data_v2
from utils.dlg import DLG

from functools import reduce
from typing import List, Tuple

import numpy as np

from torch.nn.parameter import Parameter

import numpy.typing as npt

from io import BytesIO
from dataclasses import dataclass

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str

def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=False)  # type: ignore
    return bytes_io.getvalue()

def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.models_names = args.models_names
        self.models_size = []
        self.device = args.device
        self.dataset = args.dataset
        # self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.M = len(self.dataset)
        self.global_model = [copy.deepcopy(args.model[m].student) for m in range(self.M)]
        self.num_clients = args.num_clients
        self.num_classes = args.num_classes
        self.alpha = args.alpha
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.train_metrics_names = ["Accuracy", "Loss", "Samples", "AUC", "Balanced accuracy", "Micro f1-score", "Weighted f1-score", "Macro f1-score", "Round", "Fraction fit", "# training clients", "Alpha"]
        self.test_metrics_names = ["Accuracy", "Std Accuracy", "Loss", "Std loss", "AUC", "Balanced accuracy", "Micro f1-score", "Weighted f1-score", "Macro f1-score", "Round", "Fraction fit", "# training clients", "training clients and models", "model size", "Alpha"]
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.results_train_metrics = {m: {metric: [] for metric in self.train_metrics_names} for m in range(self.M)}
        self.results_train_metrics_w = {m: {metric: [] for metric in self.train_metrics_names} for m in range(self.M)}
        self.results_test_metrics = {m: {metric: [] for metric in self.test_metrics_names} for m in range(self.M)}
        self.results_test_metrics_w = {m: {metric: [] for metric in self.test_metrics_names} for m in range(self.M)}
        self.clients_results_test_metrics = {m: {metric: [] for metric in self.test_metrics_names} for m in range(self.M)}

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.fraction_new_clients = args.fraction_new_clients
        self.round_new_clients = args.round_new_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_available_clients = int(self.num_clients * (1 - self.fraction_new_clients))
        self.available_clients = self.clients[:self.num_available_clients]
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients

        self.clients_test_metrics = {i: {metric: {m: [] for m in range(self.M)} for metric in ["Accuracy", "Loss", "Samples", "Balanced accuracy", "Micro f1-score", "Macro f1-score", "Weighted f1-score", "Round"]} for i in range(self.num_clients)}
        self.clients_train_metrics = {i: {metric: {m: [] for m in range(self.M)} for metric in ["Accuracy", "Loss", "Samples", "Balanced accuracy", "Micro f1-score", "Macro f1-score", "Weighted f1-score"]} for
                                     i in range(self.num_clients)}

        self.past_train_metrics_m = {m: {} for m in range(self.M)}
        self.max_loss_m = {m: 0 for m in range(self.M)}
        self.concept_drift = bool(self.args.concept_drift)
        self.alpha_end = self.args.alpha_end
        self.rounds_concept_drift = self.args.rounds_concept_drift

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = []
            test_data = []
            client = clientObj(self.args,
                            id=i,
                            train_slow=train_slow,
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def get_available_clients(self, t, m):

        if t < self.round_new_clients:
            self.num_available_clients = int(self.num_clients * (1 - self.fraction_new_clients))
            self.available_clients = self.clients[:self.num_available_clients]
            self.num_join_clients = int(self.num_available_clients * self.join_ratio)
        else:
            self.num_available_clients = int(self.num_clients )
            self.available_clients = self.clients
            self.num_join_clients = int(self.num_available_clients * self.join_ratio)

        return self.available_clients

    def select_clients(self, t):
        g = torch.Generator()
        g.manual_seed(t)
        np.random.seed(t)
        random.seed(t)

        if t < self.round_new_clients:
            self.num_available_clients = int(self.num_clients * (1 - self.fraction_new_clients))
            self.available_clients = self.clients[:self.num_available_clients]
            self.num_join_clients = int(self.num_clients * self.join_ratio)
        else:
            self.num_available_clients = len(self.clients)
            self.available_clients = self.clients
            self.num_join_clients = int(self.num_clients * self.join_ratio)

        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_available_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_available_clients
        np.random.seed(t)
        selected_clients = list(np.random.choice(self.available_clients, self.num_join_clients, replace=False))
        selected_clients = [i.id for i in selected_clients]

        n = len(selected_clients) // self.M
        sc = np.array_split(selected_clients, self.M)
        # sc = [np.array(selected_clients[0:6])]
        # sc.append(np.array(selected_clients[6:]))

        print("Selecionados: ", sc, [len(i) for i in sc], self.available_clients)

        return sc

    def send_models(self):
        assert (len(self.clients) > 0)

        for i in range(len(self.selected_clients)):
            start_time = time.time()
            
            self.clients[i].set_parameters_all_models(self.global_model)

            self.clients[i].send_time_cost['num_rounds'] += 1
            self.clients[i].send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        print("aa: ", len(self.selected_clients), int((1-self.client_drop_rate) * self.current_num_join_clients))
        # active_clients_m = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []

        for m in range(len(self.selected_clients)):
            tot_samples = 0
            active_clients_m = self.selected_clients[m]
            print("m: ", m, " ativos: ", len(active_clients_m))
            m_uploaded_ids = []
            m_uploaded_weights = []
            m_uploaded_models = []
            for client_id in active_clients_m:
                client = self.clients[client_id]
                try:
                    client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                            client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
                except ZeroDivisionError:
                    client_time_cost = 0
                if client_time_cost <= self.time_threthold:
                    tot_samples += client.train_samples[m]
                    m_uploaded_ids.append(client.id)
                    m_uploaded_weights.append(client.train_samples[m])
                    m_uploaded_models.append(client.model[m])
            for i, w in enumerate(m_uploaded_weights):
                m_uploaded_weights[i] = w / tot_samples

            print("modelo: ", m, " tam: ", len(m_uploaded_models))

            self.uploaded_ids.append(copy.deepcopy(m_uploaded_ids))
            self.uploaded_weights.append(copy.deepcopy(m_uploaded_weights))
            self.uploaded_models.append(copy.deepcopy(m_uploaded_models))

        # exit()

    def aggregate(self, results: List[Tuple[NDArrays, float]], m: int) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples, cid in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer.detach().cpu().numpy() * num_examples for layer in weights.parameters()] for weights, num_examples, cid in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        weights_prime = [Parameter(torch.Tensor(i.tolist())) for i in weights_prime]

        return weights_prime

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        for m in range(len(self.uploaded_models)):
            # self.global_model[m] = copy.deepcopy(self.uploaded_models[m][0])
            # for param in self.global_model[m].parameters():
            #     param.data.zero_()
            parameters_tuple = []
            for cid, w, client_model in zip(self.uploaded_ids[m], self.uploaded_weights[m], self.uploaded_models[m]):
                # self.add_parameters(w, client_model, m)
                parameters_tuple.append((client_model, w, cid))

            agg_parameters = self.aggregate(parameters_tuple, m)

            for server_param, client_param in zip(self.global_model[m].parameters(), agg_parameters):
                # print(": ", server_param.data.shape, client_param.data.clone().shape)
                server_param.data = client_param.data.clone()

        # print("agr f")
        # exit()


    def add_parameters(self, w, client_model, m):
        # if type(self.global_model[m]) == list:
        #     if len(self.global_model[m]) == 0:
        #         self.global_model[m] = client_model
        # else:
        for server_param, client_param in zip(self.global_model[m].parameters(), client_model.parameters()):
            # print(": ", server_param.data.shape, client_param.data.clone().shape)
            server_param.data += client_param.data.clone() * w

    def save_global_model(self, m):
        model_path = os.path.join("models", self.dataset[m])
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model[m], model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def get_results(self, m):

        algo = self.dataset[m] + "_" + self.algorithm
        cd = bool(self.args.concept_drift)
        if cd:
            result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(cd,
                                                                                                                        self.fraction_new_clients,
                                                                                                                        self.round_new_clients,
                                                                                                                        self.num_clients,
                                                                                                                       self.alpha,
                                                                                                                        self.alpha_end[
                                                                                                                            0],
                                                                                                                        self.alpha_end[
                                                                                                                            1],
                                                                                                                       self.dataset,
                                                                                                                        self.rounds_concept_drift[
                                                                                                                            0],
                                                                                                                        self.rounds_concept_drift[
                                                                                                                            1],
                                                                                                                       self.models_names,
                                                                                                                       self.args.join_ratio,
                                                                                                                       self.args.global_rounds,
                                                                                                                       self.local_epochs)
        elif len(self.alpha) == 1:
            result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
                cd,
                self.fraction_new_clients,
                self.round_new_clients,
                self.num_clients,
                [self.alpha[0]],
                self.alpha[
                    0],
                self.alpha[
                    0],
                [self.dataset[0]],
                0,
                0,
                [self.models_names[0]],
                self.args.join_ratio,
                self.args.global_rounds,
                self.local_epochs)
        else:

            result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
                cd,
                self.fraction_new_clients,
                self.round_new_clients,
                self.num_clients,
                self.alpha,
                self.alpha[
                    0],
                self.alpha[
                    1],
                self.dataset,
                0,
                0,
                self.models_names,
                self.args.join_ratio,
                self.args.global_rounds,
                self.local_epochs)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.csv".format(algo)
            header = self.test_metrics_names
            print(self.rs_test_acc)
            print(self.rs_test_auc)
            print(self.rs_train_loss)
            list_of_metrics = []
            for me in self.results_test_metrics[m]:
                print(me, len(self.results_test_metrics[m][me]))
                length = len(self.results_test_metrics[m][me])
                list_of_metrics.append(self.results_test_metrics[m][me])

            data = []
            for i in range(length):
                row = []
                for j in range(len(list_of_metrics)):
                    row.append(list_of_metrics[j][i])

                data.append(row)

            print("File path: " + file_path)
            print(data)
            print("me: ", self.rs_test_acc)

            return file_path, header, data

    def get_results_weighted(self, m):

        if (len(self.rs_test_acc)):
            header = self.test_metrics_names
            print(self.rs_test_acc)
            print(self.rs_test_auc)
            print(self.rs_train_loss)
            list_of_metrics = []
            for me in self.results_test_metrics_w[m]:
                print(me, len(self.results_test_metrics_w[m][me]))
                length = len(self.results_test_metrics_w[m][me])
                list_of_metrics.append(self.results_test_metrics_w[m][me])

            data = []
            for i in range(length):
                row = []
                for j in range(len(list_of_metrics)):
                    row.append(list_of_metrics[j][i])

                data.append(row)

            print(data)
            print("me: ", self.rs_test_acc)

            return header, data
        
    def save_results(self, m):

            file_path, header, data = self.get_results(m)
            self.clients_test_metrics_preprocess(m, file_path)
            self._write_header(file_path, header=header)
            self._write_outputs(file_path, data=data)
            header, data = self.get_results_weighted(m)
            file_path = file_path.replace(".csv", "_weighted.csv")
            self.clients_test_metrics_preprocess(m, file_path)
            self._write_header(file_path, header=header)
            self._write_outputs(file_path, data=data)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self, m, t):
        test_clients = self.get_available_clients(t, m)

        num_samples = []
        accs_w = []
        std_accs_w = []
        loss_w = []
        std_losses_w = []
        auc_w = []
        balanced_acc_w = []
        micro_fscore_w = []
        weighted_fscore_w = []
        macro_fscore_w = []
        accs = []
        std_accs = []
        loss = []
        std_losses = []
        auc = []
        balanced_acc = []
        micro_fscore = []
        weighted_fscore = []
        macro_fscore = []
        alpha_list = []
        for i in range(len(test_clients)):
            # if i in self.selected_clients[m] or t == 1:
            c = test_clients[i]
            global_model = self.global_model[m]
            if type(global_model) != list:
                global_model = self.global_model[m].to(self.device)
            test_acc, test_loss, test_num, test_auc, test_balanced_acc, test_micro_fscore, test_macro_fscore, test_weighted_fscore, alpha = c.test_metrics(m, t, self.global_rounds, copy.deepcopy(global_model))
            self.clients_test_metrics[i]["Accuracy"][m].append(test_acc)
            self.clients_test_metrics[i]["Loss"][m].append(test_loss)
            self.clients_test_metrics[i]["Balanced accuracy"][m].append(test_balanced_acc)
            self.clients_test_metrics[i]["Micro f1-score"][m].append(test_micro_fscore)
            self.clients_test_metrics[i]["Samples"][m].append(test_num)
            # print("test_weighted_fscore: ", test_weighted_fscore, len(self.clients_test_metrics[i]["Weighted f1-score"][m]))
            self.clients_test_metrics[i]["Weighted f1-score"][m].append(test_weighted_fscore)
            self.clients_test_metrics[i]["Macro f1-score"][m].append(test_macro_fscore)
            self.clients_test_metrics[i]["Round"][m].append(t)
            # accs.append(test_acc*test_num)
            # auc.append(test_auc*test_num)
            # num_samples.append(test_num)
            # loss.append(test_loss*test_num)
            # balanced_acc.append(test_balanced_acc*test_num)
            # micro_fscore.append(test_micro_fscore*test_num)
            # weighted_fscore.append(test_weighted_fscore*test_num)
            # macro_fscore.append(test_macro_fscore*test_num)
            accs_w.append(test_acc * test_num)
            std_accs_w.append(test_acc * test_num)
            auc_w.append(test_auc * test_num)
            num_samples.append(test_num)
            loss_w.append(test_loss * test_num)
            std_losses_w.append(test_loss * test_num)
            balanced_acc_w.append(test_balanced_acc * test_num)
            micro_fscore_w.append(test_micro_fscore * test_num)
            weighted_fscore_w.append(test_weighted_fscore * test_num)
            macro_fscore_w.append(test_macro_fscore * test_num)

            accs.append(test_acc)
            std_accs.append(test_acc)
            auc.append(test_auc)
            # num_samples.append(test_num)
            loss.append(test_loss)
            std_losses.append(test_loss)
            balanced_acc.append(test_balanced_acc)
            micro_fscore.append(test_micro_fscore)
            weighted_fscore.append(test_weighted_fscore)
            macro_fscore.append(test_macro_fscore)
            alpha_list.append(alpha)

            # print(test_num)
            # exit()

        ids = [c.id for c in test_clients]

        decimals = 5
        acc_w = round(sum(accs_w) / sum(num_samples), decimals)
        std_acc_w = np.round(np.std(np.array(std_accs_w) / sum(num_samples)), decimals)
        auc_w = round(sum(auc_w) / sum(num_samples), decimals)
        loss_w = round(sum(loss_w) / sum(num_samples), decimals)
        std_loss_w = np.round(np.std(np.array(std_losses_w) / sum(num_samples)), decimals)
        balanced_acc_w = round(sum(balanced_acc_w) / sum(num_samples), decimals)
        micro_fscore_w = round(sum(micro_fscore_w) / sum(num_samples), decimals)
        weighted_fscore_w = round(sum(weighted_fscore_w) / sum(num_samples), decimals)
        macro_fscore_w = round(sum(macro_fscore_w) / sum(num_samples), decimals)

        acc = round(np.mean(accs), decimals)
        std_acc = np.round(np.std(np.array(std_accs)), decimals)
        auc = round(np.mean(auc), decimals)
        loss = round(np.mean(loss), decimals)
        std_loss = np.round(np.std(np.array(std_losses)), decimals)
        balanced_acc = round(np.mean(balanced_acc), decimals)
        micro_fscore = round(np.mean(micro_fscore), decimals)
        weighted_fscore = round(np.mean(weighted_fscore), decimals)
        macro_fscore = round(np.mean(macro_fscore), decimals)

        server_metrics = {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc, "Std Accuracy": std_acc, 'AUC': auc,
                "Loss": loss, "Std loss": std_loss, "Balanced accuracy": balanced_acc, "Micro f1-score": micro_fscore,
                "Weighted f1-score": weighted_fscore, "Macro f1-score": macro_fscore, "Alpha": alpha_list[0]}

        server_metrics_weighted = {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc_w, "Std Accuracy": std_acc_w, 'AUC': auc_w,
                "Loss": loss_w, "Std loss": std_loss_w, "Balanced accuracy": balanced_acc_w, "Micro f1-score": micro_fscore_w,
                "Weighted f1-score": weighted_fscore_w, "Macro f1-score": macro_fscore_w, "Alpha": alpha_list[0]}

        return server_metrics, server_metrics_weighted

    def train_metrics(self, m, t):

        accs = []
        losses = []
        num_samples = []
        balanced_accs = []
        micro_fscores = []
        macro_fscores = []
        weighted_fscores = []
        available_clients = self.get_available_clients(t, m)
        for i in range(len(available_clients)):
            c = available_clients[i]
            if i in self.selected_clients[m] or t == 1:
                train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore, alpha = c.train_metrics(m, t)
                accs.append(train_acc * train_num)
                num_samples.append(train_num)
                balanced_accs.append(train_balanced_acc * train_num)
                micro_fscores.append(train_micro_fscore * train_num)
                macro_fscores.append(train_macro_fscore * train_num)
                weighted_fscores.append(train_weighted_fscore * train_num)
                self.clients_train_metrics[c.id]["Samples"][m].append(num_samples)
                self.clients_train_metrics[c.id]["Accuracy"][m].append(train_acc)
                self.clients_train_metrics[c.id]["Loss"][m].append(train_loss)
                self.clients_train_metrics[c.id]["Balanced accuracy"][m].append(train_balanced_acc)
                self.clients_train_metrics[c.id]["Micro f1-score"][m].append(train_micro_fscore)
                self.clients_train_metrics[c.id]["Macro f1-score"][m].append(train_macro_fscore)
                self.clients_train_metrics[c.id]["Weighted f1-score"][m].append(train_weighted_fscore)

        ids = [c.id for c in available_clients]

        decimals = 5
        print("amostras: ", num_samples, accs, self.selected_clients)
        if len(num_samples) == 0:
            return None
        acc = round(sum(accs) / sum(num_samples), decimals)
        loss = round(sum(losses) / sum(num_samples), decimals)
        balanced_acc = round(sum(balanced_accs) / sum(num_samples), decimals)
        micro_fscore = round(sum(micro_fscores) / sum(num_samples), decimals)
        macro_fscore = round(sum(macro_fscores) / sum(num_samples), decimals)
        weighted_fscore = round(sum(weighted_fscores) / sum(num_samples), decimals)

        return {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc, "Loss": loss, 'Balanced accuracy': balanced_acc,
                'Micro f1-score': micro_fscore, 'Macro f1-score': macro_fscore, 'Weighted f1-score': weighted_fscore, "Alpha": alpha}

    # evaluate selected clients
    def evaluate(self, m, t, acc=None, loss=None):
        test_metrics, test_metrics_w = self.test_metrics(m, t)
        test_acc = test_metrics['Accuracy']
        test_std_acc = test_metrics["Std Accuracy"]
        test_std_loss = test_metrics["Std loss"]
        test_loss = test_metrics['Loss']
        test_auc = test_metrics['AUC']

        test_acc_w = test_metrics_w['Accuracy']
        test_std_acc_w = test_metrics_w["Std Accuracy"]
        test_std_loss_w = test_metrics_w["Std loss"]
        test_loss_w = test_metrics_w['Loss']
        test_auc_w = test_metrics_w['AUC']

        train_metrics = self.train_metrics(m, t)
        if train_metrics is not None:
            self.past_train_metrics_m[m][t] = train_metrics
            self.max_loss_m[m] = max(self.max_loss_m[m], train_metrics["Loss"])
        else:
            train_metrics = self.past_train_metrics_m[m][-1]

        # test_acc = sum(stats[2])*1.0 / sum(stats[1])
        # test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # train_loss = sum(train_metrics[2])*1.0 / sum(train_metrics[1])
        train_loss = train_metrics["Loss"]
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        for metric in test_metrics:
            if metric in ['ids', 'num_samples']:
                continue
            self.results_test_metrics[m][metric].append(test_metrics[metric])
            self.results_test_metrics_w[m][metric].append(test_metrics_w[metric])
        for metric in train_metrics:
            if metric in ['ids', 'num_samples']:
                continue
            self.results_train_metrics[m][metric].append(train_metrics[metric])
            # self.results_train_metrics_w[m][metric].append(train_metrics_w[metric])
        self.results_train_metrics[m]['Round'].append(t)
        self.results_train_metrics[m]['Fraction fit'].append(self.join_ratio)
        self.results_train_metrics[m]['# training clients'].append(len(self.selected_clients[m]))

        self.results_test_metrics[m]['Round'].append(t)
        self.results_test_metrics[m]['Fraction fit'].append(self.join_ratio)
        self.results_test_metrics[m]['# training clients'].append(len(self.selected_clients[m]))
        self.results_test_metrics[m]['training clients and models'].append(self.selected_clients[m].tolist())
        print("Tamanho do modelo: ", self.models_size, m)
        self.results_test_metrics[m]['model size'].append(self.models_size[m])

        self.results_test_metrics_w[m]['Round'].append(t)
        self.results_test_metrics_w[m]['Fraction fit'].append(self.join_ratio)
        self.results_test_metrics_w[m]['# training clients'].append(len(self.selected_clients[m]))
        self.results_test_metrics_w[m]['training clients and models'].append(list(self.selected_clients[m]))
        # print("ddd: ", self.models_size)
        self.results_test_metrics_w[m]['model size'].append(self.models_size[m])

        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Evaluate model {}".format(m))
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test Accuracy w: {:.4f}".format(test_acc_w))
        print("Averaged Test Loss: {:.4f}".format(test_loss))
        print("Averaged Test Loss w: {:.4f}".format(test_loss_w))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(test_std_acc))
        print("Std Test Loss: {:.4f}".format(test_std_loss))
        print("Std Test AUC: {:.4f}".format(np.std(test_auc)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        available_clients = self.get_available_clients(t, m)
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = available_clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def clients_test_metrics_preprocess(self, m, path):

        ids = []
        rounds = []
        accs = []
        losses = []
        samples = []
        balanced_accs = []
        micro_fscore = []
        macro_fscore = []
        weighted_fscore = []
        n_clients = []
        for i in range(self.num_clients):
            c_metrics = self.clients_test_metrics[i]
            ids += [i] * len(c_metrics["Round"][m])
            n_clients += [len(self.selected_clients[m])] * len(c_metrics["Round"][m])
            rounds += c_metrics["Round"][m]
            accs += c_metrics["Accuracy"][m]
            losses += c_metrics["Loss"][m]
            samples += c_metrics["Samples"][m]
            balanced_accs += c_metrics["Balanced accuracy"][m]
            micro_fscore += c_metrics["Micro f1-score"][m]
            macro_fscore += c_metrics["Macro f1-score"][m]
            weighted_fscore += c_metrics["Weighted f1-score"][m]

        print("Accuracy", len(accs), "Loss", len(losses), "Samples", len(samples), "Balanced accuracy", len(balanced_accs), "Micro f1-score", len(micro_fscore), "Macro f1-score", len(macro_fscore), "Weighted f1-score", len(weighted_fscore))

        df = pd.DataFrame({"Cid": ids, "# training clients": n_clients, "Round": rounds, "Accuracy": accs, "Loss": losses, "Samples": samples, "Balanced accuracy": balanced_accs, "Micro f1-score": micro_fscore, "Macro f1-score": macro_fscore, "Weighted f1-score": weighted_fscore}).round(5)
        df.to_csv(path.replace(".csv", "_clients.csv"), index=False)


    def _write_header(self, filename, header):

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as server_log_file:
            writer = csv.writer(server_log_file)
            writer.writerow(header)

    def _write_outputs(self, filename, data, mode='a'):

        for i in range(len(data)):
            for j in range(len(data[i])):
                element = data[i][j]
                if type(element) == float:
                    element = round(element, 6)
                    data[i][j] = element
        with open(filename, mode) as server_log_file:
            writer = csv.writer(server_log_file)
            writer.writerows(data)

    def _get_models_size(self):

        models_size = []
        for model in self.global_model:
            parameters = [i.detach().cpu().numpy() for i in model.parameters()]
            size = 0
            for i in range(len(parameters)):
                size += parameters[i].nbytes
            models_size.append(size)
        print("models size: ", models_size)
        self.models_size = models_size


