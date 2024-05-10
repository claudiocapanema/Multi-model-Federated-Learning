# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

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
from utils.data_utils import read_client_data
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
        self.device = args.device
        self.dataset = args.dataset
        # self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.M = len(self.dataset)
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.alpha = args.alpha
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
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

        self.train_metrics_names = ["Accuracy", "Loss", "AUC"]
        self.test_metrics_names = ["Accuracy", "Loss", "AUC", "Balanced accuracy", "Micro f1-score", "Weighted f1-score", "Macro f1-score", "Round", "Fraction fit"]
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.results_test_metrics = {m: {metric: [] for metric in self.test_metrics_names} for m in range(self.M)}

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = []
            test_data = []
            for m in range(self.M):
                train_data.append(len(read_client_data(self.dataset[m], i, args=self.args, is_train=True)))
                test_data.append(len(read_client_data(self.dataset[m], i, args=self.args, is_train=False)))
            client = clientObj(self.args,
                            id=i,
                            train_samples=train_data,
                            test_samples=test_data,
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

    def select_clients(self, t):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        np.random.seed(t)
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        selected_clients = [i.id for i in selected_clients]

        n = len(selected_clients) // self.M
        selected_clients = np.array_split(selected_clients, self.M)
        # selected_clients = [selected_clients[i:i+n] for i in range(0, len(selected_clients), n)]

        return selected_clients

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

    def aggregate(self, results: List[Tuple[NDArrays, float]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer.detach().cpu().numpy() * num_examples for layer in weights.parameters()] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        weights_prime = [Parameter(torch.Tensor(i.tolist())) for i in weights_prime]

        return weights_prime

    def aggregate_parameters(self):
        print("agr: ", (len(self.uploaded_models) > 0))
        assert (len(self.uploaded_models) > 0)

        for m in range(len(self.uploaded_models)):
            # self.global_model[m] = copy.deepcopy(self.uploaded_models[m][0])
            # for param in self.global_model[m].parameters():
            #     param.data.zero_()
            parameters_tuple = []
            for w, client_model in zip(self.uploaded_weights[m], self.uploaded_models[m]):
                # self.add_parameters(w, client_model, m)
                parameters_tuple.append((client_model, w))

            agg_parameters = self.aggregate(parameters_tuple)

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
        
    def save_results(self, m):
        algo = self.dataset[m] + "_" + self.algorithm
        result_path = """../results/clients_{}/alpha_{}/""".format(self.num_clients, self.alpha)
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

            print("me: ", self.rs_test_acc)
            self._write_header(file_path, header=header)
            self._write_outputs(file_path, data=data)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self, m):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            test_clients = self.new_clients
        else:
            test_clients = self.clients

        num_samples = []
        acc = []
        loss = []
        auc = []
        balanced_acc = []
        micro_fscore = []
        weighted_fscore = []
        macro_fscore = []
        for c in test_clients:
            test_acc, test_loss, test_num, test_auc, test_balanced_acc, test_micro_fscore, test_macro_fscore, test_weighted_fscore = c.test_metrics(m, copy.deepcopy(self.global_model[m].to(self.device)))
            acc.append(test_acc*test_num)
            auc.append(test_auc*test_num)
            num_samples.append(test_num)
            loss.append(test_loss*test_num)
            balanced_acc.append(test_balanced_acc*test_num)
            micro_fscore.append(test_micro_fscore*test_num)
            weighted_fscore.append(test_weighted_fscore*test_num)
            macro_fscore.append(test_macro_fscore*test_num)

            # print(test_num)
            # exit()

        ids = [c.id for c in test_clients]

        decimals = 5
        acc = round(sum(acc) / sum(num_samples), decimals)
        auc = round(sum(auc) / sum(num_samples), decimals)
        loss = round(sum(loss) / sum(num_samples), decimals)
        balanced_acc = round(sum(balanced_acc) / sum(num_samples), decimals)
        micro_fscore = round(sum(micro_fscore) / sum(num_samples), decimals)
        weighted_fscore = round(sum(weighted_fscore) / sum(num_samples), decimals)
        macro_fscore = round(sum(macro_fscore) / sum(num_samples), decimals)

        return {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc, 'AUC': auc,
                "Loss": loss, "Balanced accuracy": balanced_acc, "Micro f1-score": micro_fscore,
                "Weighted f1-score": weighted_fscore, "Macro f1-score": macro_fscore}

    def train_metrics(self, m):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics(m)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, m, t, acc=None, loss=None):
        test_metrics = self.test_metrics(m)
        test_acc = test_metrics['Accuracy']
        test_loss = test_metrics['Loss']
        test_auc = test_metrics['AUC']
        stats_train = self.train_metrics(m)

        # test_acc = sum(stats[2])*1.0 / sum(stats[1])
        # test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        for metric in test_metrics:
            if metric in ['ids', 'num_samples']:
                continue
            self.results_test_metrics[m][metric].append(test_metrics[metric])
        self.results_test_metrics[m]['Round'].append(t)
        self.results_test_metrics[m]['Fraction fit'].append(self.join_ratio)

        
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
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(test_acc)))
        print("Std Test AUC: {:.4f}".format(np.std(test_auc)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
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
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
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

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    # def test_metrics_new_clients(self, m):
    #     num_samples = []
    #     tot_correct = []
    #     tot_auc = []
    #     for c in self.new_clients:
    #         ct, ns, auc = c.test_metrics(m)
    #         tot_correct.append(ct*1.0)
    #         tot_auc.append(auc*ns)
    #         num_samples.append(ns)
    #
    #     ids = [c.id for c in self.new_clients]
    #
    #     return ids, num_samples, tot_correct, tot_auc

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
