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

import random
import copy
import time
from flcore.clients.clientmultifedper import clientMultiFedPer
from flcore.servers.serverbase import Server
from threading import Thread

import numpy.typing as npt

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

from functools import reduce
from typing import List, Tuple

import numpy as np

from torch.nn.parameter import Parameter

import numpy.typing as npt

from io import BytesIO
from dataclasses import dataclass

import torch


class MultiFedPer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMultiFedPer)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        self._get_models_size()
        for t in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients(t)
            # self.send_models()
            print(self.selected_clients)

            for m in range(len(self.selected_clients)):


                for i in range(len(self.selected_clients[m])):
                    self.clients[self.selected_clients[m][i]].train(m, t, self.global_model[m])

                if t%self.eval_gap == 0:
                    print(f"\n-------------Round number: {t}-------------")
                    print("\nEvaluate global model for ", self.dataset[m])
                    self.evaluate(m, t=t)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and t%self.dlg_gap == 0:
                self.call_dlg(t)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        for m in range(self.M):
            self.save_results(m)
            self.save_global_model(m)

    #     for m in range(self.M):
    #         self.global_model[m] = [i.detach().cpu().numpy() for i in self.global_model[m].parameters()]
    #
    # def train(self):
    #     self._get_models_size()
    #     for t in range(1, self.global_rounds + 1):
    #         s_t = time.time()
    #         self.selected_clients = self.select_clients(t)
    #         # self.send_models()
    #         print(self.selected_clients)
    #
    #         for m in range(len(self.selected_clients)):
    #
    #             for i in range(len(self.selected_clients[m])):
    #                 self.clients[self.selected_clients[m][i]].train(m, t, self.global_model[m])
    #
    #             if t % self.eval_gap == 0:
    #                 print(f"\n-------------Round number: {t}-------------")
    #                 print("\nEvaluate global model for ", self.dataset[m])
    #                 self.evaluate(m, t=t)
    #
    #         self.receive_models()
    #         if self.dlg_eval and t % self.dlg_gap == 0:
    #             self.call_dlg(t)
    #         self.aggregate_parameters()
    #
    #         self.Budget.append(time.time() - s_t)
    #         print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
    #
    #         if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
    #             break
    #
    #     print("\nBest accuracy.")
    #     # self.print_(max(self.rs_test_acc), max(
    #     #     self.rs_train_acc), min(self.rs_train_loss))
    #     print(max(self.rs_test_acc))
    #     print("\nAverage time cost per round.")
    #     print(sum(self.Budget[1:]) / len(self.Budget[1:]))
    #
    #     for m in range(self.M):
    #         self.save_results(m)
    #         self.save_global_model(m)
    #
    # def receive_models(self):
    #     assert (len(self.selected_clients) > 0)
    #     print("aa: ", len(self.selected_clients), int((1 - self.client_drop_rate) * self.current_num_join_clients))
    #     # active_clients_m = random.sample(
    #     #     self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
    #
    #     self.uploaded_ids = []
    #     self.uploaded_weights = []
    #     self.uploaded_models = []
    #
    #     for m in range(len(self.selected_clients)):
    #         tot_samples = 0
    #         active_clients_m = self.selected_clients[m]
    #         print("m: ", m, " ativos: ", len(active_clients_m))
    #         m_uploaded_ids = []
    #         m_uploaded_weights = []
    #         m_uploaded_models = []
    #         for client_id in active_clients_m:
    #             client = self.clients[client_id]
    #             try:
    #                 client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
    #                                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
    #             except ZeroDivisionError:
    #                 client_time_cost = 0
    #             if client_time_cost <= self.time_threthold:
    #                 tot_samples += client.train_samples[m]
    #                 m_uploaded_ids.append(client.id)
    #                 m_uploaded_weights.append(client.train_samples[m])
    #                 m_uploaded_models.append(client.get_model_m(m))
    #         for i, w in enumerate(m_uploaded_weights):
    #             m_uploaded_weights[i] = w / tot_samples
    #
    #         print("modelo: ", m, " tam: ", len(m_uploaded_models))
    #
    #         self.uploaded_ids.append(copy.deepcopy(m_uploaded_ids))
    #         self.uploaded_weights.append(copy.deepcopy(m_uploaded_weights))
    #         self.uploaded_models.append(copy.deepcopy(m_uploaded_models))
    #
    # def aggregate_parameters(self):
    #     assert (len(self.uploaded_models) > 0)
    #
    #     for m in range(len(self.uploaded_models)):
    #         # self.global_model[m] = copy.deepcopy(self.uploaded_models[m][0])
    #         # for param in self.global_model[m].parameters():
    #         #     param.data.zero_()
    #         parameters_tuple = []
    #         for cid, w, client_model in zip(self.uploaded_ids[m], self.uploaded_weights[m], self.uploaded_models[m]):
    #             # self.add_parameters(w, client_model, m)
    #             parameters_tuple.append((client_model, w, cid))
    #
    #         agg_parameters = self.aggregate(parameters_tuple, m)
    #
    #         length = len(agg_parameters)
    #         # if type(self.global_model[m]) == list:
    #         #     global_model_parameters = [i for i in self.global_model[m]]
    #         # else:
    #         #     global_model_parameters = [i for i in self.global_model[m].parameters()]
    #         for i in range(length):
    #             client_param = agg_parameters[i]
    #             # global_model_parameters[i] = client_param.data.clone()
    #             self.global_model[m][i] = client_param
    #
    # def aggregate(self, results: List[Tuple[NDArrays, float]], m: int) -> NDArrays:
    #     """Compute weighted average."""
    #     # Calculate the total number of examples used during training
    #     num_examples_total = sum([num_examples for _, num_examples, cid in results])
    #
    #     # Create a list of weights, each multiplied by the related number of examples
    #     weighted_weights = [
    #         [layer * num_examples for layer in weights] for weights, num_examples, cid in results
    #     ]
    #
    #     # Compute average weights of each layer
    #     weights_prime: NDArrays = [
    #         reduce(np.add, layer_updates) / num_examples_total
    #         for layer_updates in zip(*weighted_weights)
    #     ]
    #
    #     #weights_prime = [Parameter(torch.Tensor(i.tolist())) for i in weights_prime]
    #     # weights_prime = np.array([i.tolist() for i in weights_prime])
    #
    #     return weights_prime
    #
    # def _get_models_size(self):
    #
    #     models_size = []
    #     for model in self.global_model:
    #         parameters = model
    #         size = 0
    #         for i in range(len(parameters)):
    #             size += parameters[i].nbytes
    #         models_size.append(size)
    #     print("models size: ", models_size)
    #     self.models_size = models_size