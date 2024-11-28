# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import copy
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

import time
import sys
import math
import numpy as np
import random
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

from functools import reduce
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

import torch
from torch.nn.parameter import Parameter

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]


class MultiFedSpeedDynamic(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.fairness_weight = args.fairness_weight
        self.client_class_count = {m: {i: [] for i in range(self.num_clients)} for m in range(self.M)}
        self.clients_training_count = {m: [0 for i in range(self.num_clients)] for m in range(self.M)}
        self.current_training_class_count = [[0 for i in range(self.num_classes[j])] for j in range(self.M)]
        self.non_iid_degree = {m: {'unique local classes': 0, 'samples': 0} for m in range(self.M)}
        self.client_selection_model_weight = np.array([0] * self.M)
        self.clients_cosine_similarities = {m: np.array([0 for i in range(self.num_classes[m])]) for m in range(self.M)}
        self.clients_cosine_similarities_with_current_model = {m: [0 for i in range(self.num_clients)] for m in range(self.M)}
        self.clients_training_round_per_model = {cid: {m: [] for m in range(self.M)} for cid in range(self.num_clients)}
        self.clients_rounds_since_last_training = {m: np.array([0 for i in range(self.num_clients)]) for m in range(self.M)}
        self.clients_rounds_since_last_training_probability = {m: np.array([0 for i in range(self.num_clients)]) for m in range(self.M)}
        self.rounds_client_trained_model = {cid: {m: [] for m in range(self.M)} for cid in range(self.num_clients)}
        self.minimum_training_clients_per_model = {0.1: 1, 0.2: 2, 0.3: 2}[self.join_ratio]
        self.minimum_training_clients_per_model_percentage = self.minimum_training_clients_per_model / self.num_join_clients
        self.use_cold_start_m = np.array([True for m in range(self.M)])
        self.cold_start_max_non_iid_level = 1
        self.cold_start_training_level = np.array([0] * self.M)
        self.minimum_training_level = 2 / self.num_clients
        self.models_semi_convergence_rounds_n_clients = {m: [] for m in range(self.M)}
        self.max_n_training_clients = 10
        # Semi convergence detection window of rounds
        self.tw = []
        for d in self.dataset:
            self.tw.append({"WISDM-W": 4, "WISDM-P": 4, "ImageNet": 4, "CIFAR10": 4, "ImageNet_v2": 4}[d])
        self.models_semi_convergence_min_n_training_clients = {m: self.minimum_training_clients_per_model for m in range(self.M)}
        self.models_semi_convergence_min_n_training_clients_percentage = {m: self.minimum_training_clients_per_model/self.num_join_clients for m in
                                                               range(self.M)}
        self.models_semi_convergence_flag = [False] * self.M
        self.models_convergence_flag = [False] * self.M
        self.models_semi_convergence_count = [0] * self.M
        self.models_semi_convergence_training_probability = [0] * self.M
        self.training_clients_per_model = np.array([0] * self.M)
        self.training_clients_per_model_per_round = {m: [] for m in range(self.M)}
        self.rounds_since_last_semi_convergence = {m: 0 for m in range(self.M)}
        self.unique_count_samples = {m: np.array([0 for i in range(self.num_classes[m])]) for m in range(self.M)}
        self.similarity_matrix = np.zeros((self.M, self.num_clients))
        self.accuracy_gain_models = {m: [] for m in range(self.M)}
        self.stop_cpd = [False for m in range(self.M)]
        self.re_per_model = int(args.reduction)
        # self.selected_clients_cosine = {m: {} for m in range(self.M)}

    def cold_start_selection(self):

        prob = [np.array([0 for i in range(self.num_clients)]) for j in range(self.M)]
        if (self.use_cold_start_m).all() == False:
            return prob, self.use_cold_start_m
        use_cold_start = np.array([False] * self.M)
        for m in range(self.M):
            for i in range(self.num_clients):
                if self.clients_training_count[m][i] == 0:
                    prob[m][i] = 1
                    self.use_cold_start_m[m] = True
                    use_cold_start[m] = True

        if np.sum(prob) == 0:
            self.use_cold_start_m = np.array([False for m in range(self.M)])

        return prob, use_cold_start

    def uniform_selection(self, t):

        if t == 1:
            return [np.array([1 / self.num_clients for i in range(self.num_clients)]) for j in range(self.M)]

        prob = [[0 for i in range(self.num_clients)] for j in range(self.M)]

        for m in range(self.M):
            clients = np.argsort(self.clients_training_count[m])
            p = np.array(self.clients_training_count[m]) / np.sum(self.clients_training_count[m])
            print("soma: ", p, self.clients_training_count[m])
            p = 1 - p # high probability for less training clients
            p = (-3 + np.power(2, 2 * p)) # Keeps high probability when it is very close to 1
            p[p<0] = 0 # Avoid negative probability
            prob[m] = p
            # clients = clients[0:c]
            # prob[m] = np.array([1 if i in clients else 0 for i in range(self.num_clients)])

        print("Uniform")
        print(prob)

        return prob

    def data_quality_selection(self):

        prob = [[0 for i in range(self.num_clients)] for j in range(self.M)]
        cosine_similarities = [[i for i in range(self.num_clients)] for j in range(self.M)]

        for m in range(self.M):
            for i in range(self.num_clients):

                # cosine_similarities[m][i] = np.dot(self.unique_count_samples[m],self.client_class_count[m][i])/(np.linalg.norm(self.unique_count_samples[m])*np.linalg.norm(self.client_class_count[m][i]))
                cosine_similarities[m][i] = cosine_similarity(np.array([self.unique_count_samples[m]]), np.array([self.client_class_count[m][i]]))[0][0]

        print("si: ")
        print(cosine_similarities)
        # exit()

        clients_m_p = [[i for i in range(self.num_clients)] for j in range(self.M)]

        for m in range(self.M):
            # clients_m_p[m] = np.array(cosine_similarities[m]) / np.sum(cosine_similarities[m])
            clients_m_p[m] = np.array(cosine_similarities[m])

        print("Quality")
        # print(clients_m_p)
        self.clients_cosine_similarities = clients_m_p
        return clients_m_p

    # def select_clients(self, t):
    #     np.random.seed(t)
    #     if t == 1:
    #         return super().select_clients(t)
    #     else:
    #         if self.random_join_ratio:
    #             self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
    #         else:
    #             self.current_num_join_clients = self.num_join_clients
    #         np.random.seed(t)
    #         selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
    #         selected_clients_m = [[] for i in range(self.M)]
    #         non_iid_weight_m = []
    #         untrained_clients_m = []
    #         for m in range(self.M):
    #             non_iid_weight_m.append(1 - self.non_iid_degree[m] / 100)
    #             clients_training_count = np.array(self.clients_training_count[m])
    #             clients_homogeneity_training_degree = []
    #             # untrained_clients_m.append()
    #         non_iid_weight_m = np.array(non_iid_weight_m)
    #         non_iid_weight_m = non_iid_weight_m / np.sum(non_iid_weight_m)
    #         for m in range(self.M):
    #             p_class = np.array(self.clients_training_count[m])
    #             non_iid_weight = non_iid_weight_m[m]
    #             p_class = (1 - (p_class / np.sum(p_class)) * non_iid_weight)
    #             p = p / np.sum(p)
    #             s_c_m = list(np.random.choice(self.clients, self.current_num_join_clients, p=p, replace=False))
    #
    #
    #
    #         for client in selected_clients:
    #             client_losses = []
    #             for metrics_m in client.test_metrics_list_dict:
    #                 client_losses.append(metrics_m['Loss'] * metrics_m['Samples'])
    #             client_losses = np.array(client_losses)
    #             client_losses = (np.power(client_losses, self.fairness_weight - 1)) / np.sum(client_losses)
    #             client_losses = client_losses / np.sum(client_losses)
    #             print("probal: ", client_losses)
    #             m = np.random.choice([i for i in range(self.M)], p=client_losses)
    #             selected_clients_m[m].append(client.id)
    #
    #     print("Modelos clientes: ", selected_clients_m)
    #
    #     return selected_clients_m

    def calculate_accuracy_gain_models(self):

        acc_gain_models = {m: [] for m in range(self.M)}
        relative_acc_gain_models = {m: [] for m in range(self.M)}
        acc_gain_efficiency = {m: [] for m in range(self.M)}
        for m in range(self.M):
            global_acc = self.results_test_metrics[m]["Loss"]
            for i in range(len(global_acc)):
                n_training_clinets = self.results_train_metrics[m]['# training clients'][-1]
                if i == 0:
                    acc_gain_models[m].append(global_acc[i])
                    relative_acc_gain_models[m].append(global_acc[i])
                    if n_training_clinets > 0:
                        acc_gain_efficiency[m].append(global_acc[i] / n_training_clinets)
                    else:
                        acc_gain_efficiency[m].append(0)
                else:
                    acc_gain_models[m].append(global_acc[i] - global_acc[i-1])
                    relative_acc_gain_models[m].append(-(global_acc[i] - global_acc[i-1]) / global_acc[i-1])
                    if n_training_clinets > 0:
                        acc_gain_efficiency[m].append(-(global_acc[-1] - global_acc[i]) / n_training_clinets)
                    else:
                        acc_gain_efficiency[m].append(0)

        self.accuracy_gain_models = acc_gain_models
        self.relative_accuracy_gain_models = relative_acc_gain_models
        self.accuracy_gain_efficiency = acc_gain_efficiency
        print("Ganhos de acurácia: ")
        print(self.accuracy_gain_models)
        print("Ganhos relativos de acurácia: ")
        print(self.relative_accuracy_gain_models)
        print("Ganhos de acurácia efficiência: ")
        print(self.accuracy_gain_efficiency)
        return self.accuracy_gain_models

    def select_clients(self, t):
        try:
            # np.random.seed(t)
            # if self.random_join_ratio:
            #     self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            # else:
            #     self.current_num_join_clients = self.num_join_clients
            # np.random.seed(t)
            # selected_clients_m = [[] for i in range(self.M)]
            # selected_clients = []
            # n_clients_m = [0] * self.M
            # n_clients_selected = 0
            # remaining_clients = int(self.num_clients * self.join_ratio)
            # uniform_selection_m = self.uniform_selection(t)
            # # data_quality_selection_m = self.data_quality_selection()
            # data_quality_selection_m = self.clients_cosine_similarities
            # prob_cold_start_m, use_cold_start_m = self.cold_start_selection()
            # cold_start_clients_m = [np.array([]) for i in range(self.M)]
            # print(self.client_selection_model_weight)
            # print(self.cold_start_training_level)
            # print(use_cold_start_m)
            # clients_cumulative_prob_m = np.zeros((self.M, self.num_clients))
            # total = 0
            # for client in range(self.num_clients):
            #     for m in range(self.M):
            #         clients_cumulative_prob_m[m][client] = uniform_selection_m[m][client] + data_quality_selection_m[m][client]
            #         total += clients_cumulative_prob_m[m][client]
            #
            # m_order = np.argwhere(use_cold_start_m == True)
            # m_order = m_order.flatten()
            # print("prim: ", m_order)
            # for i in range(self.M):
            #     if i not in m_order:
            #         m_order = np.append(m_order, [i])
            # print("m order: ", m_order)
            # for i in range(len(m_order)):
            #     m = m_order[i]
            #     if use_cold_start_m[m]:
            #         n_selected_clients_m = math.ceil(self.num_clients*self.join_ratio*self.cold_start_training_level[m])
            #         if n_selected_clients_m > remaining_clients:
            #             n_selected_clients_m = remaining_clients
            #         print("co: ", self.num_clients, self.join_ratio, self.cold_start_training_level[m])
            #         p = prob_cold_start_m[m]
            #         p = np.array([p[i] if i not in selected_clients else 0 for i in range(len(p))])
            #         p = np.argwhere(p == 1).flatten()[:n_selected_clients_m]
            #         # cold_start_clients_m[m] = np.argwhere(prob_cold_start_m[m][prob_cold_start_m[m] == 1]).flatten()[:n_selected_clients_m]
            #         remaining_clients = n_selected_clients_m - len(p)
            #         print("faltantes: ", n_selected_clients_m - len(p))
            #         if n_selected_clients_m - len(p) > 0:
            #             # prob_m = uniform_selection_m[m] * (1 - self.client_selection_model_weight[m]) + data_quality_selection_m[m] * self.client_selection_model_weight[m]
            #             prob_m = clients_cumulative_prob_m[m]
            #             prob_m[p] = 0
            #             prob_m = np.array([prob_m[i] if i not in selected_clients else 0 for i in range(len(prob_m))])
            #             prob_m = np.argsort(prob_m)[-(n_selected_clients_m - len(p)):]
            #             print("f adicionados: ", len(prob_m))
            #             # prob_m = np.array([i if i not in prob_m else 0 for i in prob_m])
            #             # prob_m = np.argwhere(prob_m > 0).flatten()[]
            #         else:
            #             prob_m = np.array([])
            #         if len(p) > 0 and len(prob_m) == 0:
            #             selected_clients_m[m] = p
            #         else:
            #             selected_clients_m[m] = np.array(list(p) + list(prob_m))
            #
            #     else:
            #         # print("f", self.num_clients * self.join_ratio - n_clients_selected)
            #         if True not in use_cold_start_m:
            #             n_clients = math.ceil(self.num_clients * self.join_ratio * self.client_selection_model_weight[m])
            #         else:
            #             n_clients = min(math.ceil(self.num_clients * self.join_ratio - n_clients_selected), math.ceil(self.num_clients * self.join_ratio * self.client_selection_model_weight[m]))
            #         if n_clients > remaining_clients or (n_clients < remaining_clients and i == self.M - 1):
            #             print("ent", remaining_clients)
            #             n_clients = remaining_clients
            #         prob_m = clients_cumulative_prob_m[m]
            #
            #         print(prob_m)
            #         prob_m = np.array([prob_m[i] if i not in selected_clients else 0 for i in range(len(prob_m))])
            #         prob_m = prob_m / np.sum(prob_m)
            #         print("ee: ", prob_m)
            #         prob_m = list(np.random.choice([i for i in range(len(self.clients))], n_clients, replace=False, p=prob_m))
            #         selected_clients_m[m] = prob_m
            #
            #         # np.random.seed(t)
            #         # if self.random_join_ratio:
            #         #     self.current_num_join_clients = \
            #         #     np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
            #         # else:
            #         #     self.current_num_join_clients = self.num_join_clients
            #         # np.random.seed(t)
            #         # selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
            #         # selected_clients = [i.id for i in selected_clients]
            #         #
            #         # n = len(selected_clients) // self.M
            #         # sc = np.array_split(selected_clients, self.M)
            #         # # sc = [np.array(selected_clients[0:12])]
            #         # # sc.append(np.array(selected_clients[12:]))
            #         # selected_clients_m[m] = sc[m]
            #
            #     selected_clients += list(selected_clients_m[m])
            #     n_clients_selected += len(selected_clients_m[m])
            #     remaining_clients = int(self.num_clients * self.join_ratio) - n_clients_selected
            #     print("m: ", m, " remaining: ", remaining_clients)

            g = torch.Generator()
            g.manual_seed(t)
            np.random.seed(t)
            random.seed(t)
            if t == 1:
                self.previous_selected_clients = super().select_clients(t)
                n_clients_m = {m: 0 for m in range(self.M)}
                for m in range(self.M):
                    n_clients_m[m] = len(self.previous_selected_clients[m])
                    self.training_clients_per_model_per_round[m].append(n_clients_m[m])
                    for c_id in self.previous_selected_clients[m]:
                        self.clients_training_round_per_model[c_id][m].append(t)

                    self.rounds_since_last_semi_convergence[m] += 1

                return self.previous_selected_clients
            else:
                if self.random_join_ratio:
                    self.current_num_join_clients = \
                    np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
                else:
                    self.current_num_join_clients = self.num_join_clients
                np.random.seed(t)
                selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
                selected_clients_m = [[] for i in range(self.M)]

                self.calculate_accuracy_gain_models()

                clients_losses = []
                metric = "Accuracy"
                # losses_m = {m: [] for m in range(self.M)}
                # samples_m = {m: [] for m in range(self.M)}
                # for client in self.clients:
                #     for m in range(len(client.train_metrics_list_dict)):
                #         local_loss = self.clients_train_metrics[client.id]["Loss"][m]
                #         losses_m[m].append(local_loss)
                #         samples = self.clients_train_metrics[client.id]["Samples"][m]
                #         samples_m[m].append(samples)
                # for m in range(self.M):
                #     print(losses_m[m])
                #     losses_m[m] = np.max(losses_m[m])
                #     samples_m[m] = np.max(samples_m[m])

                for client in self.clients:
                    cid = client.id
                    client_losses = []
                    improvements = []
                    for m in range(len(client.train_metrics_list_dict)):
                        rounds = self.clients_training_round_per_model[cid][m][-2:]
                        rounds = np.array(rounds) - 1
                        print("r tre", rounds)
                        metrics_m = client.train_metrics_list_dict[m]
                        local_acc = np.array(self.clients_train_metrics[client.id][metric][m])
                        local_loss = np.array(self.clients_train_metrics[client.id]["Loss"][m])
                        global_acc = self.results_train_metrics[m][metric]

                        if len(local_acc) >= 2 and len(self.clients_training_round_per_model[cid][m]) >= 2:
                            local_acc = local_acc[rounds]
                            local_loss = local_loss[rounds]
                            # tr = self.clients_training_round_per_model[cid][m]
                            local_acc_diff = (local_acc[-1] - local_acc[-2])
                            local_loss_diff = (local_loss[-2] - local_loss[-1])

                            if local_loss_diff == 0 or local_loss[-2] == 0:
                                local_loss_relative_diff = 0
                            else:
                                local_loss_relative_diff = local_loss[-1]/local_loss[-2]
                            # else:
                            #     local_diff = (local_acc[-1] - local_acc[-2]) / q
                            if local_acc_diff > 0:
                                improvement = local_acc_diff
                                improvements.append(improvement)
                            else:
                                local_acc_diff = 0.1 # 0.1 ou 0.01
                            if local_acc[-1] == 0:
                                relative_diff = 0.001
                            else:
                                relative_diff = local_acc_diff / local_acc[-1]
                            local_acc_diff = relative_diff * relative_diff + local_acc_diff
                        elif len(local_acc) == 1:
                            local_acc_diff = local_acc[-1]
                            local_loss_diff = local_loss[-1]
                            local_loss_relative_diff = 1
                        else:
                            local_acc_diff = 1
                            local_loss_diff = 1
                            local_loss_relative_diff = 1
                        local_acc_diff = max(local_acc_diff, 0.01)
                        local_loss_diff = max(local_loss_diff, 0.01)
                        if local_loss_relative_diff <= 0:
                            local_loss_relative_diff = 0
                        local_loss_relative_diff = min(local_loss_relative_diff, 1)
                        print("ant:1: ", metrics_m['Samples'],local_acc_diff, (metrics_m['Samples']) * local_loss_relative_diff)
                        client_losses.append((metrics_m['Samples'] ) * local_loss_diff)
                    client_losses = np.array(client_losses)
                    # client_losses = (client_losses / np.sum(client_losses))
                    if np.isnan(client_losses).any():
                        client_losses = np.ones(len(client_losses))
                    clients_losses.append(client_losses)

                clients_losses = np.array(clients_losses)
                print("fo: ", clients_losses.shape)
                print("Valor das losses: ", clients_losses)

                prob_cold_start_m, use_cold_start_m = self.cold_start_selection()
                print("cold: ", t, use_cold_start_m)
                print(prob_cold_start_m)
                total_clients = 0
                remaining_clients_per_model = np.array([1/self.M for i in range(self.M)])
                print("Clientes restantes (%): ", remaining_clients_per_model)
                remaining_clients_per_model[np.where(remaining_clients_per_model < self.minimum_training_clients_per_model_percentage)] = self.minimum_training_clients_per_model_percentage
                print("inter: ", remaining_clients_per_model)
                num_join_clients = self.num_join_clients
                remaining_clients_per_model = np.array(remaining_clients_per_model * num_join_clients, dtype=int)
                index = 1
                while np.sum(remaining_clients_per_model) < num_join_clients:
                    remaining_clients_per_model[index] += 1
                    index += 1
                    if index == len(remaining_clients_per_model):
                        index = 0
                print("num joint clients antes: ", num_join_clients, self.models_semi_convergence_count)
                for m in range(self.M):
                    if self.models_semi_convergence_flag[m]:
                        num_join_clients -= min(self.models_semi_convergence_count[m], self.re_per_model)
                        remaining_clients_per_model[m] -= min(self.models_semi_convergence_count[m], self.re_per_model)
                # if all(self.models_semi_convergence_flag) == True:
                #     num_join_clients -= min(min(self.models_semi_convergence_count), 5)
                print("num join clients: ", t, num_join_clients)
                # remaining_clients_per_model = np.array(remaining_clients_per_model * num_join_clients, dtype=int)

                print("Clientes resta", remaining_clients_per_model)


                loops = 0
                # include missing clients
                if np.sum(remaining_clients_per_model) < num_join_clients:
                    print("me")
                    diff = num_join_clients - np.sum(remaining_clients_per_model)
                    i = 0
                    while np.sum(remaining_clients_per_model) < num_join_clients and diff > 0 and loops < 20:
                        if remaining_clients_per_model[i] != 3:
                            remaining_clients_per_model[i] += 1
                            diff -= 1
                        i += 1
                        if i >= self.M:
                            i = 0
                        loops += 1
                elif np.sum(remaining_clients_per_model) > num_join_clients:
                    print("mais clientes")
                    diff = np.sum(remaining_clients_per_model) - num_join_clients
                    i = 0
                    while np.sum(remaining_clients_per_model) > num_join_clients and diff > 0 and loops < 20:
                        if remaining_clients_per_model[i] > self.models_semi_convergence_min_n_training_clients[i]:
                            remaining_clients_per_model[i] -= 1
                            print("redu")
                            diff -= 1
                        i += 1
                        if i >= self.M:
                            i = 0
                        loops += 1

                print("r", t)
                print("Clientes res: ", remaining_clients_per_model)
                """semi-convergence detection"""
                min_clients = np.min(list(self.models_semi_convergence_min_n_training_clients.values()))
                flag = True
                for m in range(self.M):
                    if not self.stop_cpd[m]:
                        self.rounds_since_last_semi_convergence[m] += 1
                        global_acc = np.array(self.relative_accuracy_gain_models[m][-self.tw[m]:])

                        """Stop CPD"""
                        losses = self.results_test_metrics[m]["Loss"][-self.tw[m]:]
                        cpd_min_occurrences = int(self.tw[m] * 0.5)
                        cpd_max_occurrences = int(self.tw[m] * 1)
                        if len(np.argwhere(global_acc < 0)[-self.tw[m]:]) == self.tw[m] and self.models_semi_convergence_flag[m]:
                            self.models_semi_convergence_count[m] -= 1
                            self.models_semi_convergence_count[m] = max(self.models_semi_convergence_count[m], 0)
                            self.stop_cpd[m] = True
                            print("Stop CPD, model: ", m, self.dataset[m], self.models_semi_convergence_count[m], t)
                            print("Sequence of losses: ", losses)
                        print("globl: ", global_acc)
                        if t < 40:
                            window = 7
                        else:
                            window = 5
                        idxs = np.argwhere(global_acc < 0)
                        diff = self.models_semi_convergence_min_n_training_clients[m] - min_clients
                        if m == 0:
                            print("Checagem WISDM ", t)
                            print(len(idxs) <= int(self.tw[m] * 0.75), len(idxs) >= int(self.tw[m] * 0.25), diff < 1, self.rounds_since_last_semi_convergence[m] >= 4)
                        if len(idxs) <= int(self.tw[m] * 0.75) and len(idxs) >= int(self.tw[m] * 0.25) and diff < 1 and self.rounds_since_last_semi_convergence[m] >= 4:
                            self.rounds_since_last_semi_convergence[m] = 0
                            idxs_rounds = np.array(idxs, dtype=np.int32).flatten()
                            print("indices: ", idxs_rounds, idxs_rounds[-1])
                            print("ab: ", self.training_clients_per_model_per_round[m])
                            self.models_semi_convergence_rounds_n_clients[m].append({'round': t - 2, 'n_training_clients':
                                self.training_clients_per_model_per_round[m][t-2]})
                            # more clients are trained for the semi converged model
                            print("treinados na rodada passada: ", m , self.training_clients_per_model_per_round[m][t-2])
                            # self.models_semi_convergence_min_n_training_clients[m] = min(
                            #     self.models_semi_convergence_min_n_training_clients[m] + 1, 4)

                            if flag:
                                self.models_semi_convergence_flag[m] = True
                                self.models_semi_convergence_count[m] += 1
                                flag = False
                        # updates the probability of training a semi converged model
                        self.models_semi_convergence_training_probability[m] = max((num_join_clients - self.models_semi_convergence_min_n_training_clients[
                                                                                    m]) / num_join_clients, 1/num_join_clients)
                    # self.models_semi_convergence_min_n_training_clients[m] = min(
                    #     self.models_semi_convergence_min_n_training_clients[m], num_join_clients)

                print("semi convergen: ", self.models_semi_convergence_flag, self.models_semi_convergence_count, t)
                print("semi conver prob: ", self.models_semi_convergence_training_probability)
                print("min training: ", self.models_semi_convergence_min_n_training_clients)
                print("rem: ", remaining_clients_per_model)

                """Number of models to train"""
                models_semi_convergence_min_n_training_client = copy.deepcopy(
                    list(self.models_semi_convergence_min_n_training_clients.values()))
                n_models_to_train = self.M
                print("mi train", models_semi_convergence_min_n_training_client, num_join_clients, self.models_semi_convergence_flag)
                if np.sum(models_semi_convergence_min_n_training_client) > num_join_clients and True in self.models_semi_convergence_flag:
                    excedent = np.sum(models_semi_convergence_min_n_training_client) - num_join_clients
                    ordered = np.random.choice(a=[i for i in range(self.M)], size=1)
                    print("ord: ", ordered)
                    n_models_to_train -= 1
                    models_semi_convergence_min_n_training_client[ordered[0]] = 0

                for m in range(self.M):
                    if models_semi_convergence_min_n_training_client[m] == 0:
                        remaining_clients_per_model[m] = 0
                    elif remaining_clients_per_model[m] < models_semi_convergence_min_n_training_client[m]:
                        remaining_clients_per_model[m] = models_semi_convergence_min_n_training_client[m]

                print("treinar d: ", n_models_to_train)


                # if t>= 20:
                #     if t % 2 == 0:
                #         remaining_clients_per_model = [6, 3]
                #     else:
                #         remaining_clients_per_model = [0, 7]
                print("remaining: ", remaining_clients_per_model)
                initial_remaining = copy.deepcopy(remaining_clients_per_model)
                print("join clients: ", num_join_clients)
                sc = []
                available_clients = {i: clients_losses[i] for i in range(self.num_clients)}
                initial_avialable = copy.deepcopy(available_clients)
                l = 0
                iteracoes = 0
                initial = copy.deepcopy(use_cold_start_m)
                while total_clients < num_join_clients and iteracoes < 20:
                    print("inicio disponiveis: ", len(available_clients))
                    for m in range(self.M):
                        print("m: ", m)
                        iteracoes += 1
                        print("Inicio iteracao: ", iteracoes)
                        if total_clients < num_join_clients and remaining_clients_per_model[m] > 0:
                            print("Primeiro if")
                            if use_cold_start_m[m]:
                                print("Segundo if (cold start)")
                                n_selected_clients_m = remaining_clients_per_model[m]
                                p = np.argwhere(prob_cold_start_m[m] == 1)
                                cid = -1
                                for e in p:
                                    if e[0] in available_clients:
                                        cid = e[0]
                                        break
                                print("cid: ", cid)
                                if cid == -1:
                                    use_cold_start_m[m] = False
                                    continue
                                available_clients.pop(cid)
                                selected_clients_m[m].append(cid)
                                total_clients += 1
                                remaining_clients_per_model[m] -= 1
                                print("n disponiveis: ", len(available_clients))

                            else:
                                print("sem cold start")
                                # print("tudo: ", remaining_clients, remaining_clients.shape, clients_losses.shape)
                                m_clients_losses = np.array([available_clients[i][m] for i in available_clients])
                                # m_clients_losses = m_clients_losses[available_clients]
                                cid = np.random.choice(list(available_clients.keys()), p=(m_clients_losses / np.sum(m_clients_losses)))
                                available_clients.pop(cid)
                                selected_clients_m[m].append(cid)
                                # available_clients.remove(cid)
                                sc.append(cid)
                                remaining_clients_per_model[m] -= 1
                                total_clients += 1
                                print("n disponiveis: ", len(available_clients))
                        else:

                            print("Nao adicionou ao modelo: ", m, """ restantes para o modelo {}: {}""".format(m, remaining_clients_per_model[m]))

                            print("n disponiveis: ", len(available_clients), selected_clients_m)
                            print(total_clients)
                            if l > 20 or np.sum(remaining_clients_per_model) == 0:
                                break
                            else:
                                print("faltantes por modelo: ", remaining_clients_per_model)
                                # exit()
                            l = l + 1

                        print("""Fim iteracao {} \nclientes totais {} clientes restantes {} \nclientes disponíveis {}""".format(iteracoes, total_clients, remaining_clients_per_model, available_clients))



                selected_clients = list(
                    np.random.choice(list(initial_avialable.keys()), initial_remaining, replace=False))

                print("Modelos clientes: ", selected_clients_m, self.M, selected_clients, initial_remaining, initial)
                if all(initial) == False:
                    for m in range(self.M):
                        print("Usou randomico para: ", m, t)
                        selected_clients_m[m] = selected_clients[m]

                print("Modelos clientes f: ", selected_clients_m)


                print("Selecionados: ", t, sum([len(i) for i in selected_clients_m]), selected_clients_m)
                # print("Quantidade: ", n_clients_selected)
                # exit()
                self.previous_selected_clients = selected_clients_m
                for m in range(self.M):
                    n = len(self.previous_selected_clients[m])
                    self.training_clients_per_model_per_round[m].append(n)

                for m in range(self.M):
                    self.clients_rounds_since_last_training[m] += 1
                    for c_id in self.previous_selected_clients[m]:
                        self.clients_training_round_per_model[c_id][m].append(t)
                        self.clients_rounds_since_last_training[m][c_id] = 0

                    self.clients_rounds_since_last_training_probability[m] = self.clients_rounds_since_last_training[m] / np.max(self.clients_rounds_since_last_training[m])
                    self.clients_rounds_since_last_training_probability[m] = self.clients_rounds_since_last_training_probability[m] / np.sum(self.clients_rounds_since_last_training_probability[m])

                print("selecionado: ", selected_clients_m, available_clients, remaining_clients_per_model)
                return selected_clients_m

        except Exception as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print("""{} \n{} \n{} \n{} \n{}""".format(len(available_clients), selected_clients_m, clients_losses, remaining_clients_per_model, total_clients))

    def train(self):

        self._get_models_size()
        for m in range(self.M):
            for i in range(self.num_clients):
                self.client_class_count[m][i] = self.clients[i].train_class_count[m]
                print("no train: ", " cliente: ", i, " modelo: ", m, " train class count: ", self.clients[i].train_class_count[m])

        # self.detect_non_iid_degree()

        # print("Non iid degree")
        # print("#########")
        # print("""M1: {}\nM2: {}""".format(self.non_iid_degree[0], self.non_iid_degree[1]))

        for t in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients(t)
            self.current_training_class_count = [np.array([0 for i in range(self.num_classes[j])]) for j in range(self.M)]
            # self.send_models()
            print(self.selected_clients)
            for m in range(len(self.selected_clients)):


                for i in range(len(self.selected_clients[m])):
                    client_id = self.selected_clients[m][i]
                    self.clients_training_round_per_model[client_id][m].append(t)
                    self.clients[client_id].train(m, self.global_model[m], self.clients_cosine_similarities_with_current_model[m][client_id])
                    self.clients_training_count[m][client_id] += 1
                    self.current_training_class_count[m] += self.clients[client_id].train_class_count[m]

                if t%self.eval_gap == 0:
                    print(f"\n-------------Round number: {t}-------------")
                    print("\nEvaluate global model for ", self.dataset[m])
                    self.evaluate(m, t=t)

                self.current_training_class_count[m] = np.round(self.current_training_class_count[m] / np.sum(self.current_training_class_count[m]), 2)
                # print("current training class count ", self.current_training_class_count[m])

                # print("######")
                # print("Training count: ", self.clients_training_count)

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

            if self.num_new_clients > 0:
                self.eval_new_clients = True
                self.set_new_clients(clientFedNome)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(m, t=t)

    def detect_non_iid_degree(self):

        unique_count_samples = {m: np.array([0 for i in range(self.num_classes[m])]) for m in range(self.M)}
        training_clients = self.num_join_clients
        for m in range(self.M):
            read_alpha = []
            read_dataset = []
            read_id = []
            read_num_classes = []
            read_num_samples = []

            read_std_alpha = []
            read_std_dataset = []
            read_num_samples_std = []
            samples = []

            for client_ in self.client_class_count[m]:
                client_class_count = self.client_class_count[m][client_]
                # client_training_count = self.clients_training_count[m][client]
                unique_classes = np.array(client_class_count)
                unique_classes_count = len(unique_classes[unique_classes > 0])
                read_num_classes.append((100 * unique_classes_count) / self.num_classes[m])
                unique_count = {i: 0 for i in range(self.num_classes[m])}

                # unique, count = np.unique(y_train, return_counts=True)
                balanced_samples = np.sum(client_class_count) / self.num_classes[m]
                # data_unique_count_dict = dict(zip(unique, count))
                for class_ in range(len(client_class_count)):
                    unique_count[class_] = (100 * (
                    (balanced_samples - client_class_count[class_]))) / balanced_samples
                    unique_count_samples[m][class_] += client_class_count[class_]
            d = np.array(list(unique_count.values()))
            s = np.mean(d[d > 0])
            samples.append(s)
            read_num_samples += list(samples)
            read_num_samples_std.append(np.std(samples))



            self.non_iid_degree[m]['unique local classes'] = float(sum(read_num_classes) / (len(read_num_classes)))
            self.client_selection_model_weight[m] = float(sum(read_num_classes) / (len(read_num_classes)))
            model_training_clients = int(training_clients * float(sum(read_num_samples)) / len(read_num_samples))
            self.training_clients_per_model[m] = max([model_training_clients, self.minimum_training_clients_per_model])
        # print(float(sum(read_num_classes) / (len(read_num_classes))))
        # exit()

        self.unique_count_samples = unique_count_samples
        for m in range(self.M):
            unique_count_samples[m] = unique_count_samples[m] / np.sum(unique_count_samples[m])
        print("samplles")
        print(self.unique_count_samples)
        print(unique_count_samples)
        print(self.client_selection_model_weight)
        # self.data_quality_selection()
        # print(self.clients_cosine_similarities)
        # ruim
        # self.client_selection_model_weight[0] = 0.4
        # self.client_selection_model_weight[1] = 0.6
        # self.client_selection_model_weight = np.array([0.7, 0.3])
        # self.client_selection_model_weight[0] = 0.2
        # self.client_selection_model_weight[1] = 0.8
        # self.client_selection_model_weight = self.client_selection_model_weight / np.sum(self.client_selection_model_weight)
        print("client selection model weight: ", self.client_selection_model_weight)
        self.cold_start_training_level = 1 - self.client_selection_model_weight
        for i in range(len(self.cold_start_training_level)):
            if self.client_selection_model_weight[i] < self.minimum_training_level:
                self.client_selection_model_weight[i] = self.minimum_training_level

        # self.cold_start_max_non_iid_level = self.cold_start_training_level / np.sum(self.cold_start_training_level)

    # def aggregate(self, results: List[Tuple[NDArrays, float]], m: int) -> NDArrays:
    #
    #     # pseudo_gradient: NDArrays = [
    #     #     x - y
    #     #     for x, y in zip(
    #     #         self.initial_parameters, fedavg_result
    #     #     )
    #     # ]
    #     #
    #     # fedavg_result = [
    #     #     x - self.server_learning_rate * y
    #     #     for x, y in zip(self.initial_parameters, pseudo_gradient)
    #     # ]
    #
    #
    #     """Compute weighted average."""
    #     # Calculate the total number of examples used during training
    #     num_examples_total = sum([num_examples for _, num_examples, cid in results])
    #     num_similarities_total = sum([self.clients_cosine_similarities[m][cid] for _, _, cid in results])
    #     total = num_examples_total + num_similarities_total
    #
    #     # Create a list of weights, each multiplied by the related number of examples
    #     weighted_weights = [
    #         [layer.detach().cpu().numpy() * (self.clients_cosine_similarities[m][cid]) for layer in weights.parameters()] for weights, num_examples, cid in results
    #     ]
    #
    #     # weighted_weights = [
    #     #     [layer.detach().cpu().numpy() * (num_examples / num_examples_total) for
    #     #      layer in weights.parameters()] for weights, num_examples, cid in results
    #     # ]
    #
    #     # Compute average weights of each layer
    #     # weights_prime: NDArrays = [
    #     #     reduce(np.add, layer_updates) / num_examples_total
    #     #     for layer_updates in zip(*weighted_weights)
    #     # ]
    #     print("similaridades total: ", num_similarities_total, self.clients_cosine_similarities[m][1])
    #     weights_prime: NDArrays = [
    #         reduce(np.add, layer_updates) / num_similarities_total
    #         for layer_updates in zip(*weighted_weights)
    #     ]
    #
    #     weights_prime = [Parameter(torch.Tensor(i.tolist())) for i in weights_prime]
    #
    #     self.current_model_unique_count_samples = [np.array([0.0 for i in range(self.num_classes[m])]) for m in range(self.M)]
    #
    #     for _, _, cid in results:
    #         self.current_model_unique_count_samples[m] += (np.array(self.client_class_count[m][cid]))
    #
    #     self.current_model_unique_count_samples[m] = np.array([self.current_model_unique_count_samples[m]])
    #     # print(self.current_model_unique_count_samples[m])
    #     # exit()
    #
    #
    #     # print(self.client_class_count[m][0])
    #     for i in range(self.num_clients):
    #         # self.client_class_count[m][i] = self.clients[i].train_class_count[m]
    #         total  = np.sum(self.client_class_count[m][i])
    #         normalized = np.array([self.client_class_count[m][i]])
    #         # print("unique: ", self.current_model_unique_count_samples[m], self.current_model_unique_count_samples[m].ndim)
    #         # print("normalized: ", list(normalized), cosine_similarity(self.current_model_unique_count_samples[m], normalized))
    #         self.clients_cosine_similarities_with_current_model[m][i] = cosine_similarity(self.current_model_unique_count_samples[m], normalized)[0][0]
    #
    #
    #     # exit()
    #     print("Similaridade atual: ")
    #     print(m, i, self.clients_cosine_similarities_with_current_model[m])
    #     return weights_prime

    def train_metrics(self, m, t):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        accs = []
        losses = []
        num_samples = []
        balanced_accs = []
        micro_fscores = []
        macro_fscores = []
        weighted_fscores = []
        alpha_list = []
        for i in range(len(self.clients)):
            c = self.clients[i]
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

        ids = [c.id for c in self.clients]

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

        return {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc, "Loss": loss,
                'Balanced accuracy': balanced_acc,
                'Micro f1-score': micro_fscore, 'Macro f1-score': macro_fscore, 'Weighted f1-score': weighted_fscore, "Alpha": alpha_list}




