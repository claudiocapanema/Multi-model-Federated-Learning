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
import os

import pandas as pd
from flcore.clients.clientfednome import clientFedNome
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


class MultiFedEfficiency(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedNome)

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
        self.minimum_training_clients_per_model = {0.1: 1, 0.2: 2, 0.3: 3, 0.5: 3}[self.join_ratio]
        self.minimum_training_clients_per_model_percentage = self.minimum_training_clients_per_model / self.num_join_clients
        self.use_cold_start_m = np.array([True for m in range(self.M)])
        self.cold_start_max_non_iid_level = 1
        self.cold_start_training_level = np.array([0] * self.M)
        self.minimum_training_level = 2 / self.num_clients
        self.models_semi_convergence_rounds_n_clients = {m: [] for m in range(self.M)}
        self.max_n_training_clients = 10
        self.m_clients_rounds_without_training = np.ones((self.M, len(self.clients)))
        # Semi convergence detection window of rounds
        self.tw = []
        for d in self.dataset:
            self.tw.append({"WISDM-W": args.tw, "WISDM-P": args.tw, "ImageNet": args.tw, "CIFAR10": args.tw, "ImageNet_v2": args.tw, "Gowalla": args.tw}[d])
        self.tw_range = [0.5, 0.1]
        self.models_semi_convergence_min_n_training_clients = {m: self.minimum_training_clients_per_model for m in range(self.M)}
        self.models_semi_convergence_min_n_training_clients_percentage = {m: self.minimum_training_clients_per_model/self.num_join_clients for m in
                                                               range(self.M)}
        self.models_semi_convergence_flag = [False] * self.M
        self.models_convergence_flag = [False] * self.M
        self.models_semi_convergence_count = [0] * self.M
        self.models_semi_convergence_training_probability = [0] * self.M
        self.training_clients_per_model = np.array([0] * self.M)
        self.training_clients_per_model_per_round = {m: [] for m in range(self.M)}
        self.training_clients_per_model_per_round_dict = {m: {i: [] for i in range(self.minimum_training_clients_per_model, self.num_join_clients + 1)} for m in range(self.M)}
        self.n_training_clients_efficiency = np.zeros((self.M, self.num_join_clients))
        self.rounds_since_last_semi_convergence = {m: 0 for m in range(self.M)}
        self.unique_count_samples = {m: np.array([0 for i in range(self.num_classes[m])]) for m in range(self.M)}
        self.similarity_matrix = np.zeros((self.M, self.num_clients))
        self.accuracy_gain_models = {m: [] for m in range(self.M)}
        self.stop_cpd = [False for m in range(self.M)]
        self.re_per_model = int(args.reduction)
        self.fraction_of_classes = np.zeros((self.M, self.num_clients))
        self.imbalance_level = np.zeros((self.M, self.num_clients))
        self.lim = []
        self.free_budget_distribution_factor = args.df
        # self.selected_clients_cosine = {m: {} for m in range(self.M)}

        print("Rodando com tw: ", self.tw)

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

    def random_selection(self, t):

        try:
            g = torch.Generator()
            g.manual_seed(t)
            np.random.seed(t)
            random.seed(t)

            budget = int(self.num_join_clients / self.M)
            cm = [budget] * self.M

            for m in range(self.M):
                if self.models_semi_convergence_count[m] > 0:
                    cm[m] = int(max(self.minimum_training_clients_per_model,
                                                      cm[m] - self.models_semi_convergence_count[m]))

            print("cm i: ", cm)

            if self.free_budget_distribution_factor > 0:
                free_budget = self.num_join_clients - np.sum(cm)
                k_nt = len(np.argwhere(self.need_for_training >= 0.5))
                free_budget_k = int(int(free_budget * self.free_budget_distribution_factor) / k_nt)
                rest = free_budget - free_budget_k * k_nt

                print("Free budget: ", free_budget, " k nt: ", k_nt, " Free budget k: ", free_budget_k, " resto: ", rest)

                for m in range(self.M):
                    if self.need_for_training[m] >= 0.5 and cm[m] == budget:
                        cm[m] = int(cm[m] + free_budget_k)
                        if rest > 0:
                            cm[m] += 1
                            rest -= 1
                            rest = max(rest, 0)

            selected_clients = list(np.random.choice(self.available_clients, self.num_join_clients, replace=False))
            selected_clients = [i.id for i in selected_clients]

            selected_clients_m = [None] * self.M

            print("a : ", selected_clients_m)
            print("random: ", selected_clients)
            print("cm: ", cm)
            i = 0
            reverse_list = [0, 1, 2]
            for m in reverse_list:
                j = i + cm[m]
                selected_clients_m[m] = selected_clients[i: j]
                i = j

            return selected_clients_m

        except Exception as e:
            print("Error random selection")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def weighted_selection(self, t):

        try:
            # if t == 1:
            #     return [np.array([1 / self.num_clients for i in range(self.num_clients)]) for j in range(self.M)]
            #
            # prob = [[0 for i in range(self.num_clients)] for j in range(self.M)]
            #
            # for m in range(self.M):
            #     clients = np.argsort(self.clients_training_count[m])
            #     p = np.array(self.clients_training_count[m]) / np.sum(self.clients_training_count[m])
            #     print("soma: ", p, self.clients_training_count[m])
            #     p = 1 - p # high probability for less training clients
            #     p = (-3 + np.power(2, 2 * p)) # Keeps high probability when it is very close to 1
            #     p[p<0] = 0 # Avoid negative probability
            #     prob[m] = p
            #     # clients = clients[0:c]
            #     # prob[m] = np.array([1 if i in clients else 0 for i in range(self.num_clients)])
            #
            # print("Uniform")
            # print(prob)
            #
            # return prob

            n_selected_clients_m = [self.num_join_clients / self.M] * self.M

            prob = np.ones((self.M, len(self.clients)))
            for m in range(self.M):
                n_selected_clients_m[m] = int(max(self.minimum_training_clients_per_model,
                                                  n_selected_clients_m[m] - self.models_semi_convergence_count[m]))
                # (1 - need for training) * uniform probability + (need for training) * weighted probability
                prob[m] = (1 - self.need_for_training[m]) * prob[m] / np.sum(prob[m]) + self.need_for_training[m] * np.array(self.m_clients_rounds_without_training[m]) / np.sum(self.m_clients_rounds_without_training[m])
                prob[m] = prob[m] / np.sum(prob[m])
            p = np.array([prob[m] for m in range(self.M)]).flatten()
            if np.sum(p) == 0:
                p = np.ones(p.shape)

            # print("quantidade de clientes para selecionar modelo m: ", n_selected_clients_m)

            p = p / np.sum(p)
            selected_clients_m_2 = np.random.choice([i for i in range(int(self.num_clients * self.M))],
                                                    self.current_num_join_clients, replace=False, p=p)

            selected_clients = []
            selected_clients_dict = {m: [] for m in range(self.M)}
            for i in selected_clients_m_2:
                client_id = i % self.num_clients
                client_model = i // self.num_clients
                # print("fora: ", client_model, client_id not in selected_clients,  len(selected_clients_dict[client_model]) < n_selected_clients_m[client_model], len(selected_clients_dict[client_model]), n_selected_clients_m[client_model])
                if client_id not in selected_clients and len(
                        selected_clients_dict[client_model]) < n_selected_clients_m[client_model]:
                    selected_clients.append(client_id)
                    selected_clients_dict[client_model].append(client_id)
                    # print("total por modelo: ", client_model, [len(selected_clients_dict[m]) for m in range(self.M)])
                    if len(selected_clients) == np.sum(n_selected_clients_m):
                        print("Atingiu o total: ", len(selected_clients), np.sum(n_selected_clients_m), len(selected_clients_m_2))
                        break

            # print("len selec 2: ", len(selected_clients), np.sum(n_selected_clients_m), len(selected_clients_m_2))
            final = [[] for m in range(self.M)]
            for m in range(self.M):
                final[m] = selected_clients_dict[m]

            return final

        except Exception as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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

            g = torch.Generator()
            g.manual_seed(t)
            np.random.seed(t)
            random.seed(t)
            # if t == 1:
            #     self.previous_selected_clients = super().select_clients(t)
            #     n_clients_m = {m: 0 for m in range(self.M)}
            #     for m in range(self.M):
            #         n_clients_m[m] = len(self.previous_selected_clients[m])
            #         self.training_clients_per_model_per_round[m].append(n_clients_m[m])
            #         for c_id in self.previous_selected_clients[m]:
            #             self.clients_training_round_per_model[c_id][m].append(t)
            #
            #         self.rounds_since_last_semi_convergence[m] += 1
            #
            #     return self.previous_selected_clients
            # else:
            #     if self.random_join_ratio:
            #         self.current_num_join_clients = \
            #         np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
            #     else:
            #         self.current_num_join_clients = self.num_join_clients
            #     np.random.seed(t)
            #     selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
            #     selected_clients_m = [[] for i in range(self.M)]
            #
            #     self.calculate_accuracy_gain_models()
            #
            # prob_cold_start_m, use_cold_start_m = self.cold_start_selection()
            #
            # print("Cold start")
            # print(prob_cold_start_m, use_cold_start_m)

            #
            #     clients_losses = []
            #
            #     for client in self.clients:
            #         cid = client.id
            #         client_losses = []
            #         improvements = []
            #         for m in range(len(client.train_metrics_list_dict)):
            #             rounds = self.clients_training_round_per_model[cid][m][-2:]
            #             rounds = np.array(rounds) - 1
            #             print("r tre", rounds)
            #             metrics_m = client.train_metrics_list_dict[m]
            #             local_acc = np.array(self.clients_train_metrics[client.id]["Accuracy"][m])
            #             local_loss = np.array(self.clients_train_metrics[client.id]["Loss"][m])
            #             global_acc = self.results_train_metrics[m]["Accuracy"]
            #
            #
            #
            #     print("Modelos clientes: ", selected_clients_m)

            """semi-convergence detection"""
            if t == 1:
                diff = self.tw_range[0] - self.tw_range[1]
                middle = self.tw_range[1] + diff//2
                for m in range(self.M):
                    # method 2
                    # r = diff * (1 - self.need_for_training[m])
                    # if self.need_for_training[m] >= 0.5:
                    #     # Makes it harder to reduce training intensity
                    #     lower = upper - t/2
                    #     upper = lower + diff * (1- self.need_for_training[m])
                    # else:
                    #     # Makes it easier to reduce training intensity
                    #     upper = self.tw_range[0]
                    #     lower = upper - diff * self.need_for_training[m]

                    # method 1 funciona bem
                    lower = self.tw_range[1]
                    upper = self.tw_range[0]
                    r = diff * (1 - self.need_for_training[m])
                    # Smaller training reduction interval for higher need for training
                    lower = max(middle - r/2, lower)
                    upper = min(middle + r/2, upper)

                    self.lim.append([upper, lower])
            flag = True
            print("limites: ", self.lim)
            # exit()
            loss_reduction = [0] * self.M
            for m in range(self.M):
                if not self.stop_cpd[m]:
                    self.rounds_since_last_semi_convergence[m] += 1

                    """Stop CPD"""
                    print("Modelo m: ", m)
                    print("tw: ", self.tw[m], self.results_test_metrics[m]["Loss"])
                    losses = self.results_test_metrics[m]["Loss"][-(self.tw[m]+1):]
                    losses = np.array([losses[i] - losses[i+1] for i in range(len(losses)-1)])
                    if len(losses) > 0:
                        loss_reduction[m] = losses[-1]
                    print("Modelo ", m, " losses: ", losses)
                    idxs = np.argwhere(losses < 0)
                    # lim = [[0.5, 0.25], [0.35, 0.15]]
                    upper = self.lim[m][0]
                    lower = self.lim[m][1]
                    print("Condição 1: ", len(idxs) <= int(self.tw[m] * self.lim[m][0]), "Condição 2: ",
                          len(idxs) >= int(self.tw[m] * lower))
                    print(len(idxs), self.tw[m], upper, lower, int(self.tw[m] * upper), int(self.tw[m] * lower))
                    if self.rounds_since_last_semi_convergence[m] >= 4:
                        if len(idxs) <= int(self.tw[m] * upper) and len(idxs) >= int(self.tw[m] * lower):
                            self.rounds_since_last_semi_convergence[m] = 0
                            print("a, remaining_clients_per_model, total_clientsb: ", self.training_clients_per_model_per_round[m])
                            self.models_semi_convergence_rounds_n_clients[m].append({'round': t - 2, 'n_training_clients':
                                self.training_clients_per_model_per_round[m][t - 2]})
                            # more clients are trained for the semi converged model
                            print("treinados na rodada passada: ", m, self.training_clients_per_model_per_round[m][t - 2])

                            if flag:
                                self.models_semi_convergence_flag[m] = True
                                self.models_semi_convergence_count[m] += 1
                                flag = False

                        elif len(idxs) > int(self.tw[m] * upper):
                            self.rounds_since_last_semi_convergence[m] += 1
                            self.models_semi_convergence_count[m] -= 1
                            self.models_semi_convergence_count[m] = max(0, self.models_semi_convergence_count[m])

            """Selection"""
            if t < self.round_new_clients:
                self.num_available_clients = int(self.num_clients * (1 - self.fraction_new_clients))
                self.available_clients = self.clients[:self.num_available_clients]
                self.num_join_clients = int(self.num_clients * self.join_ratio)
            else:
                self.num_available_clients = len(self.clients)
                self.available_clients = self.clients
                self.num_join_clients = int(self.num_clients * self.join_ratio)

            if self.random_join_ratio:
                self.current_num_join_clients = \
                np.random.choice(range(self.num_join_clients, self.num_available_clients + 1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_available_clients


            # if use_cold_start_m.all() == False:
            #
            #     selected_clients = list(
            #         np.random.choice(self.available_clients, self.current_num_join_clients, replace=False))
            #     selected_clients = [i.id for i in selected_clients]
            #
            #     n = len(selected_clients) // self.M
            #
            #     # selected_clients_m = np.array_split(selected_clients, self.M)
            #     n_selected_clients_m = [self.num_join_clients / self.M] * self.M
            #     for m in range(self.M):
            #         n_selected_clients_m[m] = max(self.minimum_training_clients_per_model, n_selected_clients_m[m] - self.models_semi_convergence_count[m])
            #
            #     n_selected_clients_m = np.array(n_selected_clients_m).astype(int)
            #
            #     selected_clients_m = []
            #     i = 0
            #     for m in range(self.M):
            #         j = i + n_selected_clients_m[m]
            #         # print("sec: ", m, " i ", i, " j ", j, "\n n sele: ", n_selected_clients_m, n_selected_clients_m[m], self.models_semi_convergence_count[m], self.num_join_clients)
            #         selected_clients_m.append(selected_clients[i: j])
            #         i = j
            #
            # else:
            #     n_selected_clients_m = [self.num_join_clients / self.M] * self.M
            #     remaining_clients_m = [self.num_join_clients / self.M for m in range(self.M)]
            #     selected_clients_m = [[] for m in range(self.M)]
            #     for m in range(self.M):
            #         n_selected_clients_m[m] = int(max(self.minimum_training_clients_per_model,
            #                                       n_selected_clients_m[m] - self.models_semi_convergence_count[m]))
            #         if use_cold_start_m[m]:
            #             print(np.argwhere(np.array(prob_cold_start_m[m] == 1).flatten()).flatten(), n_selected_clients_m[m])
            #             selected_clients_arg = list(np.argwhere(np.array(prob_cold_start_m[m] == 1).flatten()).flatten())[:n_selected_clients_m[m]]
            #             selected_clients_m[m] += selected_clients_arg
            #             remaining_clients_m[m] = n_selected_clients_m[m] - len(selected_clients_m[m])
            #             for m_i in range(self.M):
            #                 prob_cold_start_m[m_i][selected_clients_arg] = 0
            #
            #     print("pre: ", selected_clients_m)
            #     aux = []
            #     for m_i in range(self.M):
            #         aux += selected_clients_m[m_i]
            #     available_clients = set(i for i in range(len(self.clients))) - set(aux)
            #     print("disponiveis pos cold: ", available_clients)
            #
            #     for m in range(self.M):
            #         if remaining_clients_m[m] > 0:
            #             selected = np.random.choice(list(available_clients), remaining_clients_m[m], replace=False)
            #             available_clients -= set(selected)
            #             selected_clients_m[m] += list(selected)

            print("Semi convergências: ", self.models_semi_convergence_count)
            # selected_clients_m = self.weighted_selection(t)
            selected_clients_m = self.random_selection(t)

            selected_clients = list(np.random.choice(self.available_clients, self.num_join_clients, replace=False))
            # selected_clients = [i.id for i in selected_clients]
            #
            # selected_clients_m = []
            # a = 0
            # b = 0
            # for m in range(self.M):
            #     b = a + int(max(self.minimum_training_clients_per_model,
            #                                       self.num_join_clients//self.M - self.models_semi_convergence_count[m]))
            #     selected_clients_m.append(selected_clients[a:b])
            #     a = b

            print("Selecionados: ", t, sum([len(selected_clients_m[i]) for i in range(len(selected_clients_m))]), [len(i) for i in selected_clients_m], selected_clients_m)
            # print("Quantidade: ", n_clients_selected)
            # exit()

            if t > 2:
                for m in range(self.M):
                    n = len(self.previous_selected_clients[m])
                    self.training_clients_per_model_per_round_dict[m][n].append(loss_reduction[m])
                    recent_loss_reductions = self.training_clients_per_model_per_round_dict[m][n][-4:]
                    self.n_training_clients_efficiency[m][n-self.minimum_training_clients_per_model-1] = np.sum(recent_loss_reductions) / n

                normalized = []
                for m in range(self.M):
                    total = np.sum(self.n_training_clients_efficiency[m])
                    if total > 0:
                        normalized.append(self.n_training_clients_efficiency[m] / total)
                    else:
                        normalized.append([0] * len(self.n_training_clients_efficiency[m]))

                print("Efficiência de treinamento: ", normalized)

            self.previous_selected_clients = selected_clients_m
            for m in range(self.M):
                n = len(self.previous_selected_clients[m])
                self.training_clients_per_model_per_round[m].append(n)
                self.m_clients_rounds_without_training[m] += 1
                self.m_clients_rounds_without_training[m][np.array(selected_clients_m[m])] = 0

            selected_clients_m = [np.array(i) for i in selected_clients_m]

            aux = []
            for a in selected_clients_m:
                aux += list(a)
            if pd.Series(a).duplicated().any():
                print("Existem duplicadas")
                raise

            return selected_clients_m

        except Exception as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print("""{} \n{} \n{} \n{} \n{}""".format(len(self.available_clients), selected_clients_m, losses))

    def train(self):

        self._get_models_size()
        for m in range(self.M):
            for i in range(self.num_clients):
                self.client_class_count[m][i] = self.clients[i].train_class_count[m]
                print("no train: ", " cliente: ", i, " modelo: ", m, " train class count: ", self.clients[i].train_class_count[m])
                # non-iid degree
                self.fraction_of_classes[m][i] = self.clients[i].fraction_of_classes[m]
                self.imbalance_level[m][i] = self.clients[i].imbalance_level[m]

        # self.detect_non_iid_degree()

        # print("Non iid degree")
        # print("#########")
        # print("""M1: {}\nM2: {}""".format(self.non_iid_degree[0], self.non_iid_degree[1]))

        print(self.dataset)
        average_fraction_of_classes = 1 - np.mean(self.fraction_of_classes, axis=1)
        average_balance_level = np.mean(self.imbalance_level, axis=1)
        self.need_for_training = (average_fraction_of_classes + average_balance_level) / 2
        weighted_need_for_training = self.need_for_training / np.sum(self.need_for_training)

        print("Média fraction of classes: ", np.mean(self.fraction_of_classes, axis=1))
        print("Média imbalance level: ", np.mean(self.imbalance_level, axis=1))
        print("Need for training: ", self.need_for_training)
        print("Weighted need for training: ", weighted_need_for_training)


        min_need_for_training = np.min(self.need_for_training)


        # for i in range(len(self.dataset)):
        #     self.tw[i] = min(int((self.tw[i] * need_for_training[i]) / min_need_for_training), self.tw[i] * 2)
        #
        # self.tw = [10, 5]

        print("tw ajustado: ", self.tw)
        # exit()
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
                    self.clients[client_id].train(m, self.global_model[m], self.clients_cosine_similarities_with_current_model[m][client_id], t)
                    self.clients_training_count[m][client_id] += 1
                    self.current_training_class_count[m] += self.clients[client_id].train_class_count[m]



                self.current_training_class_count[m] = np.round(self.current_training_class_count[m] / np.sum(self.current_training_class_count[m]), 2)
                # print("current training class count ", self.current_training_class_count[m])

                # print("######")
                # print("Training count: ", self.clients_training_count)

            self.receive_models()
            if self.dlg_eval and t%self.dlg_gap == 0:
                self.call_dlg(t)
            self.aggregate_parameters()

            for m in range(len(self.selected_clients)):
                if t%self.eval_gap == 0:
                    print(f"\n-------------Round number: {t}-------------")
                    print("\nEvaluate global model for ", self.dataset[m])
                    self.evaluate(m, t=t)

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

    # def train_metrics(self, m, t):
    #
    #     accs = []
    #     losses = []
    #     num_samples = []
    #     balanced_accs = []
    #     micro_fscores = []
    #     macro_fscores = []
    #     weighted_fscores = []
    #     for i in range(len(self.clients)):
    #         c = self.clients[i]
    #         train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore, alpha = c.train_metrics(m, t)
    #         accs.append(train_acc * train_num)
    #         num_samples.append(train_num)
    #         balanced_accs.append(train_balanced_acc * train_num)
    #         micro_fscores.append(train_micro_fscore * train_num)
    #         macro_fscores.append(train_macro_fscore * train_num)
    #         weighted_fscores.append(train_weighted_fscore * train_num)
    #         self.clients_train_metrics[c.id]["Samples"][m].append(num_samples)
    #         self.clients_train_metrics[c.id]["Accuracy"][m].append(train_acc)
    #         self.clients_train_metrics[c.id]["Loss"][m].append(train_loss)
    #         self.clients_train_metrics[c.id]["Balanced accuracy"][m].append(train_balanced_acc)
    #         self.clients_train_metrics[c.id]["Micro f1-score"][m].append(train_micro_fscore)
    #         self.clients_train_metrics[c.id]["Macro f1-score"][m].append(train_macro_fscore)
    #         self.clients_train_metrics[c.id]["Weighted f1-score"][m].append(train_weighted_fscore)
    #
    #     ids = [c.id for c in self.clients]
    #
    #     decimals = 5
    #     print("amostras: ", num_samples, accs, self.selected_clients)
    #     if len(num_samples) == 0:
    #         return None
    #     acc = round(sum(accs) / sum(num_samples), decimals)
    #     loss = round(sum(losses) / sum(num_samples), decimals)
    #     balanced_acc = round(sum(balanced_accs) / sum(num_samples), decimals)
    #     micro_fscore = round(sum(micro_fscores) / sum(num_samples), decimals)
    #     macro_fscore = round(sum(macro_fscores) / sum(num_samples), decimals)
    #     weighted_fscore = round(sum(weighted_fscores) / sum(num_samples), decimals)
    #
    #     return {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc, "Loss": loss,
    #             'Balanced accuracy': balanced_acc,
    #             'Micro f1-score': micro_fscore, 'Macro f1-score': macro_fscore, 'Weighted f1-score': weighted_fscore, "Alpha": alpha}

    def get_results(self, m, train_test, mode):

        algo = self.dataset[m] + "_" + self.algorithm
        cd = bool(self.args.concept_drift)
        if cd:
            result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(cd,
                                                                                                                                                                                                    self.fraction_new_clients,
                                                                                                                                                                                                    self.round_new_clients,
                                                                                                                                                                                                    self.num_clients,
                                                                                                                                                                                                    self.alpha,
                                                                                                                                                                                                    self.alpha_end,
                                                                                                                                                                                                    self.dataset,
                                                                                                                                                                                                    self.rounds_concept_drift[
                                                                                                                            0],
                                                                                                                                                                                                    self.rounds_concept_drift[
                                                                                                                            1],
                                                                                                                                                                                                    self.models_names,
                                                                                                                                                                                                    self.args.join_ratio,
                                                                                                                                                                                                    self.args.number_of_rounds,
                                                                                                                                                                                                    self.local_epochs,
                                                                                                                                                                                                    train_test)
        elif len(self.alpha) == 1:
            result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                cd,
                self.fraction_new_clients,
                self.round_new_clients,
                self.num_clients,
                [self.alpha[0]],
                self.alpha[
                    0],
                [self.dataset[0]],
                0,
                0,
                [self.models_names[0]],
                self.args.join_ratio,
                self.args.number_of_rounds,
                self.local_epochs,
                train_test)
        else:

            result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                cd,
                self.fraction_new_clients,
                self.round_new_clients,
                self.num_clients,
                self.alpha,
                self.alpha,
                self.dataset,
                0,
                0,
                self.models_names,
                self.args.join_ratio,
                self.args.number_of_rounds,
                self.local_epochs,
                train_test)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + train_test + "_" + str(self.times)
            file_path = result_path + "{}_tw_{}_df_{}.csv".format(algo, self.tw[m], self.free_budget_distribution_factor)

        if train_test == 'test':

            if mode == '':
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

            else:
                header = self.test_metrics_names
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
                file_path = file_path.replace(".csv", "_weighted.csv")

        else:
            if mode == '':
                header = self.train_metrics_names
                list_of_metrics = []
                for me in self.results_train_metrics[m]:
                    print(me, len(self.results_train_metrics[m][me]))
                    length = len(self.results_train_metrics[m][me])
                    list_of_metrics.append(self.results_train_metrics[m][me])

                data = []
                print("tamanho: ", length, list_of_metrics)
                for i in range(length):
                    row = []
                    for j in range(len(list_of_metrics)):
                        if len(list_of_metrics[j]) > 0:
                            row.append(list_of_metrics[j][i])
                        else:
                            row.append(0)

                    data.append(row)

            else:

                header = self.train_metrics_names
                list_of_metrics = []
                for me in self.results_train_metrics_w[m]:
                    print(me, len(self.results_train_metrics_w[m][me]))
                    length = len(self.results_train_metrics_w[m][me])
                    list_of_metrics.append(self.results_train_metrics_w[m][me])

                data = []
                for i in range(length):
                    row = []
                    for j in range(len(list_of_metrics)):
                        if len(list_of_metrics[j]) > 0:
                            row.append(list_of_metrics[j][i])
                        else:
                            row.append(0)

                    data.append(row)

                file_path = file_path.replace(".csv", "_weighted.csv")

        print("File path: " + file_path)

        return file_path, header, data




