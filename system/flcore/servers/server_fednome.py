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

import time
import sys
import math
import numpy as np
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


class FedNome(Server):
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
        self.minimum_training_clients_per_model = {0.1: 1, 0.2: 2, 0.3: 3}[self.join_ratio]
        self.minimum_training_clients_per_model_percentage = self.minimum_training_clients_per_model / self.num_join_clients
        self.cold_start_max_non_iid_level = 1
        self.cold_start_training_level = np.array([0] * self.M)
        self.minimum_training_level = 2 / self.num_clients
        self.training_clients_per_model = np.array([0] * self.M)
        self.unique_count_samples = {m: np.array([0 for i in range(self.num_classes[m])]) for m in range(self.M)}
        self.similarity_matrix = np.zeros((self.M, self.num_clients))

    def cold_start_selection(self):

        prob = [np.array([0 for i in range(self.num_clients)]) for j in range(self.M)]
        use_cold_start = np.array([False] * self.M)
        for m in range(self.M):
            for i in range(self.num_clients):
                if self.clients_training_count[m][i] == 0 and self.non_iid_degree[m]['unique local classes'] <= 100 * self.cold_start_max_non_iid_level:
                    prob[m][i] = 1
                    use_cold_start[m] = True

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
        print(clients_m_p)
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

            np.random.seed(t)
            if t == 1:
                return super().select_clients(t)
            else:
                if self.random_join_ratio:
                    self.current_num_join_clients = \
                    np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
                else:
                    self.current_num_join_clients = self.num_join_clients
                np.random.seed(t)
                selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
                selected_clients_m = [[] for i in range(self.M)]

                t_size = 3
                m_global_diff_list = []
                for m in range(self.M):
                    global_acc = self.results_test_metrics[m]["Accuracy"]
                    # Train more the models that have returned higher accuracy gain / # training clients
                    if len(global_acc) >= 2:
                        m_global_diff = (global_acc[-1] - global_acc[-min(2, t_size):][0]) / self.results_train_metrics[m]['# training clients'][-1]
                    else:
                        m_global_diff = global_acc[0] / self.results_train_metrics[m]['# training clients'][-1]
                    m_global_diff_list.append(m_global_diff)

                m_global_diff_list = np.array(m_global_diff_list)
                m_global_diff_list = m_global_diff_list / np.sum(m_global_diff_list)
                m_global_diff_list = np.where(m_global_diff_list < self.minimum_training_clients_per_model_percentage, self.minimum_training_clients_per_model_percentage, m_global_diff_list)
                m_global_diff_list = m_global_diff_list / np.sum(m_global_diff_list)

                clients_losses = []
                metric = "Accuracy"
                for client in self.clients:
                    cid = client.id
                    client_losses = []
                    improvements = []
                    samples_m = []
                    for m in range(len(client.test_metrics_list_dict)):
                        metrics_m = client.test_metrics_list_dict[m]
                        samples_m.append(metrics_m['Samples'])
                        local_acc = self.clients_train_metrics[client.id][metric][m]
                        global_acc = self.results_train_metrics[m][metric]

                        if len(local_acc) >= 2 and len(self.clients_training_round_per_model[cid][m]) >= 2:
                            print("aaa: ", len(global_acc), self.clients_training_round_per_model[cid][m])
                            tr = self.clients_training_round_per_model[cid][m]
                            # if q == 0:
                            local_diff = (local_acc[-1] - local_acc[-2])
                            # else:
                            #     local_diff = (local_acc[-1] - local_acc[-2]) / q
                            if local_diff > 0:
                                improvement = local_diff
                                improvements.append(improvement)
                            else:
                                local_diff = 0.1 # 0.1 ou 0.01
                            if local_acc[-1] == 0:
                                relative_diff = 0.001
                            else:
                                relative_diff = local_diff / local_acc[-1]
                            local_diff = relative_diff * relative_diff + local_diff
                        elif len(local_acc) == 1:
                            local_diff = local_acc[-1]
                        else:
                            local_diff = 1

                        client_losses.append(local_diff * metrics_m['Samples'])
                    client_losses = np.array(client_losses) * m_global_diff_list
                    clients_losses.append((client_losses / np.sum(samples_m)))
                    client_losses = (np.power(client_losses, self.fairness_weight - 1)) / np.sum(client_losses)

                    client_losses = client_losses / np.sum(client_losses)
                    print("probal: ", client_losses)
                    # m = np.random.choice([i for i in range(self.M)], p=client_losses)
                    # selected_clients_m[m].append(client.id)

                clients_losses = np.array(clients_losses)
                print("fo: ", clients_losses.shape)

                prob_cold_start_m, use_cold_start_m = self.cold_start_selection()
                total_clients = 0
                remaining_clients = np.array([i for i in range(len(self.clients))])
                m_acc_gain = []
                for m in range(self.M):
                    global_acc = self.results_train_metrics[m][metric]
                    if len(global_acc) >= 2:
                        acc_gain = max(global_acc[-1] - global_acc[-2], 0)
                        m_acc_gain.append(acc_gain)
                    elif len(global_acc) == 1:
                        m_acc_gain.append(global_acc[0])
                    else:
                        # m_acc_gain.append(self.training_clients_per_model[m])
                        print("errado")
                        exit()
                # remaining_clients_per_model = self.training_clients_per_model
                m_acc_gain = np.array(m_acc_gain)
                print("ga: ", m_acc_gain)
                if np.sum(m_acc_gain) > 0:
                    remaining_clients_per_model = m_acc_gain / np.sum(m_acc_gain)
                else:
                    remaining_clients_per_model = np.array([1/self.M for i in range(self.M)])
                print("interme: ", remaining_clients_per_model)
                remaining_clients_per_model[np.where(remaining_clients_per_model < self.minimum_training_clients_per_model_percentage)] = self.minimum_training_clients_per_model_percentage
                remaining_clients_per_model = np.array(remaining_clients_per_model * self.num_join_clients, dtype=int)
                print("interme 2", remaining_clients_per_model)
                if np.sum(remaining_clients_per_model) < self.num_join_clients:
                    diff = self.num_join_clients - np.sum(remaining_clients_per_model)
                    i = 0
                    while np.sum(remaining_clients_per_model) < self.num_join_clients and diff > 0:
                        if remaining_clients_per_model[i] != 3:
                            remaining_clients_per_model[i] += 1
                            diff -= 1
                        i += 1
                        if i >= self.M:
                            i = 0
                print("remaining: ", remaining_clients_per_model)
                print("join clients: ", self.num_join_clients)
                sc = []
                available_clients = {i: clients_losses[i] for i in range(self.num_clients)}
                l = 0
                iteracoes = 0
                while total_clients < self.num_join_clients:
                    print("inicio disponiveis: ", len(available_clients))
                    for m in range(self.M):
                        print("m: ", m)
                        iteracoes += 1
                        print("Inicio iteracao: ", iteracoes)
                        if total_clients < self.num_join_clients and remaining_clients_per_model[m] > 0:
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
                            if l > 20:
                                exit()
                            l = l + 1

                        print("""Fim iteracao {} \nclientes totais {} clientes restantes {} \nclientes disponíveis {}""".format(iteracoes, total_clients, remaining_clients_per_model, available_clients))

                print("Modelos clientes: ", selected_clients_m)


                print("Selecionados: ", t, sum([len(i) for i in selected_clients_m]), selected_clients_m)
                # print("Quantidade: ", n_clients_selected)
                # exit()

                return selected_clients_m

        except Exception as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print("""{} \n{} \n{} \n{} \n{}""".format(len(available_clients), selected_clients_m, clients_losses, remaining_clients_per_model, total_clients))

    def train(self):

        for m in range(self.M):
            for i in range(self.num_clients):
                self.client_class_count[m][i] = self.clients[i].train_class_count[m]
                print("no train: ", " cliente: ", i, " modelo: ", m, " train class count: ", self.clients[i].train_class_count[m])

        self.detect_non_iid_degree()

        print("Non iid degree")
        print("#########")
        print("""M1: {}\nM2: {}""".format(self.non_iid_degree[0], self.non_iid_degree[1]))

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
        self.data_quality_selection()
        print(self.clients_cosine_similarities)
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

    def aggregate(self, results: List[Tuple[NDArrays, float]], m: int) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples, cid in results])
        num_similarities_total = sum([self.clients_cosine_similarities[m][cid] for _, _, cid in results])
        total = num_examples_total + num_similarities_total
        # print("exemplo: ", self.clients_cosine_similarities[m][3] / num_similarities_total)
        # print("exemplo: ", self.clients_cosine_similarities[m][4] / num_similarities_total)
        # print("exemplo: ", self.clients_cosine_similarities[m][5] / num_similarities_total)
        # print("exemplo: ", self.clients_cosine_similarities[m][6] / num_similarities_total)
        #
        # exit()


        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer.detach().cpu().numpy() * (self.clients_cosine_similarities[m][cid]) for layer in weights.parameters()] for weights, num_examples, cid in results
        ]

        # weighted_weights = [
        #     [layer.detach().cpu().numpy() * (num_examples / num_examples_total) for
        #      layer in weights.parameters()] for weights, num_examples, cid in results
        # ]

        # Compute average weights of each layer
        # weights_prime: NDArrays = [
        #     reduce(np.add, layer_updates) / num_examples_total
        #     for layer_updates in zip(*weighted_weights)
        # ]
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_similarities_total
            for layer_updates in zip(*weighted_weights)
        ]

        weights_prime = [Parameter(torch.Tensor(i.tolist())) for i in weights_prime]

        self.current_model_unique_count_samples = [np.array([0.0 for i in range(self.num_classes[m])]) for m in range(self.M)]

        for _, _, cid in results:
            self.current_model_unique_count_samples[m] += (np.array(self.client_class_count[m][cid]))

        self.current_model_unique_count_samples[m] = np.array([self.current_model_unique_count_samples[m]])
        # print(self.current_model_unique_count_samples[m])
        # exit()


        # print(self.client_class_count[m][0])
        for i in range(self.num_clients):
            # self.client_class_count[m][i] = self.clients[i].train_class_count[m]
            total  = np.sum(self.client_class_count[m][i])
            normalized = np.array([self.client_class_count[m][i]])
            # print("unique: ", self.current_model_unique_count_samples[m], self.current_model_unique_count_samples[m].ndim)
            # print("normalized: ", list(normalized), cosine_similarity(self.current_model_unique_count_samples[m], normalized))
            self.clients_cosine_similarities_with_current_model[m][i] = cosine_similarity(self.current_model_unique_count_samples[m], normalized)[0][0]


        # exit()
        print("Similaridade atual: ", m, i, self.clients_cosine_similarities_with_current_model[m])
        return weights_prime





