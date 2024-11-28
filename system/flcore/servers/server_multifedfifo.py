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


class MultiFedCP(Server):
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

    def select_clients(self, t):
        try:

            g = torch.Generator()
            g.manual_seed(t)
            np.random.seed(t)
            random.seed(t)
            if t == 1:
                self.previous_selected_clients = super().select_clients(t)
                for m in range(self.M):
                    self.clients_rounds_since_last_training[m] += 1
                    for c_id in self.previous_selected_clients[m]:
                        self.clients_training_round_per_model[c_id][m].append(t)
                        self.clients_rounds_since_last_training[m][c_id] = 0

                    self.clients_rounds_since_last_training_probability[m] = self.clients_rounds_since_last_training[
                                                                                 m] / np.max(
                        self.clients_rounds_since_last_training[m])
                    self.clients_rounds_since_last_training_probability[m] = \
                    self.clients_rounds_since_last_training_probability[m] / np.sum(
                        self.clients_rounds_since_last_training_probability[m])


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

                clients_losses = []
                metric = "Loss"

                for client in self.clients:
                    cid = client.id
                    client_losses = []
                    improvements = []
                    for m in range(len(client.train_metrics_list_dict)):
                        rounds = self.clients_training_round_per_model[cid][m][-2:]
                        rounds = np.array(rounds) - 1
                        print("r tre", rounds)
                        metrics_m = client.train_metrics_list_dict[m]
                        local_acc = np.array(self.clients_train_metrics[client.id]["Accuracy"][m])
                        local_loss = np.array(self.clients_train_metrics[client.id]["Loss"][m])
                        global_acc = self.results_train_metrics[m]["Accuracy"]

                print("fla: ", np.array([self.clients_rounds_since_last_training_probability[m] for m in range(self.M)]).flatten())
                p = np.array([self.clients_rounds_since_last_training_probability[m] for m in range(self.M)]).flatten()
                if np.sum(p) == 0:
                    p = np.ones(p.shape)

                p = p / np.sum(p)

                selected_clients_m_1 = np.random.choice([i for i in range(int(self.num_clients * self.M))], self.current_num_join_clients, replace=False, p=p)
                selected_clients_m_2 = np.random.choice([i for i in range(int(self.num_clients * self.M))],
                                                      self.current_num_join_clients*self.M, replace=False, p=p)


                selected_clients = []
                selected_clients_dict = {m: [] for m in range(self.M)}
                for i in selected_clients_m_2:
                    client_id = i % self.num_clients
                    client_model = i // self.num_clients

                    if client_id not in selected_clients and len(selected_clients_dict[client_model]) < self.current_num_join_clients // self.M:
                        selected_clients.append(client_id)
                        selected_clients_dict[client_model].append(client_id)
                        if len(selected_clients) == self.current_num_join_clients:
                            break



                self.previous_selected_clients = []
                for m in range(self.M):
                    self.previous_selected_clients.append(selected_clients_dict[m])

                selected_clients_m = self.previous_selected_clients

                for m in range(self.M):
                    self.clients_rounds_since_last_training[m] += 1
                    for c_id in self.previous_selected_clients[m]:
                        self.clients_training_round_per_model[c_id][m].append(t)
                        self.clients_rounds_since_last_training[m][c_id] = 0

                    self.clients_rounds_since_last_training_probability[m] = self.clients_rounds_since_last_training[m] / np.max(self.clients_rounds_since_last_training[m])
                    self.clients_rounds_since_last_training_probability[m] = self.clients_rounds_since_last_training_probability[m] / np.sum(self.clients_rounds_since_last_training_probability[m])


                print("selecionado: ", selected_clients_m)
                return selected_clients_m

        except Exception as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print("""\n{}""".format(selected_clients_m))

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
                self.set_new_clients(clientAVG)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(m, t=t)

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




