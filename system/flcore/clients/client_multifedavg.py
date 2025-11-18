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
import sys
import copy
import os
import torch
import numpy as np
from .utils.models_utils import load_model, get_weights, load_data, set_weights, test, train

def label_shift_config(ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds, seed=0):
    try:
        np.random.seed(seed)
        if len(experiment_id) > 0:
            if experiment_id == "label_shift#1":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3), int(n_rounds * 0.6)], [int(n_rounds * 0.3), int(n_rounds * 0.6)]]
                new_alphas = [[10.0, 0.1], [10.0, 0.1]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#2":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3), int(n_rounds * 0.6)], [int(n_rounds * 0.3), int(n_rounds * 0.6)]]
                new_alphas = [[0.1, 10.0],[0.1, 10.0]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#3_sudden":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[10.0], [10.0], [10.0]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#3_gradual":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.5) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.7) + client_id // gradual_rounds]]
                new_alphas = [[10.0], [10.0], [10.0]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#3_recurrent":
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                           [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                new_alphas = [[10.0, 0.1], [10.0, 0.1], [10.0, 0.1]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#4_sudden":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[0.1], [0.1], [0.1]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#4_gradual":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.5) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.7) + client_id // gradual_rounds]]
                new_alphas = [[0.1], [0.1], [0.1]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#4_recurrent":
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                           [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                new_alphas = [[0.1, 10.0], [0.1, 10.0], [0.1, 10.0]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#5_recurrent":
                ME_concept_drift_rounds = [[int(n_rounds * 0.1) + client_id // gradual_rounds,
                                            int(n_rounds * 0.4) + client_id // gradual_rounds,
                                            int(n_rounds * 0.7) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.2) + client_id // gradual_rounds,
                                            int(n_rounds * 0.5) + client_id // gradual_rounds,
                                            int(n_rounds * 0.8) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.3) + client_id // gradual_rounds,
                                            int(n_rounds * 0.6) + client_id // gradual_rounds,
                                            int(n_rounds * 0.9) + client_id // gradual_rounds]]
                new_alphas = [[1.0, 10.0, 0.1], [1.0, 10.0, 0.1], [1.0, 10.0, 0.1]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "label_shift#6":
                ME_concept_drift_rounds = [[int(n_rounds * 0.1), int(n_rounds * 0.4), int(n_rounds * 0.7)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.3), int(n_rounds * 0.6), int(n_rounds * 0.9)]]
                new_alphas = [[0.1, 10.0, 0.1], [0.1, 10.0, 0.1], [0.1, 10.0, 0.1]]
                type_ = "label_shift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            else:
                config = {}



        else:
            config = {}
        # else:
        #     config = {}
        return config

    except Exception as e:
        print("label_shift_config error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def global_concept_drift_config(ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds, seed=0):
    try:
        np.random.seed(seed)
        if len(experiment_id) > 0:
            type_ = "no_drift"
            if experiment_id == "concept_drift#1_sudden":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[0.1], [0.1], [0.1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "concept_drift#1_gradual":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.5) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.7) + client_id // gradual_rounds]]
                new_alphas = [[0.1], [0.1], [0.1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "concept_drift#1_recurrent":
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                           [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                new_alphas = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}

            if experiment_id == "concept_drift#2_sudden":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[10.0], [10.0], [10.0]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "concept_drift#2_recurrent":
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                           [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                new_alphas = [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif experiment_id == "concept_drift#2_gradual":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.5) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.7) + client_id // gradual_rounds]]
                new_alphas = [[10.0], [10.0], [10.0]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif "concept_drift#3" in experiment_id:
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[10.0, 1.0, 0.1], [0.1, 1.0, 10.0]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif "concept_drift#4" in experiment_id:
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[0.1, 1.0, 10.0], [10.0, 1.0, 0.1]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif "concept_drift#6" in experiment_id:
                # Melhor
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[0.1, 1.0, 10.0], [0.1, 1.0, 10.0]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif "concept_drift#7" in experiment_id:
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[10.0, 1.0, 0.1], [10.0, 1.0, 0.1]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif "concept_drift#8" in experiment_id:
                # CP real
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif "concept_drift#9" in experiment_id:
                # CP real
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            elif "concept_drift#10" in experiment_id:
                # CP real
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "type": type_} for me in range(ME)}
            else:
                config = {}



        else:
            config = {}
        # else:
        #     config = {}
        return config

    except Exception as e:
        print("global_concept_drift_config error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def get_data_shift_config(ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds):

    try:

        if "label_shift" in experiment_id:
            return label_shift_config(ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds)
        elif "concept_drift" in experiment_id:
            return global_concept_drift_config(ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds)
        else:
            return {}

    except Exception as e:
        print("get_data_shift_config error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class MultiFedAvgClient:
    def __init__(self, args, id, model):
        try:
            g = torch.Generator()
            g.manual_seed(id)
            random.seed(id)
            np.random.seed(id)
            torch.manual_seed(id)
            self.args = args
            self.batch_size = []
            for dataset in args.dataset:
                self.batch_size.append({"CIFAR10": 32, "WISDM-W": 16, "ImageNet10": 10, "Gowalla": 64, "wikitext": 256}[dataset])
            self.lr_dict = {'EMNIST':0.01,
                            'MNIST': 0.01,
                            'CIFAR10': 0.01,
                            'GTSRB': 0.01,
                            'WISDM-W': 0.001,
                            'WISDM-P': 0.001,
                            'ImageNet100': 0.01,
                            'ImageNet': 0.1,
                            'ImageNet10': 0.01,
                            "ImageNet_v2": 0.01,
                            "Gowalla": 0.001,
                            "wikitext": 0.001}
            self.model = model
            self.alpha = [float(i) for i in args.alpha]
            self.initial_alpha = self.alpha
            self.total_clients  = args.total_clients
            self.ME = len(self.model)
            self.number_of_rounds = args.number_of_rounds
            print("Preparing data...")
            print("""args do cliente: {} {}""".format(self.args.client_id, self.alpha))
            self.client_id = id
            self.trainloader = [None] * self.ME
            self.recent_trainloader = [None] * self.ME
            self.valloader = [None] * self.ME
            self.optimizer = [None] * self.ME
            self.index = 0
            self.local_epochs = self.args.local_epochs
            self.lr = self.args.learning_rate
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.lt = [0] * self.ME
            print("ler model size")
            self.models_size = self._get_models_size()
            self.n_classes = [
                {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
                 "ImageNet10": 10, "ImageNet_v2": 15, "Gowalla": 7, "wikitext": 30}[dataset] for dataset in
                self.args.dataset]
            self.loss_ME = [10] * self.ME
            # Concept drift parameters
            self.experiment_id = self.args.experiment_id
            self.gradual_rounds = 5
            self.data_shift_config = get_data_shift_config(self.ME, self.number_of_rounds, self.alpha, self.experiment_id, self.client_id, gradual_rounds=self.total_clients // self.gradual_rounds)
            self.concept_drift_window = [0] * self.ME
            self.data_shift_train_data = False

            self.concept_drift_config = {}
            print(f"concept drift config {self.concept_drift_config} concept drift id {self.experiment_id}")

            for me in range(self.ME):
                self.trainloader[me], self.valloader[me] = load_data(
                    dataset_name=self.args.dataset[me],
                    alpha=self.alpha[me],
                    data_sampling_percentage=self.args.data_percentage,
                    partition_id=self.client_id,
                    num_partitions=self.args.total_clients + 1,
                    batch_size=self.batch_size[me],
                )
                self.recent_trainloader[me] = copy.deepcopy(self.trainloader[me])
                self.optimizer[me] = self._get_optimizer(dataset_name=self.args.dataset[me], me=me)
                print("""leu dados cid: {} dataset: {} size:  {}""".format(self.client_id, self.args.dataset[me],
                                                                                 len(self.trainloader[me].dataset)))

                classes = []
                print("modelo: ", me)
                for batch in self.trainloader[me]:
                    labels = batch["label"]
                    classes += labels.numpy().tolist()
                # print("oi ", classes)
                # print("classes : ", np.unique(classes, return_counts=True))
            self.p_ME, self.fc_ME, self.il_ME = self._get_datasets_metrics(self.trainloader, self.ME,
                                                                           self.client_id,
                                                                           self.n_classes)
        except Exception as e:
            print("__init__ client error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def set_parameters(self, m, model):
        for new_param, old_param in zip(model.parameters(), self.model[m].parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def fit(self, me, t, global_model):
        """Train the model with data of this client."""
        try:
            self.lt[me] = t
            g = torch.Generator()
            g.manual_seed(t)
            random.seed(t)
            np.random.seed(t)
            torch.manual_seed(t)
            # self.trainloader[me] = self.recent_trainloader[me]
            set_weights(self.model[me], global_model)

            # Update alpha to simulate data shift
            self.update_local_train_data(t, me)

            self.optimizer[me] = self._get_optimizer(dataset_name=self.args.dataset[me], me=me)
            results = train(
                self.model[me],
                self.trainloader[me],
                self.valloader[me],
                self.optimizer[me],
                self.local_epochs,
                self.lr,
                self.device,
                self.client_id,
                t,
                self.args.dataset[me],
                self.n_classes[me],
                self.concept_drift_window[me]
            )
            results["me"] = me
            results["client_id"] = self.client_id
            results["Model size"] = self.models_size[me]
            results["alpha"] = self.alpha[me]
            self.loss_ME[me] = results["train_loss"]
            return get_weights(self.model[me]), len(self.trainloader[me].dataset), results
        except Exception as e:
            print("fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, me, t, global_model):
        """Evaluate the model on the data this client has."""
        try:
            g = torch.Generator()
            g.manual_seed(t)
            random.seed(t)
            np.random.seed(t)
            torch.manual_seed(t)
            tuple_me = {}
            nt = t - self.lt[me]
            self.update_local_test_data(t, me)
            set_weights(self.model[me], global_model)
            loss, metrics = test(self.model[me], self.valloader[me], self.device, self.client_id, t,
                                 self.args.dataset[me], self.n_classes[me], self.concept_drift_window[me])
            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            metrics["Alpha"] = self.alpha[me]
            tuple_me = (loss, len(self.valloader[me].dataset), metrics)
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def update_local_train_data(self, t, me):

        try:
            alpha_me = self._get_current_alpha(t, me)
            if self.data_shift_config != {}:
                if self.alpha[me] != alpha_me or (
                        t in self.data_shift_config[me]["data_shift_rounds"] and self.data_shift_config[me][
                    "type"] in ["label_shift"]):
                    self.alpha[me] = alpha_me
                    index = 0
                    self.recent_trainloader[me], self.valloader[me] = load_data(
                        dataset_name=self.args.dataset[me],
                        alpha=self.alpha[me],
                        data_sampling_percentage=self.args.data_percentage,
                        partition_id=int((self.args.client_id + index) % self.args.total_clients),
                        num_partitions=self.args.total_clients + 1,
                        batch_size=self.args.batch_size,
                    )
                    self.trainloader[me] = self.recent_trainloader[me]
                    p_ME, fc_ME, il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                                                                    self.n_classes, self.concept_drift_window)
                    self.p_ME, self.fc_ME, self.il_ME = p_ME, fc_ME, il_ME
                elif t in self.data_shift_config[me]["data_shift_rounds"] and self.data_shift_config[me]["type"] in [
                    "concept_drift"]:
                    # Recurrent
                    if "recurrent" in self.experiment_id:
                        if self.concept_drift_window[me] == 0:
                            self.concept_drift_window[me] = 1
                        elif self.concept_drift_window[me] == 1:
                            self.concept_drift_window[me] = 0
                    else:
                        self.concept_drift_window[me] += 1
                    p_ME, fc_ME, il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                                                                    self.n_classes, self.concept_drift_window)
                    self.p_ME, self.fc_ME, self.il_ME = p_ME, fc_ME, il_ME

        except Exception as e:
            print(f"update_local_train_data error {self.data_shift_config}")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def update_local_test_data(self, t, me):

        try:
            if self.data_shift_config != {}:
                alpha_me = self._get_current_alpha(t, me)
                if self.alpha[me] != alpha_me or (t in self.data_shift_config[me][
                    "data_shift_rounds"] and self.data_shift_config[me]["type"] in ["label_shift"]):
                    self.alpha[me] = alpha_me
                    index = 0
                    self.recent_trainloader[me], self.valloader[me] = load_data(
                        dataset_name=self.args.dataset[me],
                        alpha=self.alpha[me],
                        data_sampling_percentage=self.args.data_percentage,
                        partition_id=int((self.args.client_id + index) % self.args.total_clients),
                        num_partitions=self.args.total_clients + 1,
                        batch_size=self.args.batch_size,
                    )
                    p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
                elif t in self.data_shift_config[me][
                    "data_shift_rounds"] and self.data_shift_config[me]["type"] in ["concept_drift"] and t - self.lt[
                    me] > 0:
                    # Recurrent
                    if "recurrent" in self.experiment_id:
                        if self.concept_drift_window[me] == 0:
                            self.concept_drift_window[me] = 1
                        elif self.concept_drift_window[me] == 1:
                            self.concept_drift_window[me] = 0
                    else:
                        self.concept_drift_window[me] += 1
                    p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
                else:
                    p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
            else:
                p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME

            return p_ME, fc_ME, il_ME

        except Exception as e:
            print(f"update_local_train_data error {self.data_shift_config}")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_current_alpha(self, server_round, me):

        try:
            if self.data_shift_config == {}:
                return self.alpha[me]
            else:
                config = self.data_shift_config[me]
                alpha = None

                for i, round_ in enumerate(config["data_shift_rounds"]):
                    if server_round >= round_:
                        alpha = config["new_alphas"][i]

                if alpha is None:
                    alpha = self.alpha[me]

                return alpha
        except Exception as e:
            print(f"_get_current_alpha error {self.data_shift_config}")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_models_size(self):
        try:
            models_size = []
            for me in range(self.ME):
                parameters = [i.detach().cpu().numpy() for i in self.model[me].parameters()]
                size = 0
                for i in range(len(parameters)):
                    size += parameters[i].nbytes
                models_size.append(int(size))

            return models_size
        except Exception as e:
            print("_get_models_size error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_optimizer(self, dataset_name, me):
        try:
            return {
                    'EMNIST': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'MNIST': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'CIFAR10': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'GTSRB': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'WISDM-W': torch.optim.RMSprop(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'WISDM-P': torch.optim.RMSprop(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'ImageNet100': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'ImageNet': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    'ImageNet10': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    "ImageNet_v2": torch.optim.Adam(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    "Gowalla": torch.optim.RMSprop(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    "wikitext": torch.optim.RMSprop(self.model[me].parameters(), self.lr_dict[dataset_name])}[dataset_name]
        except Exception as e:
            print("_get_optimizer error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_datasets_metrics(self, trainloader, ME, client_id, n_classes, concept_drift_window=None):

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



        



