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


class MultiFedAvgClient:
    def __init__(self, args, id, model):
        g = torch.Generator()
        g.manual_seed(id)
        random.seed(id)
        np.random.seed(id)
        torch.manual_seed(id)
        self.args = args
        self.batch_size = []
        for dataset in args.dataset:
            self.batch_size.append({"WISDM-W": 16, "ImageNet10": 10, "Gowalla": 10}[dataset])
        self.model = model
        self.alpha = [float(i) for i in args.alpha]
        self.initial_alpha = self.alpha
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
             "ImageNet10": 10, "ImageNet_v2": 15, "Gowalla": 7}[dataset] for dataset in
            self.args.dataset]
        # Concept drift parameters
        self.experiment_id = self.args.experiment_id
        self.concept_drift_window = [0] * self.ME

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
            for batch in self.trainloader[me]:
                labels = batch["label"]
                classes += labels.numpy().tolist()
            print("oi ", classes)
            print("classes : ", np.unique(classes, return_counts=True))

            if me == 2:
                for batch in self.trainloader[me]:
                    pass

        #         print("x ", batch["sequence"])
        #
        # exit()

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
            self.trainloader[me] = self.recent_trainloader[me]
            set_weights(self.model[me], global_model)
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
                    'EMNIST': torch.optim.SGD(self.model[me].parameters(), lr=0.01, momentum=0.9),
                    'MNIST': torch.optim.SGD(self.model[me].parameters(), lr=0.01, momentum=0.9),
                    'CIFAR10': torch.optim.SGD(self.model[me].parameters(), lr=0.01, momentum=0.9),
                    'GTSRB': torch.optim.SGD(self.model[me].parameters(), lr=0.01, momentum=0.9),
                    'WISDM-W': torch.optim.RMSprop(self.model[me].parameters(), lr=0.001, momentum=0.9),
                    'WISDM-P': torch.optim.RMSprop(self.model[me].parameters(), lr=0.001, momentum=0.9),
                    'ImageNet100': torch.optim.SGD(self.model[me].parameters(), lr=0.01, momentum=0.9),
                    'ImageNet': torch.optim.SGD(self.model[me].parameters(), lr=0.1),
                    'ImageNet10': torch.optim.SGD(self.model[me].parameters(), lr=0.1),
                    "ImageNet_v2": torch.optim.Adam(self.model[me].parameters(), lr=0.01),
                    "Gowalla": torch.optim.RMSprop(self.model[me].parameters(), lr=0.00005)}[dataset_name]
        except Exception as e:
            print("_get_optimizer error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))



        



