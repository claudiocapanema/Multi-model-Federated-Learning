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
    def __init__(self, args, id, model, fold_id):
        try:
            self.args = args
            self.fold_id = fold_id
            g = torch.Generator()
            g.manual_seed(id + self.fold_id)
            random.seed(id + self.fold_id)
            np.random.seed(id + self.fold_id)
            torch.manual_seed(id + self.fold_id)
            self.dataset = args.dataset
            self.batch_size = []
            for dataset in args.dataset:
                self.batch_size.append({"CIFAR10": 32, "SVHN": 32, "MNIST": 32, "F-MNIST": 32, "EMNIST": 32, "WISDM-W": 64, "ImageNet10": 10, "Gowalla": 64, "wikitext": 256, "Foursquare": 512}[dataset])
            self.lr_dict = {'EMNIST':0.01,
                            'MNIST': 0.01,
                            "F-MNIST": 0.01,
                            'CIFAR10': 0.01,
                            'GTSRB': 0.01,
                            "SVHN": 0.01,
                            'WISDM-W': 0.001,
                            'WISDM-P': 0.001,
                            'ImageNet100': 0.01,
                            'ImageNet': 0.1,
                            'ImageNet10': 0.01,
                            "ImageNet_v2": 0.01,
                            "Gowalla": 0.001,
                            "wikitext": 0.001,
                            "Foursquare": 0.001}
            self.model = model
            self.alpha_train = [float(i) for i in args.alpha]
            self.alpha_test = [float(i) for i in args.alpha]
            self.ME = len(self.model)
            self.concept_drift_window_train = [0] * self.ME
            self.concept_drift_window_test = [0] * self.ME
            self.total_clients  = args.total_clients

            self.num_examples = [0] * self.ME

            self.number_of_rounds = args.number_of_rounds
            print("Preparing data...")
            print("""args do cliente: {} {}""".format(self.args.client_id, self.alpha_train))
            self.client_id = id
            self.trainloader = [None] * self.ME
            self.valloader = [None] * self.ME
            self.recent_trainloader = [None] * self.ME
            self.optimizer = [None] * self.ME
            self.p_ME, self.fc_ME, self.il_ME = [0] * self.ME, [0] * self.ME, [0] * self.ME
            # self.num_examples = [0] * self.ME
            self.index = 0
            self.local_epochs = self.args.local_epochs
            self.lr = self.args.learning_rate
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.lt = [-1] * self.ME
            print("ler model size")
            self.models_size = self._get_models_size()
            self.n_classes = [
                {'EMNIST': 47, 'MNIST': 10, 'F-MNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
                 "ImageNet10": 10, "ImageNet_v2": 15, "Gowalla": 7, "wikitext": 30, "Foursquare": 10}[dataset] for dataset in
                self.args.dataset]
            self.loss_ME = [10] * self.ME
            # Concept drift parameters
            self.experiment_id = self.args.experiment_id
            self.gradual_rounds = 5

            # Número de rodadas usadas para realizar a transição do alpha
            self.label_shift_transition_window = self.args.label_shift_transition_window
            self.data_shift_config = self.get_data_shift_config(
                self.ME,
                self.number_of_rounds,
                self.alpha_train,
                self.experiment_id,
                self.client_id,
                gradual_rounds=self.total_clients // self.gradual_rounds,
                seed=self.fold_id
            )
            print(f"data shift config {self.data_shift_config} data shift id {self.experiment_id}")

            for me in range(self.ME):
                # self.trainloader[me], self.valloader[me] = load_data(
                #     dataset_name=self.args.dataset[me],
                #     alpha=self.alpha_train[me],
                #     data_sampling_percentage=self.args.data_percentage,
                #     partition_id=self.client_id,
                #     num_partitions=self.args.total_clients + 1,
                #     batch_size=self.batch_size[me],
                #     fold_id=self.fold_id,
                # )
                self.update_local_train_data(1, me)
                # self.update_local_test_data(1, me)
                self.optimizer[me] = self._get_optimizer(dataset_name=self.args.dataset[me], me=me)
                print("""leu dados cid: {} dataset: {} size:  {}""".format(self.client_id, self.args.dataset[me],
                                                                                 len(self.trainloader[me].dataset)))

                self.p_ME[me], self.fc_ME[me], self.il_ME[me] = self._get_datasets_metrics(self.trainloader, self.ME,
                                                                               self.client_id,
                                                                               self.n_classes, me=me)
        except Exception as e:
            print("__init__ client error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def label_shift_config(self, ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds):
        try:
            if len(experiment_id) > 0:
                if experiment_id == "label_shift#0.1-1.0_sudden":
                    assert all(i == 0.1 for i in self.alpha_train)
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    new_alphas = [[1.0], [1.0], [1.0]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#0.1-10.0_sudden":
                    assert all(i == 0.1 for i in self.alpha_train)
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    new_alphas = [[10.0], [10.0], [10.0]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#1.0-0.1_sudden":
                    assert all(i == 1.0 for i in self.alpha_train)
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    new_alphas = [[0.1], [0.1], [0.1]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#1.0-10.0_sudden":
                    assert all(i == 1.0 for i in self.alpha_train)
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    new_alphas = [[10.0], [10.0], [10.0]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#10.0-0.1_sudden":
                    assert all(i == 10.0 for i in self.alpha_train)
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    new_alphas = [[0.1], [0.1], [0.1]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#10.0-1.0_sudden":
                    assert all(i == 10.0 for i in self.alpha_train)
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    new_alphas = [[1.0], [1.0], [1.0]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#0.1-10.0_gradual":

                    ME_concept_drift_rounds = [
                        [int(n_rounds * 0.3)],
                        [int(n_rounds * 0.5)],
                        [int(n_rounds * 0.7)]
                    ]

                    new_alphas = [[10.0], [10.0], [10.0]]

                    type_ = "label_shift"

                    config = {
                        me: {
                            "data_shift_rounds": ME_concept_drift_rounds[me],
                            "new_alphas": new_alphas[me],
                            "transition_window": self.label_shift_transition_window,
                            "type": type_
                        }
                        for me in range(ME)
                    }
                elif experiment_id == "label_shift#0.1-10.0_recurrent":
                    ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                               [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                    new_alphas = [[10.0, 0.1], [10.0, 0.1], [10.0, 0.1]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#10.0-0.1_sudden":
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    new_alphas = [[0.1], [0.1], [0.1]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#10.0-0.1_gradual":

                    ME_concept_drift_rounds = [
                        [int(n_rounds * 0.3)],
                        [int(n_rounds * 0.5)],
                        [int(n_rounds * 0.7)]
                    ]

                    new_alphas = [[0.1], [0.1], [0.1]]

                    type_ = "label_shift"

                    config = {
                        me: {
                            "data_shift_rounds": ME_concept_drift_rounds[me],
                            "new_alphas": new_alphas[me],
                            "transition_window": self.label_shift_transition_window,
                            "type": type_
                        }
                        for me in range(ME)
                    }
                elif experiment_id == "label_shift#10.0-0.1_recurrent":
                    ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                               [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                    new_alphas = [[0.1, 10.0], [0.1, 10.0], [0.1, 10.0]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#3_sudden":
                    ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                               [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                    new_alphas = [[10.0, 1.0], [10.0, 1.0], [10.0, 1.0]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                elif experiment_id == "label_shift#4_sudden":
                    ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                               [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                    new_alphas = [[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]]
                    type_ = "label_shift"
                    config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                                   "type": type_} for me in range(ME)}
                else:
                    config = {}



            else:
                config = {}
            # else:
            #     config = {}

            if len(config) == 0 and len(experiment_id) > 0:
                raise Exception(f"Experiment id {experiment_id} not supported")

            return config

        except Exception as e:
            print("label_shift_config error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
            exit()

    def global_concept_drift_config(self, ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds):
        try:
            if experiment_id == "concept_drift#0.1_sudden":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[0.1], [0.1], [0.1]]
                new_concept_drift_window = [[1], [1], [1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me], "type": type_} for me in
                          range(ME)}
            elif experiment_id == "concept_drift#0.1_gradual":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.5) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.7) + client_id // gradual_rounds]]
                new_alphas = [[0.1], [0.1], [0.1]]
                new_concept_drift_window = [[1], [1], [1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me], "type": type_} for me in
                          range(ME)}
            elif experiment_id == "concept_drift#0.1_recurrent":
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                           [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                new_alphas = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
                new_concept_drift_window = [[1, 0], [1, 0], [1, 0]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me], "type": type_} for me in
                          range(ME)}
            elif experiment_id == "concept_drift#10.0_sudden":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[10.0], [10.0], [10.0]]
                new_concept_drift_window = [[1], [1], [1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me], "type": type_} for me in
                          range(ME)}
            elif experiment_id == "concept_drift#10.0_gradual":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.5) + client_id // gradual_rounds],
                                           [int(n_rounds * 0.7) + client_id // gradual_rounds]]
                new_alphas = [[10.0], [10.0], [10.0]]
                new_concept_drift_window = [[1], [1], [1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me], "type": type_} for me in
                          range(ME)}
            elif experiment_id == "concept_drift#10.0_recurrent":
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.3), int(n_rounds * 0.6)],
                                           [int(n_rounds * 0.4), int(n_rounds * 0.7)]]
                new_alphas = [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
                new_concept_drift_window = [[1, 0], [1, 0], [1, 0]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me], "type": type_} for me in
                          range(ME)}

            elif experiment_id == "concept_drift#1.0_sudden":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[1.0], [1.0], [1.0]]
                new_concept_drift_window = [[1], [1], [1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me], "type": type_} for me in
                          range(ME)}


            else:
                config = {}

            if len(config) == 0 and len(experiment_id) > 0:
                raise Exception(f"Experiment id {experiment_id} not supported")

            return config

        except Exception as e:
            print("global_concept_drift_config error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def get_data_shift_config(self, ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds, seed):

        try:

            if "label_shift" in experiment_id:
                return self.label_shift_config(ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds)
            elif "concept_drift" in experiment_id:
                return self.global_concept_drift_config(ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds)
            else:
                return {}

        except Exception as e:
            print("get_data_shift_config error")
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
            g.manual_seed(t+self.fold_id)
            random.seed(t+self.fold_id)
            np.random.seed(t+self.fold_id)
            torch.manual_seed(t+self.fold_id)
            set_weights(self.model[me], global_model)

            # Update alpha to simulate data shift
            if t > 1:
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
                self.concept_drift_window_train[me]
            )
            results["me"] = me
            results["client_id"] = self.client_id
            results["Model size"] = self.models_size[me]
            results["alpha"] = self.alpha_train[me]
            self.loss_ME[me] = results["train_loss"]
            return get_weights(self.model[me]), len(self.trainloader[me].dataset), results
        except Exception as e:
            print("fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, me, t, global_model):
        """Evaluate the model on the data this client has."""
        try:
            g = torch.Generator()
            g.manual_seed(t+self.fold_id)
            random.seed(t+self.fold_id)
            np.random.seed(t+self.fold_id)
            torch.manual_seed(t+self.fold_id)
            tuple_me = {}
            nt = t - self.lt[me]
            self.update_local_test_data(t, me)
            set_weights(self.model[me], global_model)
            loss, metrics = test(self.model[me], self.valloader[me], self.device, self.client_id, t,
                                 self.args.dataset[me], self.n_classes[me], self.concept_drift_window_test[me])
            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            metrics["Alpha"] = self.alpha_test[me]
            tuple_me = (loss, len(self.valloader[me].dataset), metrics)
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _apply_concept_drift_to_trainloader(
            self,
            trainloader,
            me,
            concept_drift_window
    ):
        """
        Apply a pure concept-drift transformation to the local
        training data.

        The transformation preserves exactly the marginal label
        distribution P(Y), while changing the association between
        samples X and labels Y.

        Therefore:

            P_new(Y) = P_old(Y)

        while:

            P_new(X | Y) != P_old(X | Y)

        The transformation is applied ONLY to the local training
        data. Test/validation data are never used.

        recent_trainloader is not modified by this method.
        """

        try:

            if trainloader is None:
                return trainloader

            if concept_drift_window is None:
                return trainloader

            concept_drift_window = int(
                concept_drift_window
            )

            if concept_drift_window == 0:
                return trainloader

            # ---------------------------------------------------------
            # Extract all samples from the current training loader.
            #
            # We keep the complete batch dictionaries because different
            # datasets use different input keys.
            # ---------------------------------------------------------

            samples = []

            for batch in trainloader:

                if not isinstance(batch, dict):
                    raise TypeError(
                        "Expected training batches to be dictionaries."
                    )

                batch_size = (
                    batch["label"].shape[0]
                )

                for i in range(batch_size):

                    sample = {}

                    for key, value in batch.items():

                        if isinstance(value, torch.Tensor):

                            sample[key] = (
                                value[i].detach().cpu().clone()
                            )

                        else:

                            sample[key] = copy.deepcopy(
                                value[i]
                            )

                    samples.append(
                        sample
                    )

            if len(samples) <= 1:
                return trainloader

            # ---------------------------------------------------------
            # Original labels.
            # ---------------------------------------------------------

            original_labels = np.asarray(
                [
                    int(
                        sample["label"].item()
                        if isinstance(
                            sample["label"],
                            torch.Tensor
                        )
                        else sample["label"]
                    )
                    for sample in samples
                ],
                dtype=np.int64
            )

            # ---------------------------------------------------------
            # Deterministic permutation.
            #
            # The permutation depends on client, model and the
            # configured concept-drift window.
            # ---------------------------------------------------------

            seed = (
                    42
                    + int(self.client_id) * 100003
                    + int(me) * 1009
                    + int(concept_drift_window) * 9176
            )

            rng = np.random.RandomState(
                seed
            )

            permutation = rng.permutation(
                len(original_labels)
            )

            # Avoid the identity permutation.
            if np.array_equal(
                    permutation,
                    np.arange(
                        len(original_labels)
                    )
            ):
                permutation = np.roll(
                    permutation,
                    1
                )

            shifted_labels = (
                original_labels[permutation]
            )

            # ---------------------------------------------------------
            # Safety check:
            #
            # The marginal label distribution MUST be exactly the same.
            # ---------------------------------------------------------

            original_unique, original_counts = np.unique(
                original_labels,
                return_counts=True
            )

            shifted_unique, shifted_counts = np.unique(
                shifted_labels,
                return_counts=True
            )

            if (
                    not np.array_equal(
                        original_unique,
                        shifted_unique
                    )
                    or not np.array_equal(
                original_counts,
                shifted_counts
            )
            ):
                raise RuntimeError(
                    "Concept-drift transformation changed "
                    "the marginal label distribution."
                )

            # ---------------------------------------------------------
            # Build a new list of samples.
            #
            # Only the label is replaced.
            # Every X remains associated with the same physical sample.
            # ---------------------------------------------------------

            shifted_samples = []

            for i, sample in enumerate(samples):
                shifted_sample = copy.deepcopy(
                    sample
                )

                shifted_sample["label"] = torch.tensor(
                    shifted_labels[i],
                    dtype=(
                        sample["label"].dtype
                        if isinstance(
                            sample["label"],
                            torch.Tensor
                        )
                        else torch.long
                    )
                )

                shifted_samples.append(
                    shifted_sample
                )

            # ---------------------------------------------------------
            # Local Dataset wrapper.
            # ---------------------------------------------------------

            class _ConceptDriftDataset(
                torch.utils.data.Dataset
            ):

                def __init__(
                        self,
                        data
                ):
                    self.data = data

                def __len__(
                        self
                ):
                    return len(
                        self.data
                    )

                def __getitem__(
                        self,
                        index
                ):
                    return self.data[index]

            shifted_dataset = (
                _ConceptDriftDataset(
                    shifted_samples
                )
            )

            # ---------------------------------------------------------
            # Preserve the important DataLoader parameters.
            # ---------------------------------------------------------

            loader_kwargs = {
                "batch_size": (
                    trainloader.batch_size
                ),
                "shuffle": True,
                "num_workers": (
                    trainloader.num_workers
                ),
                "drop_last": (
                    trainloader.drop_last
                ),
                "pin_memory": (
                    trainloader.pin_memory
                )
            }

            # Preserve the existing collate function whenever possible.
            if (
                    hasattr(
                        trainloader,
                        "collate_fn"
                    )
                    and trainloader.collate_fn
                    is not None
            ):
                loader_kwargs["collate_fn"] = (
                    trainloader.collate_fn
                )

            shifted_trainloader = (
                torch.utils.data.DataLoader(
                    shifted_dataset,
                    **loader_kwargs
                )
            )

            print(
                f"[CONCEPT DRIFT] "
                f"client={self.client_id} "
                f"model={me} "
                f"window={concept_drift_window} "
                f"samples={len(samples)} "
                f"P(Y)_preserved=True"
            )

            return shifted_trainloader

        except Exception as e:

            print(
                "_apply_concept_drift_to_trainloader error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            return trainloader

    def update_local_train_data(
            self,
            t,
            me
    ):
        try:

            if t == 1:

                self.trainloader[me], self.valloader[me] = load_data(
                    dataset_name=self.args.dataset[me],
                    alpha=self.alpha_train[me],
                    data_sampling_percentage=self.args.data_percentage,
                    partition_id=self.client_id,
                    num_partitions=self.args.total_clients + 1,
                    batch_size=self.batch_size[me],
                    fold_id=self.fold_id,
                )

                # -----------------------------------------------------
                # KEEP THE EXISTING recent_trainloader BEHAVIOR.
                # -----------------------------------------------------

                self.recent_trainloader[me] = (
                    copy.deepcopy(
                        self.trainloader[me]
                    )
                )

                self.num_examples[me] = (
                    len(
                        self.trainloader[me].dataset
                    )
                )

            else:

                if self.data_shift_config != {}:

                    (
                        alpha_me,
                        concept_drift_window,
                        data_shift_flag
                    ) = self._data_shift_flag(
                        t,
                        me,
                        train=True
                    )

                    print(
                        f"Treinar modelo {me} "
                        f"rodada {t} "
                        f"cliente {self.client_id} - "
                        f"data drift flag "
                        f"{data_shift_flag} "
                        f"comparacao alpha "
                        f"{self.alpha_train[me]} "
                        f"novo {alpha_me} - "
                        f"concept drift_window antigo "
                        f"{self.concept_drift_window_train[me]} "
                        f"novo {concept_drift_window}"
                    )

                    print(
                        self.data_shift_config
                    )

                    # =================================================
                    # LABEL SHIFT
                    # =================================================

                    if (
                            data_shift_flag
                            and self.data_shift_config[me]["type"]
                            in ["label_shift"]
                    ):

                        if (
                                self.alpha_train[me]
                                != self.alpha_test[me]
                                and self.alpha_test[me]
                                == alpha_me
                        ):

                            self.alpha_train[me] = (
                                self.alpha_test[me]
                            )

                            self.trainloader[me] = (
                                copy.deepcopy(
                                    self.recent_trainloader[me]
                                )
                            )

                            (
                                self.p_ME[me],
                                self.fc_ME[me],
                                self.il_ME[me]
                            ) = self._get_datasets_metrics(
                                self.trainloader,
                                self.ME,
                                self.client_id,
                                self.n_classes,
                                me=me
                            )

                        else:

                            print(
                                f"Atualizou dataset de treino "
                                f"do modelo {me} "
                                f"na rodada {t}. "
                                f"Alpha de "
                                f"{self.alpha_train[me]} "
                                f"para {alpha_me} - "
                                f"concept drift_window antigo "
                                f"{self.concept_drift_window_train[me]} "
                                f"novo {concept_drift_window} - "
                                f"cliente {self.client_id}"
                            )

                            self.alpha_train[me] = (
                                alpha_me
                            )

                            self.alpha_test[me] = (
                                alpha_me
                            )

                            self.trainloader[me], self.valloader[me] = load_data(
                                dataset_name=self.args.dataset[me],
                                alpha=self.alpha_train[me],
                                data_sampling_percentage=self.args.data_percentage,
                                partition_id=self.client_id,
                                num_partitions=self.args.total_clients + 1,
                                batch_size=self.batch_size[me],
                                fold_id=self.fold_id,
                            )

                            # -------------------------------------------------
                            # KEEP EXISTING recent_trainloader BEHAVIOR.
                            # -------------------------------------------------

                            self.recent_trainloader[me] = (
                                copy.deepcopy(
                                    self.trainloader[me]
                                )
                            )

                            (
                                self.p_ME[me],
                                self.fc_ME[me],
                                self.il_ME[me]
                            ) = self._get_datasets_metrics(
                                self.trainloader,
                                self.ME,
                                self.client_id,
                                self.n_classes,
                                me=me
                            )

                    # =================================================
                    # CONCEPT DRIFT
                    # =================================================

                    elif (
                            data_shift_flag
                            and self.data_shift_config[me]["type"]
                            in ["concept_drift"]
                    ):

                        print(
                            f"Aplicando concept drift "
                            f"puro ao modelo {me} "
                            f"na rodada {t} "
                            f"cliente {self.client_id}. "
                            f"window antigo "
                            f"{self.concept_drift_window_train[me]} "
                            f"novo "
                            f"{concept_drift_window}"
                        )

                        # -------------------------------------------------
                        # Concept drift does NOT change alpha.
                        # -------------------------------------------------

                        self.alpha_train[me] = (
                            alpha_me
                        )

                        self.alpha_test[me] = (
                            alpha_me
                        )

                        self.concept_drift_window_train[me] = (
                            concept_drift_window
                        )

                        # -------------------------------------------------
                        # Apply the shift to the ACTUAL TRAINING DATA.
                        #
                        # This preserves P(Y), but changes P(X|Y).
                        # -------------------------------------------------

                        self.trainloader[me] = (
                            self._apply_concept_drift_to_trainloader(
                                self.trainloader[me],
                                me,
                                concept_drift_window
                            )
                        )

                        # -------------------------------------------------
                        # IMPORTANT:
                        #
                        # Do NOT transform labels inside
                        # _get_datasets_metrics().
                        # -------------------------------------------------

                        (
                            self.p_ME[me],
                            self.fc_ME[me],
                            self.il_ME[me]
                        ) = self._get_datasets_metrics(
                            self.trainloader,
                            self.ME,
                            self.client_id,
                            self.n_classes,
                            me=me
                        )

                self.num_examples[me] = (
                    len(
                        self.trainloader[me].dataset
                    )
                )

        except Exception as e:

            print(
                f"update_local_train_data error "
                f"{self.data_shift_config}"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

    def update_local_test_data(self, t, me):

        try:

            if self.data_shift_config != {}:
                alpha_me, concept_drift_window, data_shift_flag = self._data_shift_flag(t, me, train=False)
                if (self.alpha_test[me] != alpha_me and data_shift_flag and self.data_shift_config[me]["type"] in ["label_shift"]):
                    print(f"Atualizou dataset de teste de {me} rodada {t}. Alpha de {self.alpha_test[me]} para {alpha_me} - cliente {self.client_id} - concept drift de {self.concept_drift_window_test} para {concept_drift_window}")
                    self.alpha_test[me] = alpha_me
                    self.recent_trainloader[me], self.valloader[me] = load_data(
                        dataset_name=self.args.dataset[me],
                        alpha=self.alpha_test[me],
                        data_sampling_percentage=self.args.data_percentage,
                        partition_id=self.client_id,
                        num_partitions=self.args.total_clients + 1,
                        batch_size=self.batch_size[me],
                        fold_id=self.fold_id,
                    )
                    p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
                elif data_shift_flag and self.data_shift_config[me]["type"] in ["concept_drift"] and t - self.lt[me] > 0:
                    print(
                        f"Atualizou dataset de teste de {me} rodada {t}. Alpha de {self.alpha_test[me]} para {alpha_me} - cliente {self.client_id} - concept drift de {self.concept_drift_window_test[me]} para {concept_drift_window}")
                    # Since we assume every client is tested in every round, its local test data is updated once every
                    # time data shift occurs
                    # Recurrent
                    self.concept_drift_window_test[me] = concept_drift_window
                    p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
                else:
                    p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
            else:
                p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME

            return p_ME, fc_ME, il_ME

        except Exception as e:
            print(f"update_local_train_data error {self.data_shift_config}")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_current_alpha(self, server_round, me, train):
        """
        Retorna o alpha atual.

        Para cenários sudden:
            alpha muda instantaneamente.

        Para cenários gradual:
            alpha é interpolado linearmente ao longo de
            transition_window rodadas.
        """

        try:

            reference_alpha = (
                self.alpha_train[me]
                if train
                else self.alpha_test[me]
            )

            if self.data_shift_config == {}:
                return reference_alpha, False

            config = self.data_shift_config[me]

            shift_rounds = config["data_shift_rounds"]
            target_alphas = config["new_alphas"]

            transition_window = config.get(
                "transition_window",
                1
            )

            alpha = None

            initial_alpha = float(self.alpha_train[me])

            for i, start_round in enumerate(shift_rounds):

                target_alpha = float(target_alphas[i])

                if "gradual" in self.experiment_id:

                    end_round = start_round + transition_window

                    if server_round < start_round:
                        continue

                    elif start_round <= server_round < end_round:

                        progress = (
                                (server_round - start_round)
                                / transition_window
                        )

                        alpha = (
                                initial_alpha
                                + progress *
                                (target_alpha - initial_alpha)
                        )

                        break

                    else:
                        alpha = target_alpha
                        initial_alpha = target_alpha

                else:

                    if server_round >= start_round:
                        alpha = target_alpha

            if alpha is None:
                alpha = initial_alpha

            return alpha, abs(alpha - reference_alpha) > 1e-8

        except Exception as e:
            print(f"_get_current_alpha error {self.data_shift_config}")
            print("""Error on line {} {} {}""".format(
                sys.exc_info()[-1].tb_lineno,
                type(e).__name__,
                e
            ))

    def _data_shift_flag(self, server_round, me, train):

        try:
            alpha, label_shift_flag = self._get_current_alpha(server_round, me, train)
            concept_drift_window, concept_drift_flag = self._check_concept_drift(server_round, me, train)
            return alpha, concept_drift_window, True in [label_shift_flag, concept_drift_flag]

        except Exception as e:
            print(f"_data_shift_flag error {self.data_shift_config}")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _check_concept_drift(self, server_round, me, train):

        try:
            if self.data_shift_config == {} or self.data_shift_config[me]["type"] != "concept_drift":
                return 0, False
            else:
                reference_concept_drift_window = self.concept_drift_window_train[me] if train else self.concept_drift_window_test[me]
                config = self.data_shift_config[me]
                new_concept_drift_window = 0
                flag = False

                for i, round_ in enumerate(config["data_shift_rounds"]):
                    if server_round >= round_:
                        new_concept_drift_window = config["new_concept_drift_window"][i]

                flag = new_concept_drift_window != reference_concept_drift_window

                return new_concept_drift_window, flag
        except Exception as e:
            print(f"_check_concept_drift error {self.data_shift_config}")
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
                    'F-MNIST': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'CIFAR10': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'SVHN': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'GTSRB': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'WISDM-W': torch.optim.RMSprop(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'WISDM-P': torch.optim.RMSprop(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'ImageNet100': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
                    'ImageNet': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    'ImageNet10': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    "ImageNet_v2": torch.optim.Adam(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    "Gowalla": torch.optim.RMSprop(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    "wikitext": torch.optim.RMSprop(self.model[me].parameters(), self.lr_dict[dataset_name]),
                    "Foursquare": torch.optim.Adam(self.model[me].parameters(), self.lr_dict[dataset_name]),}[dataset_name]
        except Exception as e:
            print("_get_optimizer error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_datasets_metrics(
            self,
            trainloader,
            ME,
            client_id,
            n_classes,
            concept_drift_window=None,
            me=None
    ):
        """
        Compute local training-data metrics.

        IMPORTANT:

        The labels are read exactly as they are stored in the
        training loader.

        concept_drift_window is NOT applied here.

        Concept drift is simulated by changing the actual
        sample-label relationship in the training loader.

        Therefore this method measures the real P(Y) of the
        training data and can be safely used to calculate LS.
        """

        try:

            p_ME = []
            fc_ME = []
            il_ME = []

            ME_LIST = (
                [i for i in range(ME)]
                if me is None
                else [me]
            )

            for model_id in ME_LIST:

                labels_me = []

                n_classes_me = (
                    n_classes[model_id]
                )

                p_me = {
                    i: 0
                    for i in range(
                        n_classes_me
                    )
                }

                with torch.no_grad():

                    for batch in trainloader[model_id]:

                        labels = (
                            batch["label"]
                        )

                        if not isinstance(
                                labels,
                                torch.Tensor
                        ):
                            labels = torch.tensor(
                                labels
                            )

                        labels = (
                            labels.detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )

                        labels_me.extend(
                            labels.tolist()
                        )

                # -----------------------------------------------------
                # No samples.
                # -----------------------------------------------------

                if len(labels_me) == 0:
                    p_ME.append(
                        np.zeros(
                            n_classes_me,
                            dtype=float
                        )
                    )

                    fc_ME.append(
                        0.0
                    )

                    il_ME.append(
                        0.0
                    )

                    continue

                # -----------------------------------------------------
                # Count labels.
                # -----------------------------------------------------

                unique, count = np.unique(
                    labels_me,
                    return_counts=True
                )

                data_unique_count_dict = dict(
                    zip(
                        unique.tolist(),
                        count.tolist()
                    )
                )

                for label, label_count in (
                        data_unique_count_dict.items()
                ):

                    label = int(
                        label
                    )

                    if (
                            0 <= label
                            < n_classes_me
                    ):
                        p_me[label] = (
                            label_count
                        )

                p_me = np.asarray(
                    list(
                        p_me.values()
                    ),
                    dtype=float
                )

                total_samples = (
                    np.sum(
                        p_me
                    )
                )

                if total_samples <= 0:
                    p_ME.append(
                        np.zeros(
                            n_classes_me,
                            dtype=float
                        )
                    )

                    fc_ME.append(
                        0.0
                    )

                    il_ME.append(
                        0.0
                    )

                    continue

                # -----------------------------------------------------
                # Fraction of represented classes.
                # -----------------------------------------------------

                fc_me = (
                        np.count_nonzero(
                            p_me > 0
                        )
                        / n_classes_me
                )

                # -----------------------------------------------------
                # Imbalance level.
                # -----------------------------------------------------

                expected = (
                        total_samples
                        / n_classes_me
                )

                il_me = (
                        np.count_nonzero(
                            p_me < expected
                        )
                        / n_classes_me
                )

                # -----------------------------------------------------
                # Convert counts into P(Y).
                # -----------------------------------------------------

                p_me = (
                        p_me
                        / total_samples
                )

                p_ME.append(
                    p_me
                )

                fc_ME.append(
                    fc_me
                )

                il_ME.append(
                    il_me
                )

            if (
                    len(p_ME) == 1
                    and len(il_ME) == 1
                    and len(fc_ME) == 1
            ):
                return (
                    p_ME[0],
                    fc_ME[0],
                    il_ME[0]
                )

            return (
                p_ME,
                fc_ME,
                il_ME
            )

        except Exception as e:

            print(
                "_get_datasets_metrics error"
            )

            try:

                print(
                    f"Dataset "
                    f"{self.args.dataset[me]}"
                )

            except Exception:
                pass

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )







