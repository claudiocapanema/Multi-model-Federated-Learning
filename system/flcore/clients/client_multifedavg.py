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
import re
import torch
import numpy as np
from .utils.models_utils import load_model, get_weights, load_data, set_weights, test, train

DATASET_INPUT_MAP = {"CIFAR10": "img", "CINIC10": "image", "MNIST": "image", "EMNIST": "image", "F-MNIST": "image", "SVHN": "image", "GTSRB": "image", "Gowalla": "sequence",
                     "WISDM-W": "sequence", "ImageNet": "image", "ImageNet10": "image", "wikitext": "sequence", "Foursquare": "sequence"}


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
                self.batch_size.append({"CIFAR10": 32, "CINIC10": 32, "SVHN": 32, "MNIST": 32, "F-MNIST": 32, "EMNIST": 32, "WISDM-W": 64, "ImageNet10": 32, "Gowalla": 64, "wikitext": 256, "Foursquare": 512}[dataset])
            self.lr_dict = {'EMNIST':0.01,
                            'MNIST': 0.01,
                            "F-MNIST": 0.01,
                            'CIFAR10': 0.01,
                            'CINIC10': 0.01,
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
            self.initial_alpha = [float(i) for i in args.alpha]
            self.ME = len(self.model)
            self.label_shift_progress_train = [0.0] * self.ME
            self.label_shift_progress_test = [0.0] * self.ME

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

            # Source-data cache for gradual label/combined shift.  The cache
            # is separate from the active trainloader so delayed labeling is
            # preserved: training data changes only when fit() invokes
            # update_local_train_data() for a selected client.
            self.label_shift_data_cache = [dict() for _ in range(self.ME)]

            # Cache endpoint label-distribution metrics so gradual transitions
            # never rescan the complete training dataset on every round.
            # Keys are (alpha, partition_seed).
            self.label_shift_metrics_cache = [dict() for _ in range(self.ME)]

            self.optimizer = [None] * self.ME
            self.p_ME, self.fc_ME, self.il_ME = [0] * self.ME, [0] * self.ME, [0] * self.ME
            # self.num_examples = [0] * self.ME
            self.index = 0
            self.local_epochs = self.args.local_epochs
            self.lr = self.args.learning_rate
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.lt = [-1] * self.ME
            self.partition_seed_train = [1] * self.ME
            self.partition_seed_test = [1] * self.ME
            print("ler model size")
            self.models_size = self._get_models_size()
            self.n_classes = [
                {'EMNIST': 47, 'MNIST': 10, 'F-MNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'CINIC10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
                 "ImageNet10": 10, "ImageNet_v2": 15, "Gowalla": 7, "wikitext": 25, "Foursquare": 10}[dataset] for dataset in
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
                self.label_shift_metrics_cache[me][
                    (float(self.alpha_train[me]), int(self.partition_seed_train[me]))
                ] = (
                    np.asarray(self.p_ME[me], dtype=float).copy(),
                    float(self.fc_ME[me]),
                    float(self.il_ME[me]),
                )
        except Exception as e:
            print("__init__ client error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def label_shift_config(self, ME, n_rounds, alphas, experiment_id, client_id, gradual_rounds):
        try:
            if len(experiment_id) > 0:
                # ---------------------------------------------------------
                # Seed-based label shift experiments.
                #
                # These configurations deliberately keep alpha unchanged.
                # The label distribution changes because the Dirichlet
                # partition is regenerated with a different integer seed.
                # The experiment suffixes 0.1, 1.0 and 10.0 identify the
                # configured partition seeds 1, 10 and 100, respectively.
                # ---------------------------------------------------------
                if experiment_id == "label_shift#0.1_sudden":
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    partition_seeds = [[2], [2], [2]]
                    type_ = "label_shift"
                    config = {me: {
                        "data_shift_rounds": ME_concept_drift_rounds[me],
                        "new_alphas": [float(alphas[me])],
                        "partition_seeds": partition_seeds[me],
                        "type": type_
                    } for me in range(ME)}
                elif experiment_id == "label_shift#1.0_sudden":
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    partition_seeds = [[10], [10], [10]]
                    type_ = "label_shift"
                    config = {me: {
                        "data_shift_rounds": ME_concept_drift_rounds[me],
                        "new_alphas": [float(alphas[me])],
                        "partition_seeds": partition_seeds[me],
                        "type": type_
                    } for me in range(ME)}
                elif experiment_id == "label_shift#10.0_sudden":
                    ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                               [int(n_rounds * 0.5)],
                                               [int(n_rounds * 0.7)]]
                    partition_seeds = [[100], [100], [100]]
                    type_ = "label_shift"
                    config = {me: {
                        "data_shift_rounds": ME_concept_drift_rounds[me],
                        "new_alphas": [float(alphas[me])],
                        "partition_seeds": partition_seeds[me],
                        "type": type_
                    } for me in range(ME)}
                elif experiment_id == "label_shift#0.1-1.0_sudden":
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
                elif re.fullmatch(r"label_shift#(?:0\.1|1\.0|10\.0)-(?:0\.1|1\.0|10\.0)_gradual", experiment_id):
                    # Generic gradual label-shift transition.
                    # The experiment id encodes the fixed initial alpha and
                    # target alpha (e.g. 0.1-1.0). The transition is performed
                    # by mixing endpoint datasets; alpha itself is not
                    # interpolated.
                    transition = experiment_id[len("label_shift#"):-len("_gradual")]
                    initial_alpha_str, target_alpha_str = transition.split("-", 1)
                    initial_alpha = float(initial_alpha_str)
                    target_alpha = float(target_alpha_str)

                    if not all(abs(float(alpha) - initial_alpha) <= 1e-8 for alpha in self.alpha_train):
                        raise ValueError(
                            f"{experiment_id} requires initial alpha={initial_alpha}, "
                            f"but received {self.alpha_train}"
                        )

                    ME_concept_drift_rounds = [
                        [int(n_rounds * 0.3)],
                        [int(n_rounds * 0.5)],
                        [int(n_rounds * 0.7)]
                    ]
                    new_alphas = [[target_alpha] for _ in range(ME)]
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
                elif experiment_id.startswith("combined_shift#") and experiment_id.endswith("_gradual"):
                    combined_transitions = {
                        "0.1_1.0": (0.1, 1.0), "0.1-1.0": (0.1, 1.0),
                        "0.1_10.0": (0.1, 10.0), "0.1-10.0": (0.1, 10.0),
                        "1.0_0.1": (1.0, 0.1), "1.0-0.1": (1.0, 0.1),
                        "1.0_10.0": (1.0, 10.0), "1.0-10.0": (1.0, 10.0),
                        "10.0_0.1": (10.0, 0.1), "10.0-0.1": (10.0, 0.1),
                        "10.0_1.0": (10.0, 1.0), "10.0-1.0": (10.0, 1.0),
                    }
                    transition_key = experiment_id.replace("combined_shift#", "").replace("_gradual", "")
                    if transition_key not in combined_transitions:
                        config = {}
                    else:
                        initial_alpha, target_alpha = combined_transitions[transition_key]
                        if not all(abs(float(alpha) - initial_alpha) <= 1e-8 for alpha in self.alpha_train):
                            raise ValueError(
                                f"{experiment_id} requires initial alpha={initial_alpha}, but received {self.alpha_train}"
                            )
                        rounds = [[int(n_rounds * 0.3)], [int(n_rounds * 0.5)], [int(n_rounds * 0.7)]]
                        config = {
                            me: {
                                "data_shift_rounds": rounds[me],
                                "new_alphas": [target_alpha],
                                "new_concept_drift_window": [1],
                                "transition_window": self.label_shift_transition_window,
                                "type": "combined_shift",
                            }
                            for me in range(ME)
                        }

                elif experiment_id.startswith("combined_shift#") and experiment_id.endswith("_sudden"):
                    # ---------------------------------------------------------
                    # Combined shift:
                    #   - alpha changes from the value encoded before "_"
                    #     to the value encoded after "_";
                    #   - concept drift is activated at the SAME round;
                    #   - both changes affect the data used by this model.
                    #
                    # Example:
                    #   combined_shift#0.1_1.0_sudden
                    # means alpha: 0.1 -> 1.0, with concept drift enabled
                    # simultaneously at the shift round.
                    # ---------------------------------------------------------
                    combined_transitions = {
                        "0.1_1.0": (0.1, 1.0),
                        "0.1-1.0": (0.1, 1.0),
                        "0.1_10.0": (0.1, 10.0),
                        "0.1-10.0": (0.1, 10.0),
                        "1.0_0.1": (1.0, 0.1),
                        "1.0-0.1": (1.0, 0.1),
                        "1.0_10.0": (1.0, 10.0),
                        "1.0-10.0": (1.0, 10.0),
                        "10.0_0.1": (10.0, 0.1),
                        "10.0-0.1": (10.0, 0.1),
                        "10.0_1.0": (10.0, 1.0),
                        "10.0-1.0": (10.0, 1.0),
                    }

                    transition_key = experiment_id.replace(
                        "combined_shift#", ""
                    ).replace("_sudden", "")

                    if transition_key not in combined_transitions:
                        config = {}
                    else:
                        initial_alpha, target_alpha = combined_transitions[
                            transition_key
                        ]

                        if not all(
                            abs(float(alpha) - initial_alpha) <= 1e-8
                            for alpha in self.alpha_train
                        ):
                            raise ValueError(
                                f"{experiment_id} requires initial alpha="
                                f"{initial_alpha}, but received "
                                f"{self.alpha_train}"
                            )

                        ME_concept_drift_rounds = [
                            [int(n_rounds * 0.3)],
                            [int(n_rounds * 0.5)],
                            [int(n_rounds * 0.7)]
                        ]

                        new_alphas = [
                            [target_alpha]
                            for _ in range(ME)
                        ]

                        new_concept_drift_window = [
                            [1]
                            for _ in range(ME)
                        ]

                        type_ = "combined_shift"

                        config = {
                            me: {
                                "data_shift_rounds": ME_concept_drift_rounds[me],
                                "new_alphas": new_alphas[me],
                                "new_concept_drift_window": (
                                    new_concept_drift_window[me]
                                ),
                                "type": type_
                            }
                            for me in range(ME)
                        }

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
                # Correct gradual drift: every client starts at the same
                # change point and the probability of drawing the new
                # concept increases continuously during transition_window.
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[0.1], [0.1], [0.1]]
                new_concept_drift_window = [[1], [1], [1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me],
                               "transition_window": self.label_shift_transition_window, "type": type_} for me in
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
                # Correct gradual drift: same change point for all clients;
                # only the fraction of new-concept samples changes over time.
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[10.0], [10.0], [10.0]]
                new_concept_drift_window = [[1], [1], [1]]
                type_ = "concept_drift"

                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me], "transition_window": self.label_shift_transition_window, "type": type_} for me in
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


            elif experiment_id == "concept_drift#1.0_gradual":
                ME_concept_drift_rounds = [[int(n_rounds * 0.3)],
                                           [int(n_rounds * 0.5)],
                                           [int(n_rounds * 0.7)]]
                new_alphas = [[1.0], [1.0], [1.0]]
                new_concept_drift_window = [[1], [1], [1]]
                type_ = "concept_drift"
                config = {me: {"data_shift_rounds": ME_concept_drift_rounds[me],
                               "new_alphas": new_alphas[me],
                               "new_concept_drift_window": new_concept_drift_window[me],
                               "transition_window": self.label_shift_transition_window,
                               "type": type_} for me in range(ME)}
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
            # IMPORTANT: combined_shift must be checked before label_shift
            # and concept_drift.  Its implementation is handled by
            # label_shift_config(), but it has its own type and must not be
            # routed to the label-only/concept-only configurations.
            if "combined_shift" in experiment_id:
                return self.label_shift_config(
                    ME,
                    n_rounds,
                    alphas,
                    experiment_id,
                    client_id,
                    gradual_rounds
                )
            elif "label_shift" in experiment_id:
                return self.label_shift_config(
                    ME,
                    n_rounds,
                    alphas,
                    experiment_id,
                    client_id,
                    gradual_rounds
                )
            elif "concept_drift" in experiment_id:
                return self.global_concept_drift_config(
                    ME,
                    n_rounds,
                    alphas,
                    experiment_id,
                    client_id,
                    gradual_rounds
                )
            else:
                return {}

        except Exception as e:
            print("get_data_shift_config error")
            print("""Error on line {} {} {}""".format(
                sys.exc_info()[-1].tb_lineno,
                type(e).__name__,
                e
            ))
            return {}

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

            g = torch.Generator()
            g.manual_seed(t+self.fold_id)
            random.seed(t+self.fold_id)
            np.random.seed(t+self.fold_id)
            torch.manual_seed(t+self.fold_id)
            set_weights(self.model[me], global_model)

            # Update alpha to simulate data shift
            if t > 1:
                self.update_local_train_data(t, me)
            self.lt[me] = t
            self.optimizer[me] = self._get_optimizer(dataset_name=self.args.dataset[me], me=me)

            print(
                f"[TRAIN DEBUG] "
                f"client={self.client_id} "
                f"model={me} "
                f"dataset={self.args.dataset[me]} "
                f"n_classes={self.n_classes[me]}"
            )
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
                                 self.args.dataset[me], self.n_classes[me])
            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            metrics["Alpha"] = self.alpha_test[me]
            tuple_me = (loss, len(self.valloader[me].dataset), metrics)
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_label_shift_source_loaders(self, me, base_alpha, target_alpha, partition_seed):
        """Load and cache source loaders used by gradual label shift."""
        key = (float(base_alpha), float(target_alpha), int(partition_seed))
        cache = self.label_shift_data_cache[me]

        if key not in cache:
            old_loader, old_val = load_data(
                dataset_name=self.args.dataset[me],
                alpha=float(base_alpha),
                data_sampling_percentage=self.args.data_percentage,
                partition_id=self.client_id,
                num_partitions=self.args.total_clients + 1,
                batch_size=self.batch_size[me],
                fold_id=self.fold_id,
                partition_seed=int(partition_seed),
            )
            target_loader, target_val = load_data(
                dataset_name=self.args.dataset[me],
                alpha=float(target_alpha),
                data_sampling_percentage=self.args.data_percentage,
                partition_id=self.client_id,
                num_partitions=self.args.total_clients + 1,
                batch_size=self.batch_size[me],
                fold_id=self.fold_id,
                partition_seed=int(partition_seed),
            )
            cache[key] = {
                "old_loader": old_loader,
                "old_val": old_val,
                "target_loader": target_loader,
                "target_val": target_val,
            }

        return cache[key]

    def _set_cached_label_shift_metrics(self, me, alpha, partition_seed, loader=None):
        """Get label-distribution metrics from a cached endpoint.

        The endpoint is scanned at most once for each (alpha, partition_seed)
        pair. Subsequent rounds reuse the cached P(Y), FC and IL.
        """
        key = (float(alpha), int(partition_seed))
        cache = self.label_shift_metrics_cache[me]

        if key not in cache:
            if loader is None:
                loader, _ = load_data(
                    dataset_name=self.args.dataset[me],
                    alpha=float(alpha),
                    data_sampling_percentage=self.args.data_percentage,
                    partition_id=self.client_id,
                    num_partitions=self.args.total_clients + 1,
                    batch_size=self.batch_size[me],
                    fold_id=self.fold_id,
                    partition_seed=int(partition_seed),
                )

            metric_loaders = [None] * self.ME
            metric_loaders[me] = loader
            p, fc, il = self._get_datasets_metrics(
                metric_loaders, self.ME, self.client_id, self.n_classes, me=me
            )
            cache[key] = (
                np.asarray(p, dtype=float).copy(),
                float(fc),
                float(il),
            )

        p, fc, il = cache[key]
        self.p_ME[me] = p.copy()
        self.fc_ME[me] = fc
        self.il_ME[me] = il
        return p, fc, il

    def _update_gradual_label_shift_metrics(
            self, me, base_alpha, target_alpha, partition_seed, progress
    ):
        """Update P(Y), FC and IL for a gradual label/combined shift.

        Endpoint distributions are computed once and then interpolated. This
        avoids iterating through the complete mixed training DataLoader at
        every transition round.
        """
        progress = float(np.clip(progress, 0.0, 1.0))

        old_key = (float(base_alpha), int(partition_seed))
        new_key = (float(target_alpha), int(partition_seed))
        cache = self.label_shift_metrics_cache[me]

        # The initial metrics were already computed during client creation.
        initial_key = (float(self.initial_alpha[me]), 1)
        if old_key not in cache and old_key == initial_key:
            cache[old_key] = (
                np.asarray(self.p_ME[me], dtype=float).copy(),
                float(self.fc_ME[me]),
                float(self.il_ME[me]),
            )

        # Endpoint data are loaded by _get_label_shift_source_loaders().
        cached_loaders = self._get_label_shift_source_loaders(
            me, base_alpha, target_alpha, partition_seed
        )

        if old_key not in cache:
            self._set_cached_label_shift_metrics(
                me, base_alpha, partition_seed,
                loader=cached_loaders["old_loader"]
            )

        if new_key not in cache:
            self._set_cached_label_shift_metrics(
                me, target_alpha, partition_seed,
                loader=cached_loaders["target_loader"]
            )

        p_old, fc_old, il_old = cache[old_key]
        p_new, fc_new, il_new = cache[new_key]

        if progress <= 0.0:
            p, fc, il = p_old, fc_old, il_old
        elif progress >= 1.0:
            p, fc, il = p_new, fc_new, il_new
        else:
            p = (1.0 - progress) * p_old + progress * p_new

            # FC and IL are defined from the resulting class distribution.
            # For a mixture, p_i > 0 is equivalent to count_i > 0, and
            # p_i < 1/C is equivalent to count_i < N/C.
            n_classes = int(self.n_classes[me])
            fc = float(np.count_nonzero(p > 0.0) / n_classes)
            il = float(np.count_nonzero(p < (1.0 / n_classes)) / n_classes)

        self.p_ME[me] = np.asarray(p, dtype=float).copy()
        self.fc_ME[me] = float(fc)
        self.il_ME[me] = float(il)

        return self.p_ME[me], self.fc_ME[me], self.il_ME[me]

    def _mix_label_shift_loaders(self, old_loader, new_loader, progress, shuffle=True):
        """Create a lazy gradual label-shift mixture without materialization.

        The old implementation materialized and deep-copied every sample from
        both loaders on every call.  Here we retain the same sampling rule but
        store only a deterministic mask and target indices.  Samples are read
        from the source datasets only when requested by the DataLoader.
        """
        progress = float(np.clip(progress, 0.0, 1.0))
        if progress <= 0.0:
            return old_loader
        if progress >= 1.0:
            return new_loader

        old_dataset = old_loader.dataset
        new_dataset = new_loader.dataset
        old_size = len(old_dataset)
        new_size = len(new_dataset)

        if old_size == 0 or new_size == 0:
            return old_loader if progress < 0.5 else new_loader

        rng = np.random.RandomState(
            17011 + int(self.client_id) * 100003 + int(self.fold_id) * 1009
        )
        use_new = rng.rand(old_size) < progress
        selected_positions = np.flatnonzero(use_new)
        new_indices = np.full(old_size, -1, dtype=np.int64)
        if len(selected_positions) > 0:
            new_indices[selected_positions] = rng.randint(
                0, new_size, size=len(selected_positions)
            )

        class MixedLabelShiftDataset(torch.utils.data.Dataset):
            def __init__(self, old_dataset, new_dataset, use_new, new_indices):
                self.old_dataset = old_dataset
                self.new_dataset = new_dataset
                self.use_new = use_new
                self.new_indices = new_indices

            def __len__(self):
                return len(self.old_dataset)

            def __getitem__(self, index):
                if self.use_new[index]:
                    return self.new_dataset[int(self.new_indices[index])]
                return self.old_dataset[index]

        mixed_dataset = MixedLabelShiftDataset(
            old_dataset, new_dataset, use_new, new_indices
        )

        kwargs = {
            "batch_size": old_loader.batch_size,
            "shuffle": shuffle,
            "num_workers": old_loader.num_workers,
            "drop_last": old_loader.drop_last,
            "pin_memory": old_loader.pin_memory,
        }
        if getattr(old_loader, "collate_fn", None) is not None:
            kwargs["collate_fn"] = old_loader.collate_fn
        if getattr(old_loader, "persistent_workers", False) and old_loader.num_workers > 0:
            kwargs["persistent_workers"] = True
        if getattr(old_loader, "prefetch_factor", None) is not None and old_loader.num_workers > 0:
            kwargs["prefetch_factor"] = old_loader.prefetch_factor

        return torch.utils.data.DataLoader(mixed_dataset, **kwargs)

    def _get_label_shift_state(self, server_round, me, train):
        """Return (base_alpha, target_alpha, progress, active, gradual)."""
        if self.data_shift_config == {}:
            a = self.initial_alpha[me]
            return a, a, 0.0, False, False
        config = self.data_shift_config[me]
        if config.get("type") not in ("label_shift", "combined_shift"):
            a = self.initial_alpha[me]
            return a, a, 0.0, False, False
        rounds = config.get("data_shift_rounds", [])
        targets = config.get("new_alphas", [])
        gradual = "gradual" in self.experiment_id
        base = self.initial_alpha[me]
        target = base
        progress = 0.0
        active = False
        for i, start in enumerate(rounds):
            if server_round < start:
                break
            base = target
            target = float(targets[i])
            active = True
            if gradual:
                # transition_window=0 is an explicit alias for sudden:
                # the target distribution must be active immediately at
                # the change point, exactly as in the sudden experiment.
                w = int(config.get("transition_window", self.label_shift_transition_window))
                if w <= 0:
                    progress = 1.0
                else:
                    progress = float(np.clip((server_round - start) / float(w), 0.0, 1.0))
            else:
                progress = 1.0
        if not active:
            base = self.initial_alpha[me]
            target = base
            progress = 0.0
        state = self.label_shift_progress_train if train else self.label_shift_progress_test
        changed = abs(progress - state[me]) > 1e-12
        state[me] = progress
        return base, target, progress, changed, gradual

    def _get_gradual_concept_drift_classes(self, me, progress):
        """Return the globally affected class IDs for gradual concept drift.

        ``progress`` is the temporal transition progress in [0, 1].  The
        number of affected classes is computed from the GLOBAL class set, not
        from each client's local class coverage.  Thus all clients observe the
        same class-level drift schedule.
        """
        n_classes = int(self.n_classes[me])
        progress = float(np.clip(progress, 0.0, 1.0))
        if n_classes <= 0 or progress <= 0.0:
            return set()
        if progress >= 1.0:
            n_affected = n_classes
        else:
            # Nearest integer gives the closest realizable global class
            # fraction (e.g., 20% of 10 classes = 2 classes).
            n_affected = int(np.floor(progress * n_classes + 0.5))
        n_affected = max(0, min(n_classes, n_affected))
        if n_affected == 0:
            return set()

        # Deterministic GLOBAL ordering: deliberately independent of client_id.
        seed = 42 + int(me) * 1009 + int(self.fold_id) * 9176 + 1543
        rng = np.random.RandomState(seed)
        class_order = np.arange(n_classes, dtype=np.int64)
        rng.shuffle(class_order)
        return {int(c) for c in class_order[:n_affected]}

    def _apply_concept_drift_to_loader(
            self,
            loader,
            me,
            concept_drift_fraction,
            shuffle=False,
            shift_context="concept_drift",
            affected_classes=None
    ):
        """Apply concept drift while preserving P(Y).

        For gradual concept drift, ``affected_classes`` contains the GLOBAL
        class IDs that have already transitioned to the new concept. Every
        local sample belonging to an affected class is transformed completely;
        unaffected classes remain unchanged. Labels are never modified, so P(Y)
        is preserved.

        ``concept_drift_fraction`` is retained for sudden/recurrent callers.
        When ``affected_classes`` is provided, the fraction is ignored and the
        transformation is performed at class level.
        """
        try:
            if loader is None or concept_drift_fraction is None:
                return loader

            fraction = float(np.clip(concept_drift_fraction, 0.0, 1.0))
            if affected_classes is None and fraction <= 0.0:
                return loader

            if affected_classes is not None:
                affected_classes = {int(c) for c in affected_classes}
                if not affected_classes:
                    return loader

            dataset_name = self.args.dataset[me]
            input_key = DATASET_INPUT_MAP.get(dataset_name)
            if input_key is None:
                raise ValueError(f"Unknown input key for dataset {dataset_name}")

            samples = []
            for batch in loader:
                if not isinstance(batch, dict):
                    raise TypeError("Expected DataLoader batches to be dictionaries.")
                if "label" not in batch:
                    raise KeyError(f"'label' not found in batch for dataset={dataset_name}")
                if input_key not in batch:
                    raise KeyError(f"Input key '{input_key}' not found in batch for dataset={dataset_name}")

                batch_size = batch["label"].shape[0]
                for i in range(batch_size):
                    sample = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            sample[key] = value[i].detach().cpu().clone()
                        elif hasattr(value, "__getitem__"):
                            try:
                                sample[key] = copy.deepcopy(value[i])
                            except Exception:
                                sample[key] = copy.deepcopy(value)
                        else:
                            sample[key] = copy.deepcopy(value)
                    samples.append(sample)

            if not samples:
                return loader

            labels = np.asarray([
                int(sample["label"].item())
                if isinstance(sample["label"], torch.Tensor)
                else int(sample["label"])
                for sample in samples
            ], dtype=np.int64)

            n_classes = int(self.n_classes[me])
            invalid_labels = sorted(set(
                int(label) for label in labels
                if label < 0 or label >= n_classes
            ))
            if invalid_labels:
                raise ValueError(
                    f"Invalid labels during concept drift: client={self.client_id}, "
                    f"model={me}, dataset={dataset_name}, n_classes={n_classes}, "
                    f"invalid={invalid_labels}"
                )

            class_indices = {
                class_id: np.where(labels == class_id)[0].tolist()
                for class_id in range(n_classes)
            }
            present_classes = [c for c in range(n_classes) if class_indices[c]]
            if len(present_classes) < 2:
                return loader

            # Stable randomness: the same class-level transformation is used
            # across clients and across rounds.  This is important for gradual
            # drift: a class that has transitioned must stay transitioned, while
            # newly affected classes are added to the global set.
            seed = (
                42
                + int(me) * 1009
                + int(self.fold_id) * 9176
                + 7919
            )
            rng = np.random.RandomState(seed)

            if affected_classes is not None:
                # ---------------------------------------------------------
                # GLOBAL CLASS-LEVEL GRADUAL CONCEPT DRIFT
                # ---------------------------------------------------------
                # All samples of every affected class are transformed.
                # The same global class IDs are therefore used by every client;
                # clients simply ignore classes that are absent locally.
                target_by_class = {
                    c: list(class_indices[c])
                    for c in present_classes
                    if c in affected_classes
                }
                if not target_by_class:
                    return loader

                source_classes = present_classes.copy()
                rng.shuffle(source_classes)

                # Each affected target class receives X from another class.
                # A cyclic permutation guarantees source_class != target_class
                # whenever at least two local classes are present. Sampling with
                # replacement is allowed because class sizes can differ.
                if len(source_classes) < 2:
                    return loader

                source_for_target = {
                    source_classes[i]: source_classes[(i + 1) % len(source_classes)]
                    for i in range(len(source_classes))
                }

                assignments = []
                for target_class, target_indices in target_by_class.items():
                    source_class = source_for_target[target_class]
                    source_indices = np.asarray(class_indices[source_class], dtype=np.int64)
                    if len(source_indices) == 0:
                        continue
                    sampled_sources = rng.choice(
                        source_indices, size=len(target_indices), replace=True
                    )
                    assignments.extend(
                        (int(target_idx), int(source_idx), target_class, source_class)
                        for target_idx, source_idx in zip(target_indices, sampled_sources)
                    )
            else:
                # ---------------------------------------------------------
                # EXISTING SUDDEN/RECURRENT SAMPLE-LEVEL BEHAVIOR
                # ---------------------------------------------------------
                selected_by_class = {}
                selected_targets = []
                for class_id in present_classes:
                    indices = np.asarray(class_indices[class_id], dtype=np.int64)
                    n_selected = int(round(fraction * len(indices)))
                    if fraction > 0.0 and n_selected == 0 and len(indices) > 0:
                        n_selected = 1
                    n_selected = min(n_selected, len(indices))
                    if n_selected > 0:
                        chosen = rng.choice(indices, size=n_selected, replace=False)
                        chosen = np.asarray(chosen, dtype=np.int64)
                        selected_by_class[class_id] = chosen.tolist()
                        selected_targets.extend((int(idx), class_id) for idx in chosen)
                    else:
                        selected_by_class[class_id] = []

                if not selected_targets:
                    return loader

                source_by_class = {
                    class_id: selected_by_class[class_id].copy()
                    for class_id in present_classes
                }
                for class_id in present_classes:
                    rng.shuffle(source_by_class[class_id])

                class_order = present_classes.copy()
                rng.shuffle(class_order)
                if len(class_order) > 1:
                    target_to_source_class = {
                        class_order[i]: class_order[(i + 1) % len(class_order)]
                        for i in range(len(class_order))
                    }
                else:
                    return loader

                available_sources = {
                    c: list(source_by_class[c]) for c in present_classes
                }
                for c in present_classes:
                    rng.shuffle(available_sources[c])

                assignments = []
                selected_targets_by_class = {
                    c: list(selected_by_class[c]) for c in present_classes
                }
                for c in present_classes:
                    rng.shuffle(selected_targets_by_class[c])

                pending = []
                for target_class in present_classes:
                    source_class = target_to_source_class[target_class]
                    targets = selected_targets_by_class[target_class]
                    sources = available_sources[source_class]
                    n = min(len(targets), len(sources))
                    for target_idx, source_idx in zip(targets[:n], sources[:n]):
                        assignments.append((target_idx, source_idx, target_class, source_class))
                    del targets[:n]
                    del sources[:n]
                    pending.extend((target_idx, target_class) for target_idx in targets)

                used_sources = {source_idx for _, source_idx, _, _ in assignments}
                for target_idx, target_class in pending:
                    candidates = [
                        idx for idx in range(len(samples))
                        if idx not in used_sources and int(labels[idx]) != target_class
                    ]
                    if not candidates:
                        continue
                    source_idx = int(candidates[rng.randint(len(candidates))])
                    source_class = int(labels[source_idx])
                    assignments.append((target_idx, source_idx, target_class, source_class))
                    used_sources.add(source_idx)

            shifted_samples = [copy.deepcopy(sample) for sample in samples]
            for target_idx, source_idx, _, _ in assignments:
                shifted_samples[target_idx][input_key] = copy.deepcopy(
                    samples[source_idx][input_key]
                )

            shifted_labels = np.asarray([
                int(sample["label"].item())
                if isinstance(sample["label"], torch.Tensor)
                else int(sample["label"])
                for sample in shifted_samples
            ], dtype=np.int64)

            if not np.array_equal(labels, shifted_labels):
                raise RuntimeError("Concept drift changed labels. This violates P(Y) preservation.")

            original_classes, original_counts = np.unique(labels, return_counts=True)
            shifted_classes, shifted_counts = np.unique(shifted_labels, return_counts=True)
            if not np.array_equal(original_classes, shifted_classes) or not np.array_equal(original_counts, shifted_counts):
                raise RuntimeError("Concept drift changed class frequencies. P(Y) was not preserved.")

            changed_x = 0
            total_abs_change = 0.0
            total_elements = 0
            for original_sample, shifted_sample in zip(samples, shifted_samples):
                original_x = original_sample[input_key]
                shifted_x = shifted_sample[input_key]
                if isinstance(original_x, torch.Tensor) and isinstance(shifted_x, torch.Tensor):
                    if not torch.equal(original_x, shifted_x):
                        changed_x += 1
                    if original_x.shape == shifted_x.shape:
                        diff = torch.abs(
                            original_x.detach().cpu().float() - shifted_x.detach().cpu().float()
                        )
                        total_abs_change += float(diff.sum().item())
                        total_elements += int(diff.numel())

            conditional_mean_changes = []
            classes_with_conditional_change = 0
            for class_id in present_classes:
                idxs = np.where(labels == class_id)[0]
                original_tensors = [samples[i][input_key] for i in idxs]
                shifted_tensors = [shifted_samples[i][input_key] for i in idxs]
                if all(isinstance(x, torch.Tensor) for x in original_tensors + shifted_tensors):
                    original_stack = torch.stack([x.detach().cpu().float() for x in original_tensors])
                    shifted_stack = torch.stack([x.detach().cpu().float() for x in shifted_tensors])
                    change = float(torch.abs(original_stack.mean(0) - shifted_stack.mean(0)).mean().item())
                    conditional_mean_changes.append(change)
                    if change > 1e-12:
                        classes_with_conditional_change += 1

            mean_conditional_change = float(np.mean(conditional_mean_changes)) if conditional_mean_changes else 0.0
            changed_fraction = len(assignments) / max(len(samples), 1)
            empirical_x_changed_fraction = changed_x / max(len(samples), 1)
            mean_abs_x_change = total_abs_change / max(total_elements, 1)
            conditional_change_fraction = classes_with_conditional_change / max(len(present_classes), 1)
            concept_drift_confirmed = bool(
                assignments and changed_x > 0 and
                np.array_equal(labels, shifted_labels) and
                np.array_equal(original_counts, shifted_counts) and
                mean_conditional_change > 1e-12
            )

            print(
                f"[CONCEPT DRIFT VERIFY] shift_type={shift_context} "
                f"client={self.client_id} model={me} dataset={dataset_name} "
                f"fraction={fraction:.4f} X_changed={changed_x}/{len(samples)} "
                f"X_changed_fraction={empirical_x_changed_fraction:.4f} "
                f"mean_abs_X_change={mean_abs_x_change:.8f} "
                f"P(X|Y)_mean_change={mean_conditional_change:.8f} "
                f"P(X|Y)_change_fraction={conditional_change_fraction:.4f} "
                f"concept_drift_confirmed={concept_drift_confirmed}"
            )

            class ConceptDriftDataset(torch.utils.data.Dataset):
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, index):
                    return self.data[index]

            shifted_dataset = ConceptDriftDataset(shifted_samples)
            loader_kwargs = {
                "batch_size": loader.batch_size,
                "shuffle": shuffle,
                "num_workers": loader.num_workers,
                "drop_last": loader.drop_last,
                "pin_memory": loader.pin_memory,
            }
            if getattr(loader, "collate_fn", None) is not None:
                loader_kwargs["collate_fn"] = loader.collate_fn
            if getattr(loader, "persistent_workers", False) and loader.num_workers > 0:
                loader_kwargs["persistent_workers"] = True
            if getattr(loader, "prefetch_factor", None) is not None and loader.num_workers > 0:
                loader_kwargs["prefetch_factor"] = loader.prefetch_factor

            return torch.utils.data.DataLoader(shifted_dataset, **loader_kwargs)

        except Exception as e:
            print("_apply_concept_drift_to_loader error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))
            return loader

    def update_local_train_data(
            self,
            t,
            me
    ):
        try:

            # =========================================================
            # INITIALIZATION
            # =========================================================

            if t == 1:
                self.trainloader[me], self.valloader[me] = load_data(
                    dataset_name=self.args.dataset[me],
                    alpha=self.alpha_train[me],
                    data_sampling_percentage=self.args.data_percentage,
                    partition_id=self.client_id,
                    num_partitions=self.args.total_clients + 1,
                    batch_size=self.batch_size[me],
                    fold_id=self.fold_id,
                    partition_seed=self.partition_seed_train[me],
                )

                # Keep the original, unshifted training dataset.
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

                return

            # =========================================================
            # DATA SHIFT
            # =========================================================

            if self.data_shift_config != {}:

                (
                    alpha_me,
                    partition_seed,
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
                    f"alpha atual "
                    f"{self.alpha_train[me]} "
                    f"novo {alpha_me} - "
                    f"concept drift atual "
                    f"{self.concept_drift_window_train[me]} "
                    f"novo {concept_drift_window}"
                )

                # =====================================================
                # TRUE GRADUAL LABEL / COMBINED SHIFT
                # =====================================================
                if data_shift_flag and "gradual" in self.experiment_id and self.data_shift_config[me]["type"] in ("label_shift", "combined_shift"):
                    base_alpha, target_alpha, label_progress, _, _ = self._get_label_shift_state(t, me, True)
                    cached = self._get_label_shift_source_loaders(
                        me, base_alpha, target_alpha, partition_seed
                    )
                    old_loader = cached["old_loader"]
                    old_val = cached["old_val"]
                    target_loader = cached["target_loader"]
                    target_val = cached["target_val"]
                    mixed_loader = self._mix_label_shift_loaders(
                        old_loader, target_loader, label_progress, shuffle=True
                    )
                    if self.data_shift_config[me]["type"] == "combined_shift" and concept_drift_window > 0:
                        mixed_loader = self._apply_concept_drift_to_loader(
                            mixed_loader, me, concept_drift_window, shuffle=True, shift_context="combined_shift"
                        )
                    self.trainloader[me] = mixed_loader
                    self.valloader[me] = target_val if label_progress >= 1.0 else old_val
                    self.recent_trainloader[me] = old_loader
                    self.alpha_train[me] = target_alpha
                    self.alpha_test[me] = target_alpha
                    self.partition_seed_train[me] = int(partition_seed)
                    self.partition_seed_test[me] = int(partition_seed)
                    if self.data_shift_config[me]["type"] == "combined_shift":
                        self.concept_drift_window_train[me] = concept_drift_window
                    print(
                        f"[GRADUAL DATA SHIFT - TRAIN] client={self.client_id} model={me} "
                        f"round={t} label_progress={label_progress:.4f} "
                        f"concept_progress={float(concept_drift_window):.4f}"
                    )
                    # P(Y), FC and IL are obtained from the endpoint
                    # distributions. Do NOT scan the mixed DataLoader here:
                    # this branch is executed on every selected client during
                    # the transition window.
                    self._update_gradual_label_shift_metrics(
                        me, base_alpha, target_alpha, partition_seed,
                        label_progress
                    )
                    self.num_examples[me] = len(self.trainloader[me].dataset)
                    return

                # =====================================================
                # LABEL SHIFT
                # =====================================================

                if (
                        data_shift_flag
                        and self.data_shift_config[me]["type"]
                        == "label_shift"
                ):

                    if (
                            self.alpha_train[me] != self.alpha_test[me]
                            and self.alpha_test[me] == alpha_me
                            and self.partition_seed_train[me] == partition_seed
                    ):

                        self.alpha_train[me] = (
                            self.alpha_test[me]
                        )

                        self.trainloader[me] = (
                            copy.deepcopy(
                                self.recent_trainloader[me]
                            )
                        )

                        self._set_cached_label_shift_metrics(
                            me,
                            self.alpha_train[me],
                            self.partition_seed_train[me],
                            loader=self.trainloader[me]
                        )

                    else:

                        self.alpha_train[me] = (
                            alpha_me
                        )

                        self.alpha_test[me] = (
                            alpha_me
                        )

                        self.partition_seed_train[me] = int(partition_seed)
                        self.partition_seed_test[me] = int(partition_seed)

                        (
                            self.trainloader[me],
                            self.valloader[me]
                        ) = load_data(
                            dataset_name=self.args.dataset[me],
                            alpha=self.alpha_train[me],
                            data_sampling_percentage=self.args.data_percentage,
                            partition_id=self.client_id,
                            num_partitions=self.args.total_clients + 1,
                            batch_size=self.batch_size[me],
                            fold_id=self.fold_id,
                            partition_seed=self.partition_seed_train[me] if t == 1 else partition_seed,
                        )

                        self.recent_trainloader[me] = (
                            copy.deepcopy(
                                self.trainloader[me]
                            )
                        )

                        self._set_cached_label_shift_metrics(
                            me,
                            self.alpha_train[me],
                            self.partition_seed_train[me],
                            loader=self.trainloader[me]
                        )

                # =====================================================
                # COMBINED SHIFT
                # =====================================================

                elif (
                        data_shift_flag
                        and self.data_shift_config[me]["type"]
                        == "combined_shift"
                        and t - self.lt[me] > 0
                ):

                    print(
                        f"[COMBINED SHIFT - TRAIN] "
                        f"client={self.client_id} "
                        f"model={me} "
                        f"round={t}: "
                        f"alpha {self.alpha_train[me]} -> {alpha_me}; "
                        f"concept drift window "
                        f"{self.concept_drift_window_train[me]} -> "
                        f"{concept_drift_window}"
                    )

                    # Combined shift changes P(Y) through the new
                    # Dirichlet alpha and P(X|Y) through concept drift.
                    self.alpha_train[me] = alpha_me
                    self.alpha_test[me] = alpha_me
                    self.partition_seed_train[me] = int(partition_seed)
                    self.partition_seed_test[me] = int(partition_seed)
                    self.concept_drift_window_train[me] = (
                        concept_drift_window
                    )

                    (
                        self.trainloader[me],
                        self.valloader[me]
                    ) = load_data(
                        dataset_name=self.args.dataset[me],
                        alpha=self.alpha_train[me],
                        data_sampling_percentage=self.args.data_percentage,
                        partition_id=self.client_id,
                        num_partitions=self.args.total_clients + 1,
                        batch_size=self.batch_size[me],
                        fold_id=self.fold_id,
                        partition_seed=partition_seed,
                    )

                    if concept_drift_window > 0:
                        self.trainloader[me] = (
                            self._apply_concept_drift_to_loader(
                                self.trainloader[me],
                                me,
                                concept_drift_window,
                                shuffle=True,
                                shift_context="combined_shift"
                            )
                        )

                    # Concept drift preserves P(Y); only the label-shift
                    # component can change the dataset metrics.
                    if self.data_shift_config[me].get("type") == "combined_shift":
                        self._set_cached_label_shift_metrics(
                            me,
                            self.alpha_train[me],
                            self.partition_seed_train[me],
                            loader=self.trainloader[me]
                        )

                # =====================================================
                # CONCEPT DRIFT
                # =====================================================

                elif (
                        data_shift_flag
                        and self.data_shift_config[me]["type"]
                        == "concept_drift" and t - self.lt[me] > 0
                ):

                    print(
                        f"[CONCEPT DRIFT - TRAIN] "
                        f"client={self.client_id} "
                        f"model={me} "
                        f"round={t} "
                        f"window="
                        f"{self.concept_drift_window_train[me]}"
                        f" -> "
                        f"{concept_drift_window}"
                    )

                    # Concept drift does NOT change alpha.
                    self.alpha_train[me] = (
                        alpha_me
                    )

                    self.alpha_test[me] = (
                        alpha_me
                    )

                    # -------------------------------------------------
                    # IMPORTANT:
                    #
                    # Always start from the ORIGINAL training data.
                    #
                    # This prevents cumulative transformations and
                    # correctly supports recurrent drift:
                    #
                    #     0 -> 1 -> 0 -> 1
                    #
                    # -------------------------------------------------

                    self.concept_drift_window_train[me] = (
                        concept_drift_window
                    )

                    if concept_drift_window == 0:

                        (
                            self.trainloader[me],
                            self.valloader[me]
                        ) = load_data(
                            dataset_name=self.args.dataset[me],
                            alpha=self.alpha_train[me],
                            data_sampling_percentage=self.args.data_percentage,
                            partition_id=self.client_id,
                            num_partitions=self.args.total_clients + 1,
                            batch_size=self.batch_size[me],
                            fold_id=self.fold_id,
                            partition_seed=self.partition_seed_train[me] if t == 1 else partition_seed,
                        )

                        print(
                            f"[CONCEPT DRIFT - TRAIN] "
                            f"client={self.client_id} "
                            f"model={me} "
                            f"drift removed; "
                            f"original training data restored"
                        )

                        # Concept drift changes P(X|Y), not P(Y), so the
                        # dataset metrics remain unchanged.

                    else:

                        (
                            self.trainloader[me],
                            self.valloader[me]
                        ) = load_data(
                            dataset_name=self.args.dataset[me],
                            alpha=self.alpha_train[me],
                            data_sampling_percentage=self.args.data_percentage,
                            partition_id=self.client_id,
                            num_partitions=self.args.total_clients + 1,
                            batch_size=self.batch_size[me],
                            fold_id=self.fold_id,
                            partition_seed=self.partition_seed_train[me] if t == 1 else partition_seed,
                        )
                        if "gradual" in self.experiment_id:
                            affected_classes = self._get_gradual_concept_drift_classes(
                                me, concept_drift_window
                            )
                            self.trainloader[me] = self._apply_concept_drift_to_loader(
                                self.trainloader[me],
                                me,
                                concept_drift_window,
                                shuffle=True,
                                affected_classes=affected_classes
                            )
                        else:
                            self.trainloader[me] = (
                                self._apply_concept_drift_to_loader(
                                    self.trainloader[me],
                                    me,
                                    concept_drift_window,
                                    shuffle=True
                                )
                            )

                    # Concept drift preserves the label distribution.
                    # No full-dataset metrics scan is necessary here.

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

    def update_local_test_data(
            self,
            t,
            me
    ):
        try:

            # =========================================================
            # NO DATA SHIFT CONFIGURATION
            # =========================================================

            if self.data_shift_config == {}:
                return (
                    self.p_ME,
                    self.fc_ME,
                    self.il_ME
                )

            (
                alpha_me,
                partition_seed,
                concept_drift_window,
                data_shift_flag
            ) = self._data_shift_flag(
                t,
                me,
                train=False
            )

            # =========================================================
            # TRUE GRADUAL LABEL / COMBINED SHIFT - TEST
            # =========================================================
            if data_shift_flag and "gradual" in self.experiment_id and self.data_shift_config[me]["type"] in ("label_shift", "combined_shift"):
                base_alpha, target_alpha, label_progress, _, _ = self._get_label_shift_state(t, me, False)
                cached = self._get_label_shift_source_loaders(
                    me, base_alpha, target_alpha, partition_seed
                )
                old_loader = cached["old_loader"]
                target_loader = cached["target_loader"]
                mixed_loader = self._mix_label_shift_loaders(
                    old_loader, target_loader, label_progress, shuffle=False
                )
                if self.data_shift_config[me]["type"] == "combined_shift" and concept_drift_window > 0:
                    mixed_loader = self._apply_concept_drift_to_loader(
                        mixed_loader, me, concept_drift_window, shuffle=False, shift_context="combined_shift"
                    )
                self.valloader[me] = mixed_loader
                self.recent_trainloader[me] = old_loader
                self.alpha_test[me] = target_alpha
                self.partition_seed_test[me] = int(partition_seed)
                if self.data_shift_config[me]["type"] == "combined_shift":
                    self.concept_drift_window_test[me] = concept_drift_window
                print(
                    f"[GRADUAL DATA SHIFT - TEST] client={self.client_id} model={me} "
                    f"round={t} label_progress={label_progress:.4f} "
                    f"concept_progress={float(concept_drift_window):.4f}"
                )
                return self.p_ME, self.fc_ME, self.il_ME

            # =========================================================
            # LABEL SHIFT
            # =========================================================

            if (
                    data_shift_flag
                    and self.data_shift_config[me]["type"]
                    == "label_shift"
                    and (self.alpha_test[me] != alpha_me or self.partition_seed_test[me] != partition_seed)
            ):
                print(
                    f"[LABEL SHIFT - TEST] "
                    f"client={self.client_id} "
                    f"model={me} "
                    f"round={t}: "
                    f"alpha "
                    f"{self.alpha_test[me]} "
                    f"-> {alpha_me}"
                )

                self.alpha_test[me] = (
                    alpha_me
                )
                self.partition_seed_test[me] = int(partition_seed)

                (
                    self.recent_trainloader[me],
                    self.valloader[me]
                ) = load_data(
                    dataset_name=self.args.dataset[me],
                    alpha=self.alpha_test[me],
                    data_sampling_percentage=self.args.data_percentage,
                    partition_id=self.client_id,
                    num_partitions=self.args.total_clients + 1,
                    batch_size=self.batch_size[me],
                    fold_id=self.fold_id,
                    partition_seed=partition_seed,
                )

                return (
                    self.p_ME,
                    self.fc_ME,
                    self.il_ME
                )

            # =========================================================
            # COMBINED SHIFT
            # =========================================================

            if (
                    self.data_shift_config[me]["type"]
                    == "combined_shift"
                    and data_shift_flag
                    and (
                        self.alpha_test[me] != alpha_me
                        or self.partition_seed_test[me] != partition_seed
                        or self.concept_drift_window_test[me]
                        != concept_drift_window
                    )
            ):
                print(
                    f"[COMBINED SHIFT - TEST] "
                    f"client={self.client_id} "
                    f"model={me} "
                    f"round={t}: "
                    f"alpha {self.alpha_test[me]} -> {alpha_me}; "
                    f"concept drift window "
                    f"{self.concept_drift_window_test[me]} -> "
                    f"{concept_drift_window}"
                )

                self.alpha_test[me] = alpha_me
                self.partition_seed_test[me] = int(partition_seed)
                self.concept_drift_window_test[me] = (
                    concept_drift_window
                )

                _, original_valloader = load_data(
                    dataset_name=self.args.dataset[me],
                    alpha=self.alpha_test[me],
                    data_sampling_percentage=self.args.data_percentage,
                    partition_id=self.client_id,
                    num_partitions=self.args.total_clients + 1,
                    batch_size=self.batch_size[me],
                    fold_id=self.fold_id,
                    partition_seed=partition_seed,
                )

                if concept_drift_window <= 0:
                    self.valloader[me] = original_valloader
                else:
                    self.valloader[me] = (
                        self._apply_concept_drift_to_loader(
                            original_valloader,
                            me,
                            concept_drift_window,
                            shuffle=False,
                            shift_context="combined_shift"
                        )
                    )

                return (
                    self.p_ME,
                    self.fc_ME,
                    self.il_ME
                )

            # =========================================================
            # CONCEPT DRIFT
            # =========================================================

            if (
                    self.data_shift_config[me]["type"]
                    == "concept_drift"
                    and data_shift_flag and t - self.lt[me] > 0 and self.concept_drift_window_test[me] != concept_drift_window
            ):

                old_window = (
                    self.concept_drift_window_test[me]
                )

                print(
                    f"[CONCEPT DRIFT - TEST] "
                    f"client={self.client_id} "
                    f"model={me} "
                    f"round={t} "
                    f"window="
                    f"{old_window}"
                    f" -> "
                    f"{concept_drift_window}"
                )

                # -----------------------------------------------------
                # IMPORTANT:
                #
                # Test data changes for EVERY CLIENT.
                #
                # It does NOT depend on self.lt[me].
                # -----------------------------------------------------

                self.concept_drift_window_test[me] = (
                    concept_drift_window
                )

                # -----------------------------------------------------
                # Concept drift does not change alpha.
                # -----------------------------------------------------

                self.alpha_test[me] = (
                    alpha_me
                )

                # -----------------------------------------------------
                # Reload the ORIGINAL test dataset.
                #
                # This prevents cumulative transformations and allows:
                #
                #     0 -> 1 -> 0 -> 1
                #
                # to correctly represent the environment state.
                # -----------------------------------------------------

                _, original_valloader = load_data(
                    dataset_name=self.args.dataset[me],
                    alpha=self.alpha_test[me],
                    data_sampling_percentage=self.args.data_percentage,
                    partition_id=self.client_id,
                    num_partitions=self.args.total_clients + 1,
                    batch_size=self.batch_size[me],
                    fold_id=self.fold_id,
                    partition_seed=partition_seed,
                )

                # -----------------------------------------------------
                # Apply the CURRENT environment state.
                #
                # window = 0 -> original test data
                # window > 0 -> shifted test data
                # -----------------------------------------------------

                if concept_drift_window == 0:

                    self.valloader[me] = (
                        original_valloader
                    )

                    print(
                        f"[CONCEPT DRIFT - TEST] "
                        f"client={self.client_id} "
                        f"model={me} "
                        f"drift removed; "
                        f"original test data restored"
                    )

                else:

                    if "gradual" in self.experiment_id:
                        affected_classes = self._get_gradual_concept_drift_classes(
                            me, concept_drift_window
                        )
                        self.valloader[me] = self._apply_concept_drift_to_loader(
                            original_valloader,
                            me,
                            concept_drift_window,
                            shuffle=False,
                            affected_classes=affected_classes
                        )
                    else:
                        self.valloader[me] = (
                            self._apply_concept_drift_to_loader(
                                original_valloader,
                                me,
                                concept_drift_window,
                                shuffle=False
                            )
                        )

                return (
                    self.p_ME,
                    self.fc_ME,
                    self.il_ME
                )

            # =========================================================
            # NO NEW SHIFT
            # =========================================================

            return (
                self.p_ME,
                self.fc_ME,
                self.il_ME
            )

        except Exception as e:

            print(
                f"update_local_test_data error "
                f"{self.data_shift_config}"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            return (
                self.p_ME,
                self.fc_ME,
                self.il_ME
            )

    def _get_current_partition_seed(self, server_round, me, train=True):
        """Return the Dirichlet partition seed active at ``server_round``.

        For the seed-based label-shift experiments, alpha is kept fixed and
        only the partition seed changes.  Other shift configurations retain
        the original seed (1).
        """
        try:
            default_seed = 1

            if self.data_shift_config == {}:
                return default_seed, False

            config = self.data_shift_config[me]
            seeds = config.get("partition_seeds")
            shift_rounds = config.get("data_shift_rounds", [])

            if not seeds:
                return default_seed, False

            current_seed = default_seed
            for i, start_round in enumerate(shift_rounds):
                if server_round >= start_round:
                    current_seed = int(seeds[i])

            reference_seed = getattr(self, "partition_seed_train", [default_seed] * self.ME)[me] if train else getattr(self, "partition_seed_test", [default_seed] * self.ME)[me]
            return current_seed, current_seed != reference_seed

        except Exception as e:
            print(f"_get_current_partition_seed error {self.data_shift_config}")
            print("Error on line {} {} {}".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
            return 1, False

    def _get_current_alpha(self, server_round, me, train):
        """Return the fixed target alpha and the transition-change flag.

        For gradual label/combined shift, alpha itself is NOT interpolated.
        The old and target partitions are mixed probabilistically by
        ``_mix_label_shift_loaders``. This is a true gradual transition:
        P_t(Y) is a mixture of the old and target local label distributions.
        """
        try:
            reference = self.alpha_train[me] if train else self.alpha_test[me]
            base, target, progress, progress_changed, gradual = self._get_label_shift_state(
                server_round, me, train
            )
            if not self.data_shift_config or self.data_shift_config[me].get("type") not in ("label_shift", "combined_shift"):
                return reference, False
            if gradual:
                return target, progress_changed or abs(float(target) - float(reference)) > 1e-8
            changed = abs(float(target) - float(reference)) > 1e-8
            return target, changed
        except Exception as e:
            print(f"_get_current_alpha error {self.data_shift_config}")
            print("Error on line {} {} {}".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
            return self.alpha_train[me] if train else self.alpha_test[me], False

    def _data_shift_flag(self, server_round, me, train):

        try:
            alpha, label_shift_flag = self._get_current_alpha(server_round, me, train)
            partition_seed, seed_shift_flag = self._get_current_partition_seed(server_round, me, train)
            concept_drift_window, concept_drift_flag = self._check_concept_drift(server_round, me, train)
            return alpha, partition_seed, concept_drift_window, True in [label_shift_flag, seed_shift_flag, concept_drift_flag]

        except Exception as e:
            print(f"_data_shift_flag error {self.data_shift_config}")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _check_concept_drift(self, server_round, me, train):
        """Return the current concept-drift intensity in [0, 1].

        For gradual concept drift, transition_window is the number of
        federated rounds used for the transition.  The returned value is the
        fraction of local samples that receive the new concept.

        Thus, with W=5, the transition is 0.0, 0.2, 0.4, 0.6, 0.8, 1.0.
        W=0 is an explicit alias for sudden drift: intensity becomes 1.0 at
        the change point.

        Sudden/recurrent concept-drift configurations keep their previous
        behavior: an active marker produces intensity 1.0 and an inactive
        marker produces 0.0.
        """
        try:
            if (self.data_shift_config == {} or
                    self.data_shift_config[me]["type"] not in ("concept_drift", "combined_shift")):
                return 0.0, False

            reference = (
                self.concept_drift_window_train[me]
                if train else self.concept_drift_window_test[me]
            )
            config = self.data_shift_config[me]
            gradual = "gradual" in self.experiment_id
            current_intensity = 0.0

            for i, start_round in enumerate(config.get("data_shift_rounds", [])):
                if server_round < start_round:
                    break

                if gradual:
                    w = int(config.get("transition_window", self.label_shift_transition_window))
                    if w <= 0:
                        current_intensity = 1.0
                    else:
                        current_intensity = float(np.clip(
                            (server_round - start_round) / float(w), 0.0, 1.0
                        ))
                else:
                    marker_list = config.get("new_concept_drift_window", [1])
                    marker = int(marker_list[i]) if i < len(marker_list) else 1
                    current_intensity = 1.0 if marker > 0 else 0.0

            changed = abs(float(current_intensity) - float(reference)) > 1e-12
            return float(current_intensity), changed

        except Exception as e:
            print(f"_check_concept_drift error {self.data_shift_config}")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))
            return 0.0, False

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
                    'CINIC10': torch.optim.SGD(self.model[me].parameters(), self.lr_dict[dataset_name], momentum=0.9),
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