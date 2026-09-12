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

import os
import csv
import copy
import time
import numpy as np
from flcore.clients.client_multifedavg_with_multifedpredict import ClientMultiFedAvgWithMultiFedPredict
from flcore.clients.client_multifedavg_with_fedpredict import ClientMultiFedAvgWithFedPredict
from flcore.clients.client_multifedavg_with_fedpredict_dynamic import ClientMultiFedAvgWithFedPredictDynamic
from flcore.servers.server_multifedavg_with_multifedpredict_v0 import MultiFedAvgWithMultiFedPredictv0
import sys
from collections import Counter
from fedpredict import fedpredict_server, fedpredict_layerwise_similarity
import flwr
import math
from flwr.common import (
    EvaluateIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from functools import partial, reduce
from typing import Any, Callable, Union

import numpy as np

from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import torch
import random

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg


def get_weights(net):
    try:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    except Exception as e:
        print("get_weights error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def weighted_average_fit(metrics):
    try:
        # Multiply accuracy of each client by number of Papers examples used
        # print(f"metricas recebidas: {metrics}")
        accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
        balanced_accuracies = [num_examples * m["train_balanced_accuracy"] for num_examples, m in metrics]
        loss = [num_examples * m["train_loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        aggregated = {"Accuracy": sum(accuracies) / sum(examples),
                      "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
                      "Loss": sum(loss) / sum(examples),
                      "Round (t)": metrics[0][1]["Round (t)"],
                      "Model size": metrics[0][1]["Model size"]}

        # combined_train_accuracy is generated during client.fit(), so it
        # must be aggregated together with the other fit-time evidence.
        combined_values = [
            num_examples * float(m["combined_train_accuracy"])
            for num_examples, m in metrics
            if m.get("combined_train_accuracy") is not None
        ]
        combined_examples = [
            num_examples
            for num_examples, m in metrics
            if m.get("combined_train_accuracy") is not None
        ]
        if combined_examples and sum(combined_examples) > 0:
            aggregated["combined_train_accuracy"] = (
                sum(combined_values) / sum(combined_examples)
            )

        return aggregated
    except Exception as e:
        print("weighted_average_fit error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


class MultiFedAvgWithMultiFedPredict(MultiFedAvgWithMultiFedPredictv0):
    def __init__(self, args, times, version, fold_id):
        try:
            super().__init__(
                args,
                times,
                fold_id
            )

            self.t_hat = [1] * self.ME

            self.reduced_training_intensity_flag = [
                                                       False
                                                   ] * self.ME

            self.train_accuracy_list = {
                me: [] for me in range(self.ME)
            }
            # Aggregated local-model training accuracy indexed by round.
            # This is kept separately from train_accuracy_list so that
            # rounds in which a model was not trained are not populated
            # with the previous round's value in the CSV.
            self.train_accuracy_by_round = {
                me: {} for me in range(self.ME)
            }

            # Server-side history of the combined model's training accuracy.
            self.combined_train_accuracy = {me: 0.0 for me in range(self.ME)}
            self.combined_train_accuracy_history = {me: [] for me in range(self.ME)}
            self.combined_accuracy_reference = {me: 0.0 for me in range(self.ME)}
            self.combined_accuracy_drop = {me: 0.0 for me in range(self.ME)}
            self.combined_accuracy_drop_significant = {me: False for me in range(self.ME)}

            # FedPredict weights returned by client.evaluate().
            # These must come exclusively from evaluate(), never from fit().
            self.gw = {me: [] for me in range(self.ME)}
            self.lw = {me: [] for me in range(self.ME)}
            self.gw_by_round = {me: {} for me in range(self.ME)}
            self.lw_by_round = {me: {} for me in range(self.ME)}

            self.combined_accuracy_history_window = 10
            self.combined_accuracy_drop_threshold = 0.15

            self.max_number_of_rounds_data_drift_adaptation = (
                    len(self.clients)
                    // self.num_training_clients
            )

            self.increased_training_intensity = [
                                                    0
                                                ] * self.ME

            self.reduced_training_intensity_flag = [
                                                       False
                                                   ] * self.ME

            self.last_round_increased_training_intensity = [
                                                               0
                                                           ] * self.ME

            self.version = version

            self.train_losses = {
                me: [] for me in range(self.ME)
            }

            self.fit_metrics_aggregation_fn = (
                weighted_average_fit
            )

            self.data_drift_model = -1

            self.reduction_fraction_list = {
                me: [] for me in range(self.ME)
            }

            # ============================================================
            # PS
            #
            # Kept for backward compatibility with the current
            # FedPredict implementation.
            #
            # PS is NOT used as the shift detector.
            # ============================================================

            self.ps_list = {
                me: [] for me in range(self.ME)
            }

            # ============================================================
            # LABEL SHIFT
            #
            # Scalar LS values received from participating clients.
            # The server never receives client class distributions.
            # ============================================================

            self.ls = [
                          0.0
                      ] * self.ME

            self.ls_list = {
                me: [] for me in range(self.ME)
            }

            # ============================================================
            # GENERIC DATA SHIFT
            #
            # Scalar CD values received from participating clients.
            # The server never receives X, Y, P(Y), or P(X|Y).
            # ============================================================

            # ============================================================
            # DATA HETEROGENEITY
            #
            # DH remains independent from shift detection.
            # ============================================================

            self.heterogeneity_degree = [
                                            -1
                                        ] * self.ME

            self.heterogeneity_degree_list = {
                me: [] for me in range(self.ME)
            }

            # ============================================================
            # DETECTOR STATE
            # ============================================================

            self.data_shift_detected = [
                                         False
                                     ] * self.ME

            self.data_shift_score = [
                                       0.0
                                   ] * self.ME


            # Unified operational detector threshold. LS and CD are
            # complementary evidence only; the operational state is
            # exclusively DATA_SHIFT or NO_SHIFT.
            self.data_shift_threshold = 0.2

            # ============================================================
            # DATA-SHIFT ADAPTATION
            # ============================================================

            self.min_drift_interval = 10

            self.last_drift_round = [
                                        -self.min_drift_interval
                                    ] * self.ME

            self.in_adaptation = [
                                     False
                                 ] * self.ME

            self.adaptation_until = [
                                        -1
                                    ] * self.ME

            self.data_drift_model = -1

            # ============================================================
            # SHIFT-DETECTION EVALUATION
            # ============================================================

            self.detector = self.strategy_name

            self.dataset = self.args.dataset

            if "combined_shift" in self.args.experiment_id:
                self.shift_type = "COMBINED_SHIFT"
            elif "label_shift" in self.args.experiment_id:
                self.shift_type = "LABEL_SHIFT"
            elif "concept_drift" in self.args.experiment_id:
                self.shift_type = "CONCEPT_DRIFT"
            else:
                self.shift_type = "NO_SHIFT"


            self.shift_configuration = (
                self.args.experiment_id
            )

        except Exception as e:
            print("__init__ error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

    def set_clients(self):

        try:

            # ============================================================
            # Shift detector state
            # ============================================================

            self.data_shift_detected = {
                me: False
                for me in range(self.ME)
            }

            self.data_shift_score = {
                me: 0.0
                for me in range(self.ME)
            }


            self.combined_train_accuracy = {me: 0.0 for me in range(self.ME)}
            self.combined_train_accuracy_history = {me: [] for me in range(self.ME)}
            self.combined_accuracy_reference = {me: 0.0 for me in range(self.ME)}
            self.combined_accuracy_drop = {me: 0.0 for me in range(self.ME)}
            self.combined_accuracy_drop_significant = {me: False for me in range(self.ME)}

            self.gw = {me: [] for me in range(self.ME)}
            self.lw = {me: [] for me in range(self.ME)}
            self.gw_by_round = {me: {} for me in range(self.ME)}
            self.lw_by_round = {me: {} for me in range(self.ME)}

            # ============================================================
            # Client-level generic-data-shift information
            # ============================================================

            # ============================================================
            # Shift history
            # ============================================================

            self.shift_rounds = {
                me: []
                for me in range(self.ME)
            }

            self.shift_detected = {
                me: []
                for me in range(self.ME)
            }

            self.shift_ground_truth = {
                me: []
                for me in range(self.ME)
            }

            self.shift_ground_truth_state = {
                me: []
                for me in range(self.ME)
            }

            self.shift_ground_truth_event = {
                me: []
                for me in range(self.ME)
            }

            # ============================================================
            # Detector state
            # ============================================================

            self.previous_detector_state = {
                me: False
                for me in range(self.ME)
            }

            self.detection_event = {
                me: 0
                for me in range(self.ME)
            }

            self.first_data_shift_round = {
                me: None
                for me in range(self.ME)
            }

            self.false_alarm_rounds = {
                me: []
                for me in range(self.ME)
            }

            self.true_detection_round = {
                me: None
                for me in range(self.ME)
            }

            self.detection_delay = {
                me: -1
                for me in range(self.ME)
            }

            # ============================================================
            # Existing shift information
            # ============================================================

            self.data_shift_model = -1

            self.data_shift_round = {
                me: -1
                for me in range(self.ME)
            }

            # ============================================================
            # Model-level metric containers
            # ============================================================

            self.fc = {
                me: 0.0
                for me in range(self.ME)
            }

            self.il = {
                me: 0.0
                for me in range(self.ME)
            }

            self.ps = {
                me: 0.0
                for me in range(self.ME)
            }

            self.ls = {
                me: 0.0
                for me in range(self.ME)
            }

            self.similarity = {
                me: 1.0
                for me in range(self.ME)
            }

            self.heterogeneity_degree = {
                me: 0.0
                for me in range(self.ME)
            }

            # ============================================================
            # Temporal histories
            # ============================================================

            self.fc_list = {
                me: []
                for me in range(self.ME)
            }

            self.il_list = {
                me: []
                for me in range(self.ME)
            }

            self.ps_list = {
                me: []
                for me in range(self.ME)
            }

            self.ls_list = {
                me: []
                for me in range(self.ME)
            }

            self.similarity_list = {
                me: []
                for me in range(self.ME)
            }

            self.heterogeneity_degree_list = {
                me: []
                for me in range(self.ME)
            }

            # ============================================================
            # IMPORTANT:
            # Create clients before using self.clients_ids.
            # ============================================================

            client_class = (
                ClientMultiFedAvgWithMultiFedPredict
            )

            for i in range(self.total_clients):
                client = client_class(
                    self.args,
                    id=i,
                    model=copy.deepcopy(
                        self.global_model
                    ),
                    fold_id=self.fold_id
                )

                self.clients.append(
                    client
                )

            # ============================================================
            # Client IDs
            # ============================================================

            self.clients_ids = [
                client.client_id
                for client in self.clients
            ]

            self.clients_ids_uniform_selection = [
                client_id
                for client_id in copy.deepcopy(
                    self.clients_ids
                )
            ]

            # ============================================================
            # Client-level metric containers
            # ============================================================

            self.client_metrics = {
                client_id: {
                    me: {}
                    for me in range(self.ME)
                }
                for client_id in self.clients_ids
            }

            self.selected_clients_m = [
                []
                for me in range(self.ME)
            ]

            # ============================================================
            # ADAPTATION STATE
            #
            # Persistent set containing all clients that have already
            # trained during the current adaptation phase.
            #
            # This MUST be independent from selected_clients_m because
            # selected_clients_m is reconstructed every round.
            # ============================================================

            self.adaptation_trained_clients = {
                me: set()
                for me in range(self.ME)
            }

            # ============================================================
            # Ground-truth shift rounds
            #
            # These come from the same configuration used by the
            # client-side detector.
            # ============================================================

            if len(self.clients) > 0:

                for me in range(self.ME):

                    if (
                            me
                            in self.clients[0].data_shift_config
                    ):
                        self.shift_rounds[me] = (
                            self.clients[0]
                            .data_shift_config[me]
                            ["data_shift_rounds"]
                        )

        except Exception as e:

            print(
                "set_clients error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            # Do not silently continue with a partially
            # initialized server.

            raise

    # original
    def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
    ):
        """Aggregate fit results using weighted average."""
        try:
            # ============================================================
            # MultiFedAvg
            # ============================================================

            self.selected_clients_m = [
                []
                for me in range(self.ME)
            ]

            trained_models = []

            results_mefl = {
                me: []
                for me in range(self.ME)
            }

            for i in range(len(results)):

                parameter, num_examples, result = (
                    results[i]
                )

                me = result["me"]

                if me not in trained_models:
                    trained_models.append(me)

                client_id = result["client_id"]

                self.selected_clients_m[
                    me
                ].append(client_id)

                results_mefl[
                    me
                ].append(
                    results[i]
                )

            # ============================================================
            # Aggregate model parameters
            # ============================================================

            aggregated_ndarrays_mefl = {
                me: []
                for me in range(self.ME)
            }

            print(
                f"modelos treinados rodada "
                f"{server_round} "
                f"trained models "
                f"{trained_models}"
            )

            for me in trained_models:

                weights_results = [
                    (
                        parameters,
                        num_examples
                    )
                    for (
                        parameters,
                        num_examples,
                        fit_res
                    ) in results_mefl[me]
                ]

                if len(weights_results) > 1:

                    aggregated_ndarrays_mefl[
                        me
                    ] = self.aggregate(
                        weights_results,
                        self.heterogeneity_degree[me],
                        self.parameters_aggregated_mefl[me],
                        server_round,
                        me
                    )

                elif len(weights_results) == 1:

                    aggregated_ndarrays_mefl[
                        me
                    ] = results_mefl[me][0][0]

            for me in trained_models:
                self.parameters_aggregated_mefl[
                    me
                ] = aggregated_ndarrays_mefl[me]

            # ============================================================
            # Aggregate custom training metrics
            # ============================================================

            metrics_aggregated_mefl = {
                me: []
                for me in range(self.ME)
            }

            for me in trained_models:

                if self.fit_metrics_aggregation_fn:

                    fit_metrics = [
                        (
                            num_examples,
                            metrics
                        )
                        for (
                            _,
                            num_examples,
                            metrics
                        ) in results_mefl[me]
                    ]

                    metrics_aggregated_mefl[
                        me
                    ] = self.fit_metrics_aggregation_fn(
                        fit_metrics
                    )

                    self.train_losses[me].append(
                        metrics_aggregated_mefl[
                            me
                        ]["Loss"]
                    )

                    aggregated_train_accuracy = metrics_aggregated_mefl[me].get(
                        "Accuracy"
                    )
                    if aggregated_train_accuracy is not None:
                        aggregated_train_accuracy = float(
                            np.clip(aggregated_train_accuracy, 0.0, 1.0)
                        )
                        self.train_accuracy_list[me].append(
                            aggregated_train_accuracy
                        )
                        self.train_accuracy_by_round[me][server_round] = (
                            aggregated_train_accuracy
                        )

                    print(
                        f"Teste data shift "
                        f"modelo {me} "
                        f"rodada {server_round} "
                        f"data shift="
                        f"{'DATA_SHIFT' if self.data_shift_detected[me] else 'NO_SHIFT'}"
                    )

                else:

                    print("nao tem")

            # ============================================================
            # Shift-detection CSVs
            # ============================================================

            print(
                "finalizou aggregated fit"
            )

            self.metrics_aggregated_mefl = (
                metrics_aggregated_mefl
            )

            parameters_aggregated_mefl = (
                self.parameters_aggregated_mefl
            )

            metrics_aggregated_mefl = (
                self.metrics_aggregated_mefl
            )

            if server_round == 1:

                for me in range(self.ME):
                    self.model_shape_mefl[me] = [
                        i.shape
                        for i in
                        parameters_aggregated_mefl[me]
                    ]

            # ============================================================
            # Collect client-level metrics
            # ============================================================

            clients_parameters_mefl = {
                me: []
                for me in range(self.ME)
            }

            fc_list = {
                me: []
                for me in range(self.ME)
            }

            il_list = {
                me: []
                for me in range(self.ME)
            }

            ps_list = {
                me: []
                for me in range(self.ME)
            }

            ls_list = {
                me: []
                for me in range(self.ME)
            }


            num_participating_clients = {
                me: 0
                for me in range(self.ME)
            }

            similarity_list = {
                me: []
                for me in range(self.ME)
            }

            num_samples_list = {
                me: []
                for me in range(self.ME)
            }

            # ============================================================
            # Process only clients that actually trained this round
            # ============================================================

            for i in range(len(results)):

                parameter, num_examples, result = (
                    results[i]
                )

                alpha = result["alpha"]

                me = result["me"]

                client_id = result["client_id"]

                non_iid = result.get(
                    "non_iid",
                    {}
                )

                fc = float(
                    non_iid.get(
                        "fc",
                        0.0
                    )
                )

                il = float(
                    non_iid.get(
                        "il",
                        0.0
                    )
                )

                ps = float(
                    non_iid.get(
                        "ps",
                        0.0
                    )
                )

                similarity = float(
                    non_iid.get(
                        "similarity",
                        1.0
                    )
                )

                # ========================================================
                # LS
                # ========================================================

                ls = float(
                    non_iid.get(
                        "ls",
                        0.0
                    )
                )

                ls = float(
                    np.clip(
                        ls,
                        0.0,
                        1.0
                    )
                )


                # ========================================================
                # Client-level generic-data-shift evidence
                # ========================================================

                num_participating_clients[me] += 1

                # ========================================================
                # Client metric history
                # ========================================================

                if (
                        alpha
                        not in
                        self.client_metrics[
                            client_id
                        ][me].keys()
                ):
                    self.client_metrics[
                        client_id
                    ][me][alpha] = {
                        "fc": None,
                        "il": None,
                        "similarity": None,
                        "ls": None,
                    }

                self.client_metrics[
                    client_id
                ][me][alpha]["fc"] = fc

                self.client_metrics[
                    client_id
                ][me][alpha]["il"] = il

                self.client_metrics[
                    client_id
                ][me][alpha]["similarity"] = (
                    similarity
                )

                self.client_metrics[
                    client_id
                ][me][alpha]["ls"] = ls

                # ========================================================
                # Per-model lists
                # ========================================================

                fc_list[me].append(fc)

                il_list[me].append(il)

                ps_list[me].append(ps)

                ls_list[me].append(ls)


                similarity_list[
                    me
                ].append(
                    similarity
                )

                num_samples_list[
                    me
                ].append(
                    num_examples
                )

                clients_parameters_mefl[
                    me
                ].append(
                    results[i][0]
                )

            print(
                f"Metricas antes rodada "
                f"{server_round}"
            )

            print(
                "fc_list",
                fc_list
            )

            print(
                "il_list",
                il_list
            )

            print(
                "ps_list",
                ps_list
            )

            print(
                "ls_list",
                ls_list
            )

            print(
                "num_samples_list",
                num_samples_list
            )

            # ============================================================
            # Aggregate metrics for trained models
            # ============================================================

            for me in trained_models:
                self.fc[me] = (
                    self._weighted_average(
                        fc_list[me],
                        num_samples_list[me]
                    )
                )

                self.il[me] = (
                    self._weighted_average(
                        il_list[me],
                        num_samples_list[me]
                    )
                )

                self.ps[me] = (
                    self._weighted_average(
                        ps_list[me],
                        num_samples_list[me]
                    )
                )

                # ========================================================
                # Aggregate LS
                # ========================================================

                self.ls[me] = (
                    self._weighted_average(
                        ls_list[me],
                        num_samples_list[me]
                    )
                )

                self.similarity[me] = (
                    self._weighted_average(
                        similarity_list[me],
                        num_samples_list[me]
                    )
                )

                # ========================================================
                # DH
                #
                # DH remains independent from LS/CD.
                # ========================================================

                self.heterogeneity_degree[
                    me
                ] = round(
                    (
                            (1 - self.fc[me])
                            + self.il[me]
                    ) / 2,
                    2
                )

                # ========================================================
                # Store temporal histories
                #
                # IMPORTANT:
                # These are histories of scalar signals received from
                # participating clients.
                # ========================================================

                self.ls_list[me].append(
                    self.ls[me]
                )

                # combined_train_accuracy is returned by client.fit().
                # Keep the same weighted aggregation semantics used for
                # the ordinary training accuracy.
                combined_train_accuracy = metrics_aggregated_mefl[me].get(
                    "combined_train_accuracy"
                )
                if combined_train_accuracy is not None:
                    self.combined_train_accuracy[me] = float(
                        np.clip(combined_train_accuracy, 0.0, 1.0)
                    )
                    self.combined_train_accuracy_history[me].append(
                        self.combined_train_accuracy[me]
                    )

                self.heterogeneity_degree_list[
                    me
                ].append(
                    self.heterogeneity_degree[me]
                )

                print(
                    f"round {server_round} "
                    f"fc {self.fc[me]} "
                    f"il {self.il[me]} "
                    f"similarity "
                    f"{self.similarity[me]} "
                    f"ps {self.ps[me]} "
                    f"ls {self.ls[me]} "
                    f"data_shift_score {self.data_shift_score[me]} "
                    f"heterogeneity_degree "
                    f"{self.heterogeneity_degree[me]}"
                )

            # ============================================================
            # DATA-SHIFT DETECTION
            # ============================================================
            # IMPORTANT: this must run AFTER the loop above has stored the
            # current round's LS and combined_train_accuracy histories.
            # Detection is therefore entirely fit-time. aggregate_evaluate()
            # is not involved in detecting data shift.
            for me in trained_models:
                self._detect_data_shift_after_fit(
                    server_round, model=me
                )

            # ============================================================
            # Layer-wise FedPredict similarity
            # ============================================================

            flag = False

            if server_round == 1:
                flag = True

            print(
                "Flag: ",
                flag
            )

            for me in range(self.ME):

                if "dls" in self.compression:

                    if flag:

                        (
                            self.similarity_between_layers_per_round_and_client[
                                me
                            ][server_round],
                            self.similarity_between_layers_per_round[
                                me
                            ][server_round],
                            self.mean_similarity_per_round[
                                me
                            ][server_round],
                            self.similarity_list_per_layer[me],
                            self.df[me]
                        ) = fedpredict_layerwise_similarity(
                            parameters_aggregated_mefl[me],
                            clients_parameters_mefl[me],
                            self.similarity_list_per_layer[me]
                        )

                    else:

                        (
                            self.similarity_between_layers_per_round_and_client[
                                me
                            ][server_round],
                            self.similarity_between_layers_per_round[
                                me
                            ][server_round],
                            self.mean_similarity_per_round[
                                me
                            ][server_round],
                            self.similarity_list_per_layer[me]
                        ) = (
                            self.similarity_between_layers_per_round_and_client[
                                me
                            ][server_round - 1],
                            self.similarity_between_layers_per_round[
                                me
                            ][server_round - 1],
                            self.mean_similarity_per_round[
                                me
                            ][server_round - 1],
                            self.similarity_list_per_layer[me]
                        )

                else:

                    self.similarity_between_layers_per_round[
                        me
                    ][server_round] = []

                    self.mean_similarity_per_round[
                        me
                    ][server_round] = 0

                    self.similarity_between_layers_per_round_and_client[
                        me
                    ][server_round] = []

                    self.df[me] = 1

            print(
                f"df: {self.df}"
            )

            return (
                parameters_aggregated_mefl,
                metrics_aggregated_mefl
            )

        except Exception as e:

            print(
                "aggregate_fit error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

    def aggregate(
            self,
            results: list[tuple[NDArrays, int]],
            heterogeneity_degree: float,
            current_parameters: list[tuple[NDArrays, int]],
            t: int,
            me: int
    ) -> NDArrays:

        try:

            """Compute weighted average."""

            # Calculate the total number of examples used during
            # training.
            num_examples_total = sum(
                num_examples
                for (_, num_examples) in results
            )

            # Create a list of weights, each multiplied by the
            # related number of examples.
            weighted_parameters_update_list = [
                [
                    layer * num_examples
                    for layer in weights
                ]
                for weights, num_examples in results
            ]

            weighted_parameters_update_list = []

            for i, r in enumerate(results):

                weights, num_examples = r

                client_update = []

                for j, layer in enumerate(weights):
                    original_layer = current_parameters[j]

                    update = (
                            layer
                            - original_layer
                    )

                    client_update.append(
                        update * num_examples
                    )

                weighted_parameters_update_list.append(
                    client_update
                )

            # Compute average weights of each layer.
            weighted_parameters_update: NDArrays = [
                reduce(
                    np.add,
                    layer_updates
                ) / num_examples_total
                for layer_updates
                in zip(
                    *weighted_parameters_update_list
                )
            ]

            threshold = [0.3, 0.6, 0.7]

            # ---------------------------------------------------------
            # IMPORTANT:
            #
            # LS replaces PS as the signal indicating label-
            # distribution change.
            #
            # DH itself continues to control the aggregation degree.
            # ---------------------------------------------------------
            if (
                    self.version in ["iti"]
                    or t == 1
            ):
                heterogeneity_degree = 0

            elif (
                    heterogeneity_degree > threshold[me] and heterogeneity_degree < 0.8
            ):
                heterogeneity_degree = (
                    heterogeneity_degree
                )

            elif heterogeneity_degree >= 0.8:
                heterogeneity_degree = 1

            else:
                heterogeneity_degree = 0

            global_lr = 1 - heterogeneity_degree

            weighted_parameters_update_list = [
                np.array(original_layer + (1 - heterogeneity_degree) * layer)
                for original_layer, layer
                in zip(
                    current_parameters,
                    weighted_parameters_update
                )
            ]

            return weighted_parameters_update_list

        except Exception as e:

            print("aggregate error")

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

    def binomial(self, sucessos, n_treinados):

        try:
            # Dados observados
            frac_treinados = sucessos / n_treinados

            # Prior uniforme Beta(1,1)
            alpha_prior, beta_prior = 2, 2

            # Posterior Beta(alpha+sucessos, beta+(n-sucessos))
            alpha_post = alpha_prior + sucessos
            beta_post = beta_prior + (n_treinados - sucessos)

            print(f"Posterior: Beta({alpha_post}, {beta_post})")

            # Probabilidade esperada (valor médio de p)
            p_media = alpha_post / (alpha_post + beta_post)
            print(f"Probabilidade esperada de um cliente não-treinado ter acurácia menor: {p_media:.4f}")

            return p_media

        except Exception as e:
            print("binomial error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _detect_data_shift_after_fit(self, server_round, model=None):
        """Detect data shift from fit-time evidence of ``server_round``.

        The detector runs at the end of ``aggregate_fit()``, after the
        current round's LS and combined-train-accuracy values have been
        aggregated and appended to their histories.  No validation/test
        metrics are required for shift detection.
        """
        try:
            ls_threshold = float(self.data_shift_threshold)
            acc_drop_threshold = float(self.combined_accuracy_drop_threshold)
            history_window = int(self.combined_accuracy_history_window)

            models = range(self.ME) if model is None else [int(model)]
            for me in models:
                ls_history = self.ls_list[me]
                current_ls = (
                    float(np.clip(ls_history[-1], 0.0, 1.0))
                    if ls_history else 0.0
                )
                previous_ls_history = ls_history[:-1]
                ls_significant = current_ls >= ls_threshold

                history = self.combined_train_accuracy_history[me]
                current_accuracy = (
                    float(np.clip(history[-1], 0.0, 1.0))
                    if history else float(np.clip(self.combined_train_accuracy[me], 0.0, 1.0))
                )
                previous_history = history[:-1]

                accuracy_history_ready = len(previous_history) >= history_window
                if accuracy_history_ready:
                    reference = float(np.mean(previous_history[-history_window:]))
                    relative_drop = (
                        max(0.0, (reference - current_accuracy) / reference)
                        if reference > 1e-12 else 0.0
                    )
                    accuracy_drop_significant = relative_drop >= acc_drop_threshold
                else:
                    reference = 0.0
                    relative_drop = 0.0
                    accuracy_drop_significant = False

                self.combined_accuracy_reference[me] = reference
                self.combined_accuracy_drop[me] = relative_drop
                self.combined_accuracy_drop_significant[me] = accuracy_drop_significant

                label_shift_history_ready = len(previous_ls_history) >= history_window

                if self.shift_type == "LABEL_SHIFT":
                    shift_detection_ready = label_shift_history_ready
                    detected = bool(shift_detection_ready and ls_significant)
                else:
                    shift_detection_ready = accuracy_history_ready
                    detected = bool(
                        shift_detection_ready
                        and (ls_significant or accuracy_drop_significant)
                    )

                self.data_shift_detected[me] = detected
                self.data_shift_score[me] = max(current_ls, relative_drop)


                print(
                    f"[DATA SHIFT DETECTOR] round={server_round} model={me} "
                    f"evidence_round={server_round} "
                    f"shift_type={self.shift_type} "
                    f"LS={current_ls:.6f} LS_threshold={ls_threshold:.6f} "
                    f"LS_significant={ls_significant} "
                    f"combined_train_accuracy={current_accuracy:.6f} "
                    f"accuracy_reference={reference:.6f} "
                    f"relative_accuracy_drop={relative_drop:.6f} "
                    f"drop_threshold={acc_drop_threshold:.6f} "
                    f"accuracy_drop_significant={accuracy_drop_significant} "
                    f"ls_history_size={len(ls_history)} "
                    f"label_shift_history_ready={label_shift_history_ready} "
                    f"accuracy_history_size={len(history)} "
                    f"accuracy_history_ready={accuracy_history_ready} "
                    f"state={'DATA_SHIFT' if detected else 'NO_SHIFT'}"
                )

        except Exception as e:
            print("_detect_data_shift_after_fit error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))

    def _commit_detector_state(self):
        """Commit the detector state after the round's client selection."""
        for me in range(self.ME):
            self.previous_detector_state[me] = bool(self.data_shift_detected[me])

    def select_clients(self, t):

        try:

            g = torch.Generator()

            g.manual_seed(
                t
            )

            random.seed(
                t
            )

            np.random.seed(
                t
            )

            torch.manual_seed(
                t
            )

            if self.version in ["dh"]:
                return super().select_clients(
                    t
                )

            # ============================================================
            # SERVER-SIDE DATA-SHIFT DETECTION
            #
            # Detection is completed in aggregate_fit(), after the current
            # round's fit-time evidence has been aggregated. Therefore,
            # select_clients(t) consumes the detector state produced by the
            # preceding round.
            # ============================================================
            shift_detected = [
                bool(self.data_shift_detected[me])
                for me in range(self.ME)
            ]

            # ============================================================
            # Determine whether a NEW adaptation must be started
            #
            # IMPORTANT:
            #
            # The clients responsible for detecting the shift are the
            # clients that trained in the previous round.
            #
            # They have already processed the new data and therefore
            # MUST NOT be trained again during this adaptation.
            #
            # The persistent adaptation_trained_clients set is initialized
            # with those clients and then accumulates every client trained
            # during the adaptation phase.
            # ============================================================

            newly_detected_model = -1

            for me in range(self.ME):

                if not shift_detected[me]:
                    continue

                # --------------------------------------------------------
                # A persistent shift is not a new event.
                #
                # Adaptation starts only on the transition:
                #
                # NO_SHIFT -> DATA_SHIFT
                # --------------------------------------------------------

                previous_state = bool(
                    self.previous_detector_state.get(
                        me,
                        False
                    )
                )

                current_state = bool(
                    self.data_shift_detected[me]
                )

                new_shift_event = (
                        not previous_state
                        and current_state
                )

                if not new_shift_event:
                    print(
                        f"[SHIFT EVENT IGNORED] "
                        f"round={t} "
                        f"model={me} "
                        f"state="
                        f"{'DATA_SHIFT' if current_state else 'NO_SHIFT'} "
                        f"previous="
                        f"{'DATA_SHIFT' if previous_state else 'NO_SHIFT'} "
                        f"reason=persistent_shift"
                    )

                    continue

                # --------------------------------------------------------
                # Do not start another adaptation too soon.
                # --------------------------------------------------------

                if (
                        t
                        - self.last_drift_round[me]
                        < self.min_drift_interval
                ):
                    continue

                # --------------------------------------------------------
                # Start a NEW adaptation phase.
                # --------------------------------------------------------

                self.last_drift_round[me] = t

                # --------------------------------------------------------
                # RESET THE COMBINED-ACCURACY HISTORY AT THE NEW SHIFT
                # EVENT.
                #
                # The current round must NOT be discarded: its accuracy
                # is the first observation of the new data regime.
                # Keeping only the current value also prevents pre-shift
                # accuracies from contaminating future references.
                #
                # The detector has already used the PRE-SHIFT history
                # above to confirm this shift, so resetting here does not
                # affect the current detection decision.
                # --------------------------------------------------------
                current_accuracy_for_new_regime = float(
                    np.clip(self.combined_train_accuracy[me], 0.0, 1.0)
                )
                self.combined_train_accuracy_history[me] = [
                    current_accuracy_for_new_regime
                ]

                # LABEL_SHIFT has its own temporal warm-up because the
                # combined-model test is disabled for label-shift experiments.
                # Preserve the current LS observation as the first sample of
                # the new regime and require a full subsequent LS window
                # before another detection can be confirmed.
                current_ls_for_new_regime = float(np.clip(self.ls[me], 0.0, 1.0))
                self.ls_list[me] = [current_ls_for_new_regime]

                print(
                    f"[DATA SHIFT ACCURACY HISTORY RESET] "
                    f"round={t} model={me} "
                    f"preserved_current_accuracy={current_accuracy_for_new_regime:.6f} "
                    f"history_size={len(self.combined_train_accuracy_history[me])}"
                )
                print(
                    f"[DATA SHIFT LS HISTORY RESET] "
                    f"round={t} model={me} "
                    f"preserved_current_ls={current_ls_for_new_regime:.6f} "
                    f"history_size={len(self.ls_list[me])}"
                )

                self.in_adaptation[me] = True

                self.adaptation_until[me] = (
                        t + self.min_drift_interval
                )

                self.data_drift_model = me

                newly_detected_model = me

                # --------------------------------------------------------
                # IMPORTANT:
                #
                # The clients selected in the previous round are the
                # clients whose training produced the evidence of the
                # shift.
                #
                # Therefore, they are considered ALREADY TRAINED for
                # this adaptation phase.
                #
                # This state persists across subsequent rounds.
                # --------------------------------------------------------

                detected_clients = set(
                    self.selected_clients_m[me]
                )

                self.adaptation_trained_clients[me] = (
                    set(detected_clients)
                )

                print(
                    f"[ADAPTATION START] "
                    f"round={t} "
                    f"model={me} "
                    f"detected_clients="
                    f"{sorted(detected_clients)} "
                    f"already_covered="
                    f"{len(detected_clients)}/"
                    f"{len(self.clients_ids)}"
                )

                break

            # ============================================================
            # If there is no active adaptation, use the normal
            # MultiFedPredict selection mechanism.
            # ============================================================

            if (
                    self.data_drift_model < 0
                    or not self.in_adaptation[
                self.data_drift_model
            ]
            ):
                selected = super().select_clients(
                    t
                )
                self._commit_detector_state()
                return selected

            # ============================================================
            # Active adaptation
            # ============================================================

            adaptation_model = (
                self.data_drift_model
            )

            # ============================================================
            # Persistent adaptation coverage
            #
            # Unlike selected_clients_m, this set survives across rounds.
            # It contains:
            #
            # 1. Clients from the round that produced the shift detection.
            # 2. Clients already selected during adaptation.
            #
            # Thus, a client is never selected twice in the same
            # adaptation phase.
            # ============================================================

            already_adapted_clients = set(
                self.adaptation_trained_clients[
                    adaptation_model
                ]
            )

            print(
                f"[ADAPTATION] "
                f"round={t} "
                f"model={adaptation_model} "
                f"already_covered="
                f"{sorted(already_adapted_clients)} "
                f"covered="
                f"{len(already_adapted_clients)}/"
                f"{len(self.clients_ids)}"
            )

            # ============================================================
            # Build the pool of clients that HAVE NOT YET participated
            # in this adaptation phase.
            #
            # IMPORTANT:
            #
            # We intentionally DO NOT use
            # self.clients_ids_uniform_selection here.
            #
            # That list belongs to the normal MultiFedPredict selection
            # mechanism and can already have clients removed from it due
            # to previous normal rounds.
            #
            # Using it during adaptation could prevent some clients from
            # ever being selected, defeating the guarantee that every
            # client is trained exactly once after the detected shift.
            # ============================================================

            available_clients = [
                client_id
                for client_id in self.clients_ids
                if client_id not in already_adapted_clients
            ]

            # ============================================================
            # Select clients for the current adaptation round.
            #
            # At most num_training_clients are selected.
            # If fewer clients remain, all remaining clients are selected.
            # ============================================================

            selected_clients = []

            if len(available_clients) > 0:
                number_to_select = min(
                    self.num_training_clients,
                    len(available_clients)
                )

                selected_clients = sorted(
                    random.sample(
                        available_clients,
                        number_to_select
                    )
                )

            # ============================================================
            # Persistently register the clients selected in this adaptation
            # round.
            # ============================================================

            self.adaptation_trained_clients[
                adaptation_model
            ].update(
                selected_clients
            )

            # ============================================================
            # Remove selected clients from the normal uniform-selection
            # pool as well.
            #
            # This prevents them from being immediately selected again by
            # the normal mechanism after adaptation finishes.
            # ============================================================

            self.clients_ids_uniform_selection = [
                client_id
                for client_id
                in self.clients_ids_uniform_selection
                if client_id not in selected_clients
            ]

            # ============================================================
            # Select clients only for the model undergoing adaptation.
            # ============================================================

            sc = []

            for me in range(self.ME):

                if me == adaptation_model:

                    sc.append(
                        selected_clients
                    )

                else:

                    sc.append([])

            # ============================================================
            # Current adaptation coverage
            # ============================================================

            covered_clients = set(
                self.adaptation_trained_clients[
                    adaptation_model
                ]
            )

            remaining_clients = [
                client_id
                for client_id in self.clients_ids
                if client_id not in covered_clients
            ]

            print(
                f"[ADAPTATION SELECTION] "
                f"round={t} "
                f"model={adaptation_model} "
                f"selected={selected_clients} "
                f"covered="
                f"{len(covered_clients)}/"
                f"{len(self.clients_ids)} "
                f"remaining="
                f"{len(remaining_clients)}"
            )

            # ============================================================
            # Adaptation completion
            #
            # PRIMARY STOP CONDITION:
            #
            # The adaptation ends immediately when EVERY client has been
            # trained once during the adaptation phase, counting the
            # clients that trained in the round that produced the shift.
            #
            # The interval is NOT used as the normal adaptation duration.
            #
            # It remains only as a safeguard against an unexpectedly long
            # adaptation phase.
            # ============================================================

            adaptation_finished = False

            if (
                    len(covered_clients)
                    >= len(self.clients_ids)
            ):

                adaptation_finished = True

                print(
                    f"[ADAPTATION END] "
                    f"round={t} "
                    f"model={adaptation_model} "
                    f"reason=all_clients_covered "
                    f"covered="
                    f"{len(covered_clients)}/"
                    f"{len(self.clients_ids)}"
                )

            elif (
                    t >= self.adaptation_until[
                adaptation_model
            ]
            ):

                # --------------------------------------------------------
                # Safety timeout.
                #
                # This should NOT normally determine the duration.
                # It exists only to prevent an infinite adaptation in
                # case something unexpected prevents client coverage.
                # --------------------------------------------------------

                adaptation_finished = True

                print(
                    f"[ADAPTATION END] "
                    f"round={t} "
                    f"model={adaptation_model} "
                    f"reason=safety_interval_elapsed "
                    f"covered="
                    f"{len(covered_clients)}/"
                    f"{len(self.clients_ids)} "
                    f"remaining="
                    f"{len(remaining_clients)}"
                )

            # ============================================================
            # Reset adaptation state.
            # ============================================================

            if adaptation_finished:
                self.increased_training_intensity[
                    adaptation_model
                ] = 0

                self.in_adaptation[
                    adaptation_model
                ] = False

                self.data_drift_model = -1

                # --------------------------------------------------------
                # Clear the persistent adaptation state.
                #
                # A future shift event must start a new coverage cycle.
                # --------------------------------------------------------

                self.adaptation_trained_clients[
                    adaptation_model
                ] = set()

                # --------------------------------------------------------
                # Restore the normal uniform-selection pool.
                # --------------------------------------------------------

                self.clients_ids_uniform_selection = [
                    client_id
                    for client_id in copy.deepcopy(
                        self.clients_ids
                    )
                ]

            # ============================================================
            # If there are no clients to train in this adaptation round,
            # fall back to the normal selection mechanism.
            # ============================================================

            if len(selected_clients) == 0:
                print(
                    f"[ADAPTATION] "
                    f"round={t} "
                    f"model={adaptation_model} "
                    f"no_clients_available"
                )

                selected = super().select_clients(
                    t
                )
                self._commit_detector_state()
                return selected

            self._commit_detector_state()
            return sc

        except Exception as e:

            print(
                "select_clients error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            raise

    def evaluate(
            self,
            t,
            parameters_aggregated_mefl
    ):

        try:

            evaluate_results = []

            print(
                "inicio s"
            )

            for me in range(self.ME):

                clients_evaluate_list = []

                metrics = {
                    "fc": self.fc[me],

                    "il": self.il[me],

                    "heterogeneity_degree": (
                        self.heterogeneity_degree[me]
                    ),

                    # ----------------------------------------------------
                    # Backward compatibility.
                    # PS is NOT used for shift detection.
                    # ----------------------------------------------------

                    "ps": self.ps[me],

                    "similarity": (
                        self.similarity[me]
                    ),

                    # ----------------------------------------------------
                    # Label Shift
                    # ----------------------------------------------------

                    "ls": self.ls[me],

                    # ----------------------------------------------------
                    # Final general detector state
                    # ----------------------------------------------------

                    "data_shift": (
                        self.data_shift_detected[me]
                    ),

                    "data_shift_score": (
                        self.data_shift_score[me]
                    )
                }

                print(
                    f"data shift "
                    f"na rodada {t} "
                    f"no modelo {me} "
                    f"{'DATA_SHIFT' if self.data_shift_detected[me] else 'NO_SHIFT'} "
                    f"score={self.data_shift_score[me]:.6f} "
                    f"LS={self.ls[me]:.6f} "
                )

                for i in range(
                        len(self.clients)
                ):
                    client_dict = {}

                    client_dict[
                        "client"
                    ] = self.clients[i]

                    client_dict[
                        "cid"
                    ] = self.clients[
                        i
                    ].client_id

                    client_dict[
                        "nt"
                    ] = (
                            t
                            - self.clients[i].lt[me]
                    )

                    client_dict[
                        "lt"
                    ] = self.clients[
                        i
                    ].lt[me]

                    clients_evaluate_list.append(
                        (
                            self.clients[i],
                            EvaluateIns(
                                ndarrays_to_parameters(
                                    parameters_aggregated_mefl[
                                        me
                                    ]
                                ),
                                client_dict
                            )
                        )
                    )

                print(
                    f"submetidos t: "
                    f"{self.t_hat[me]} "
                    f"T: "
                    f"{self.number_of_rounds} "
                    f"df: "
                    f"{self.df[me]}"
                )

                clients_compressed_parameters = (
                    fedpredict_server(
                        global_model_parameters=(
                            parameters_aggregated_mefl[
                                me
                            ]
                        ),
                        client_evaluate_list=(
                            clients_evaluate_list
                        ),
                        t=t,
                        T=self.number_of_rounds,
                        df=self.df[me],
                        compression=self.compression,
                        fl_framework="flwr",
                        k_ratio=0.3
                    )
                )

                for i in range(
                        len(self.clients)
                ):
                    evaluate_results.append(
                        self.clients[i].evaluate(
                            me,
                            t,
                            parameters_to_ndarrays(
                                clients_compressed_parameters[
                                    i
                                ][1].parameters
                            ),
                            metrics
                        )
                    )

            (
                loss_aggregated_mefl,
                metrics_aggregated_mefl
            ) = self.aggregate_evaluate(
                server_round=t,
                results=evaluate_results,
                failures=[]
            )

        except Exception as e:

            print(
                "evaluate error"
            )

            print(
                "Error on line {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__
                )
            )

    def aggregate_evaluate(
            self,
            server_round,
            results,
            failures,
    ):
        """Aggregate evaluation metrics and combined-model train accuracy."""
        try:
            results_mefl = {me: [] for me in range(self.ME)}
            for loss, num_examples, result in results:
                metrics = result[2] if isinstance(result, tuple) and len(result) == 3 else result
                me = int(metrics["me"])
                results_mefl[me].append((loss, num_examples, metrics))

            loss_aggregated_mefl = {me: 0.0 for me in range(self.ME)}
            metrics_aggregated_mefl = {me: {} for me in range(self.ME)}

            for me in range(self.ME):
                rows = results_mefl[me]
                if not rows:
                    continue

                loss_aggregated_mefl[me] = weighted_loss_avg(
                    [(n, loss) for loss, n, _ in rows]
                )

                def weighted_metric(name):
                    values, weights = [], []
                    for _, n, m in rows:
                        value = m.get(name)
                        if value is None:
                            continue
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            continue
                        values.append(value)
                        weights.append(n)
                    if not values or sum(weights) <= 0:
                        return None
                    return float(np.average(values, weights=weights))

                train_acc = weighted_metric("train_accuracy")

                # gw/lw are produced by client.evaluate().
                # Keep them exactly as in the original implementation:
                # lists containing one value per evaluating client.
                self.gw[me] = [
                    metrics.get("gw")
                    for _, _, metrics in rows
                    if metrics.get("gw") is not None
                ]
                self.lw[me] = [
                    metrics.get("lw")
                    for _, _, metrics in rows
                    if metrics.get("lw") is not None
                ]
                self.gw_by_round[me][server_round] = list(self.gw[me])
                self.lw_by_round[me][server_round] = list(self.lw[me])

                metrics_aggregated_mefl[me] = {
                    "combined_train_accuracy": self.combined_train_accuracy[me] if self.combined_train_accuracy_history[me] else None,
                    "train_accuracy": train_acc,
                    "Round (t)": server_round,
                }

                if self.evaluate_metrics_aggregation_fn:
                    custom = self.evaluate_metrics_aggregation_fn(
                        [(n, m) for _, n, m in rows]
                    )
                    if custom:
                        metrics_aggregated_mefl[me].update(custom)

                if train_acc is not None:
                    metrics_aggregated_mefl[me]["train_accuracy"] = train_acc

                self.add_metrics(server_round, metrics_aggregated_mefl, me)
                self._save_results(server_round, me)


            # Detection metrics are written only after the current round's
            # evaluation has produced the detector state.
            self._save_shift_detection_metrics(server_round)
            self._save_shift_detection_curve(server_round)

            return loss_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            print("aggregate_evaluate error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))
            return {me: 0.0 for me in range(self.ME)}, {me: {} for me in range(self.ME)}

    def add_metrics(
            self,
            server_round,
            metrics_aggregated,
            me
    ):

        try:

            # ============================================================
            # Metrics added by MultiFedAvg/MultiFedPredict
            # ============================================================

            metrics_aggregated[
                me
            ]["Fraction fit"] = (
                self.fraction_fit
            )

            metrics_aggregated[
                me
            ]["# training clients"] = (
                self.n_trained_clients
            )

            metrics_aggregated[
                me
            ]["training clients and models"] = (
                self.selected_clients_m[me]
            )

            metrics_aggregated[
                me
            ]["Fold ID"] = (
                self.fold_id
            )

            metrics_aggregated[
                me
            ]["fc"] = (
                self.fc[me]
            )

            metrics_aggregated[
                me
            ]["il"] = (
                self.il[me]
            )

            metrics_aggregated[
                me
            ]["dh"] = (
                self.heterogeneity_degree[me]
            )

            metrics_aggregated[
                me
            ]["ls"] = (
                self.ls[me]
            )

            metrics_aggregated[
                me
            ]["ps"] = (
                self.ps[me]
            )

            metrics_aggregated[
                me
            ]["gw"] = (
                self.gw[me]
            )

            metrics_aggregated[
                me
            ]["lw"] = (
                self.lw[me]
            )

            # ============================================================
            # Combined-model accuracy evidence
            # ============================================================
            # Keep ONE canonical combined-train-accuracy field.
            metrics_aggregated[me]["combined_train_accuracy"] = self.combined_train_accuracy[me]
            metrics_aggregated[me]["Combined accuracy reference"] = self.combined_accuracy_reference[me]
            metrics_aggregated[me]["Combined accuracy drop"] = self.combined_accuracy_drop[me]
            metrics_aggregated[me]["Combined accuracy drop significant"] = self.combined_accuracy_drop_significant[me]
            metrics_aggregated[me]["Combined accuracy history size"] = len(self.combined_train_accuracy_history[me])

            # train_accuracy comes from aggregate_fit() and is indexed by
            # round, so skipped-training rounds remain empty.
            round_train_accuracy = self.train_accuracy_by_round[me].get(
                server_round
            )
            metrics_aggregated[me]["train_accuracy"] = round_train_accuracy

            # ============================================================
            # Data-shift information
            # ============================================================

            metrics_aggregated[
                me
            ]["Data shift"] = (
                "DATA_SHIFT"
                if self.data_shift_detected[me]
                else "NO_SHIFT"
            )

            metrics_aggregated[
                me
            ]["Ground truth shift"] = (
                self.shift_ground_truth_state[me][-1]
                if len(
                    self.shift_ground_truth_state[me]
                ) > 0
                else 0
            )

            print(
                f"[Metrics] "
                f"model={me} | "
                f"Data shift="
                f"{'DATA_SHIFT' if self.data_shift_detected[me] else 'NO_SHIFT'} | "
                f"score="
                f"{self.data_shift_score[me]:.6f} | "
                f"LS="
                f"{self.ls[me]:.6f}"
            )

            # ============================================================
            # Dynamic result dictionary
            # ============================================================

            if me not in self.results_test_metrics:
                self.results_test_metrics[
                    me
                ] = {}

            for metric, value in (
                    metrics_aggregated[me].items()
            ):

                if (
                        metric
                        not in
                        self.results_test_metrics[me]
                ):

                    self.results_test_metrics[
                        me
                    ][metric] = []

                    if (
                            metric
                            not in
                            self.test_metrics_names
                    ):
                        self.test_metrics_names.append(
                            metric
                        )

                self.results_test_metrics[
                    me
                ][metric].append(
                    value
                )

        except Exception as e:

            print(
                "add_metrics error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

    # ================================================================
    # Shift-detection evaluation
    # ================================================================

    def _write_header(
            self,
            file_path,
            header,
            mode="w"
    ):
        """
        Write a CSV header.

        mode="w" is used when initializing a new experiment so that
        previous experiment results are removed.

        The default is intentionally "w".
        """
        with open(
                file_path,
                mode,
                newline="",
                encoding="utf-8"
        ) as f:
            csv.writer(f).writerow(
                header
            )

    def _write_rows(
            self,
            file_path,
            rows
    ):
        """
        Append rows to an existing CSV file.

        This method must use append mode because the shift-detection
        metrics and curve are generated incrementally, one row per
        server round.
        """
        with open(
                file_path,
                "a",
                newline="",
                encoding="utf-8"
        ) as f:
            csv.writer(f).writerows(
                rows
            )

    def _init_shift_detection_files(
            self
    ):
        """
        Initialize the shift-detection CSV files for a NEW experiment.

        IMPORTANT:
        This method intentionally opens the files with mode="w".
        Therefore, previous results from an earlier experiment are
        discarded and the files start with only their headers.

        After initialization, _save_shift_detection_metrics() and
        _save_shift_detection_curve() use _write_rows(), which appends
        one row per round.
        """

        result_path = (
            self.get_result_path("test")
        )
        print("result pat", result_path)

        os.makedirs(
            result_path,
            exist_ok=True
        )

        # ============================================================
        # Shift detection metrics
        # ============================================================

        metrics_file = os.path.join(
            result_path,
            f"shift_detection_metrics_"
            f"{self.strategy_name}.csv"
        )

        # ============================================================
        # Shift detection curve
        # ============================================================

        curve_file = os.path.join(
            result_path,
            f"shift_detection_curve_"
            f"{self.strategy_name}.csv"
        )

        # ============================================================
        # Initialize metrics CSV
        # ============================================================

        self._write_header(
            metrics_file,
            [
                "Detector",
                "Dataset",
                "Fold ID",
                "Round",
                "Model",
                "Shift Type",
                "Shift Configuration",
                "Precision",
                "Recall",
                "F1",
                "Detection Delay",
                "False Alarms",
                "First Detection Round",
                "Shift Round",
                "TP",
                "FN",
                "TN",
            ],
            mode="w",
        )

        # ============================================================
        # Initialize curve CSV
        # ============================================================

        self._write_header(
            curve_file,
            [
                "Detector",
                "Dataset",
                "Fold ID",
                "Round",
                "Model",
                "Ground Truth",
                "Detection Event",
                "Detector State",
            ],
            mode="w",
        )

    def _save_shift_detection_metrics(self, server_round):
        """Save conventional round-level shift detection metrics.

        A prediction is the detector state for the current round and the
        ground truth is the configured shift state for that round. This
        avoids counting only the first detection event as TP and therefore
        produces standard TP/FP/FN/TN based Precision, Recall and F1.
        """
        try:
            result_path = self.get_result_path("test")
            os.makedirs(result_path, exist_ok=True)
            file_path = os.path.join(
                result_path,
                f"shift_detection_metrics_{self.strategy_name}.csv"
            )

            for me in range(self.ME):
                predicted = int(
                    self.data_shift_detected[me]
                )

                shift_rounds = sorted(self.shift_rounds.get(me, []))
                ground_truth = int(any(
                    server_round >= r for r in shift_rounds
                ))
                ground_truth_event = int(server_round in shift_rounds)

                history = getattr(self, "_round_confusion", None)
                if history is None:
                    self._round_confusion = {
                        m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                        for m in range(self.ME)
                    }
                cm = self._round_confusion[me]
                if predicted and ground_truth:
                    cm["tp"] += 1
                elif predicted and not ground_truth:
                    cm["fp"] += 1
                elif not predicted and ground_truth:
                    cm["fn"] += 1
                else:
                    cm["tn"] += 1

                tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2.0 * precision * recall / (precision + recall)
                      if (precision + recall) else 0.0)

                if ground_truth_event and self.true_detection_round[me] is None and predicted:
                    self.true_detection_round[me] = server_round
                    self.detection_delay[me] = server_round - shift_rounds[0]
                if predicted and not ground_truth:
                    self.false_alarm_rounds[me].append(server_round)

                if self.first_data_shift_round[me] is None and predicted:
                    self.first_data_shift_round[me] = server_round

                first_detection_round = (
                    self.first_data_shift_round[me]
                    if self.first_data_shift_round[me] is not None else -1
                )
                true_detection_round = (
                    self.true_detection_round[me]
                    if self.true_detection_round[me] is not None else -1
                )
                detection_delay = (
                    self.detection_delay[me]
                    if self.detection_delay[me] >= 0 else -1
                )
                shift_round = shift_rounds[0] if shift_rounds else -1

                self.detection_event[me] = int(
                    predicted and not getattr(self, "_previous_shift_prediction", {}).get(me, 0)
                )
                if not hasattr(self, "_previous_shift_prediction"):
                    self._previous_shift_prediction = {m: 0 for m in range(self.ME)}
                self._previous_shift_prediction[me] = predicted

                self.shift_ground_truth_state[me].append(ground_truth)
                self.shift_ground_truth_event[me].append(ground_truth_event)
                self.shift_detected[me].append(predicted)
                row = [[
                    self.detector,
                    self.dataset[me],
                    self.fold_id,
                    server_round,
                    me,
                    self.shift_type,
                    self.shift_configuration,
                    precision,
                    recall,
                    f1,
                    detection_delay,
                    fp,
                    first_detection_round,
                    shift_round,
                    tp,
                    fn,
                    cm["tn"],
                ]]

                self._write_rows(file_path, row)

        except Exception as e:
            print("_save_shift_detection_metrics error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _save_shift_detection_curve(
            self,
            server_round
    ):

        try:

            result_path = (
                self.get_result_path("test")
            )

            os.makedirs(
                result_path,
                exist_ok=True
            )

            file_path = os.path.join(
                result_path,
                f"shift_detection_curve_"
                f"{self.strategy_name}.csv"
            )

            for me in range(self.ME):
                ground_truth = (
                    self.shift_ground_truth_state[
                        me
                    ][-1]
                    if self.shift_ground_truth_state[
                        me
                    ]
                    else 0
                )

                detector_state = (
                    "DATA_SHIFT"
                    if self.data_shift_detected[me]
                    else "NO_SHIFT"
                )

                row = [[
                    self.detector,
                    self.dataset[me],
                    self.fold_id,
                    server_round,
                    me,
                    ground_truth,
                    self.detection_event[me],
                    detector_state,
                ]]

                self._write_rows(
                    file_path,
                    row
                )

        except Exception as e:

            print(
                "_save_shift_detection_curve error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

    def _get_results(self, train_test, mode, me):

        try:
            algo = self.dataset[me] + "_" + self.strategy_name

            result_path = self.get_result_path(train_test)

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            compression = self.compression
            if len(compression) > 0:
                compression = "_" + compression
            file_path = result_path + "{}{}.csv".format(algo, compression)

            # print("arquivo nome v2: ", file_path)
            # print(self.results_test_metrics[me])

            if train_test == 'test':

                header = self.test_metrics_names
                # print(self.rs_test_acc[me])
                # print(self.rs_test_auc[me])
                # print(self.rs_train_loss[me])
                list_of_metrics = []
                for metric in self.results_test_metrics[me]:
                    # print(me, len(self.results_test_metrics[me][metric]))
                    length = len(self.results_test_metrics[me][metric])
                    list_of_metrics.append(self.results_test_metrics[me][metric])

                data = []
                for i in range(length):
                    row = []
                    for j in range(len(list_of_metrics)):
                        row.append(list_of_metrics[j][i])

                    data.append(row)

            else:
                if mode == '':
                    header = self.train_metrics_names
                    list_of_metrics = []
                    for metric in self.results_train_metrics[me]:
                        # print(me, len(self.results_train_metrics[me][metric]))
                        length = len(self.results_train_metrics[me][metric])
                        list_of_metrics.append(self.results_train_metrics[me][metric])

                    data = []
                    # print("""tamanho: {}    {}""".format(length, list_of_metrics))
                    for i in range(length):
                        row = []
                        for j in range(len(list_of_metrics)):
                            if len(list_of_metrics[j]) > 0:
                                row.append(list_of_metrics[j][i])
                            else:
                                row.append(0)

                        data.append(row)

            # print("File path: " + file_path)
            print(data)

            return file_path, header, data
        except Exception as e:
            print("get_results error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _weighted_average(self, values, weights):

        try:
            values = np.array([i * j for i, j in zip(values, weights)])
            values = np.sum(values) / np.sum(weights)
            return round(float(values), 3)

        except Exception as e:
            print("_weighted_average error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))