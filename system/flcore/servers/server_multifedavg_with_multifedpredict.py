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
from scipy.stats import ks_2samp

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
        return {"Accuracy": sum(accuracies) / sum(examples),
                "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
                "Loss": sum(loss) / sum(examples), "Round (t)": metrics[0][1]["Round (t)"],
                "Model size": metrics[0][1]["Model size"]}
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

            self.gds = [
                          0.0
                      ] * self.ME

            self.gds_pvalue = [
                               1.0
                           ] * self.ME

            self.gds_pvalue_list = {
                me: [] for me in range(self.ME)
            }

            self.gds_evidence_streak = [
                                         0
                                     ] * self.ME

            self.gds_list = {
                me: [] for me in range(self.ME)
            }

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

            self.data_shift_score_list = {
                me: [] for me in range(self.ME)
            }

            # Unified operational detector threshold. LS and CD are
            # complementary evidence only; the operational state is
            # exclusively DATA_SHIFT or NO_SHIFT.
            self.data_shift_threshold = 0.10

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

            if "label_shift" in self.args.experiment_id:
                self.shift_type = "LABEL_SHIFT"
            elif "concept_drift" in  self.args.experiment_id:
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

            self.data_shift_score_list = {
                me: []
                for me in range(self.ME)
            }

            # ============================================================
            # Client-level generic-data-shift information
            # ============================================================

            self.gds_clients = {
                me: 0
                for me in range(self.ME)
            }

            self.gds_rate = {
                me: 0.0
                for me in range(self.ME)
            }

            self.max_gds = {
                me: 0.0
                for me in range(self.ME)
            }

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

            self.gds_rate_history = {
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

            self.gds = {
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

            self.gds_list = {
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
                for client_id
                in copy.deepcopy(
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

            gds_list = {
                me: []
                for me in range(self.ME)
            }

            data_shift_score_list = {
                me: []
                for me in range(self.ME)
            }

            gds_client_ids = {
                me: []
                for me in range(self.ME)
            }

            gds_client_scores = {
                me: []
                for me in range(self.ME)
            }

            gds_client_pvalues = {
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
                # CD
                # ========================================================

                gds = float(
                    non_iid.get(
                        "gds",
                        0.0
                    )
                )

                gds = float(np.clip(gds, 0.0, 1.0))

                gds_pvalue = float(
                    non_iid.get(
                        "gds_pvalue",
                        1.0
                    )
                )
                gds_pvalue = float(np.clip(gds_pvalue, 0.0, 1.0))

                data_shift_score = float(
                    non_iid.get(
                        "data_shift_score",
                        max(ls, gds)
                    )
                )
                data_shift_score = float(
                    np.clip(
                        data_shift_score,
                        0.0,
                        1.0
                    )
                )

                # ========================================================
                # Client-level generic-data-shift evidence
                # ========================================================

                num_participating_clients[me] += 1

                gds_client_scores[me].append(gds)
                gds_client_pvalues[me].append(gds_pvalue)

                if gds_pvalue < 0.05:
                    gds_client_ids[me].append(client_id)

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
                        "gds": None
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

                self.client_metrics[
                    client_id
                ][me][alpha]["gds"] = gds

                # ========================================================
                # Per-model lists
                # ========================================================

                fc_list[me].append(fc)

                il_list[me].append(il)

                ps_list[me].append(ps)

                ls_list[me].append(ls)

                gds_list[me].append(gds)

                data_shift_score_list[me].append(
                    data_shift_score
                )

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
                "gds_list",
                gds_list
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

                # ========================================================
                # Aggregate CD
                # ========================================================

                self.gds[me] = (
                    self._weighted_average(
                        gds_list[me],
                        num_samples_list[me]
                    )
                )

                # ========================================================
                # GENERAL DATA-SHIFT SCORE
                #
                # LS and CD remain available only as diagnostics.  The
                # adaptation policy uses this single aggregated score.
                # ========================================================

                self.data_shift_score[me] = (
                    self._weighted_average(
                        data_shift_score_list[me],
                        num_samples_list[me]
                    )
                )

                self.data_shift_score_list[me].append(
                    self.data_shift_score[me]
                )

                # Combine independent client-level p-values using Simes.
                # The server receives only scalar evidence, never P(X|Y).
                valid_pvalues = [
                    p for p in gds_client_pvalues[me]
                    if np.isfinite(p)
                ]
                if valid_pvalues:
                    p_sorted = np.sort(np.asarray(valid_pvalues, dtype=float))
                    m = float(len(p_sorted))
                    self.gds_pvalue[me] = float(np.clip(
                        np.min((m / np.arange(1.0, m + 1.0)) * p_sorted),
                        0.0, 1.0))
                else:
                    self.gds_pvalue[me] = 1.0


                # ========================================================
                # Client-level CD statistics
                # ========================================================

                self.gds_clients[me] = len(
                    gds_client_ids[me]
                )

                if num_participating_clients[me] > 0:

                    self.gds_rate[me] = round(
                        self.gds_clients[me]
                        / num_participating_clients[me],
                        3
                    )

                else:

                    self.gds_rate[me] = 0.0

                # ========================================================
                # Maximum client-level CD score
                # ========================================================

                if len(gds_client_scores[me]) > 0:

                    self.max_gds[me] = round(
                        float(
                            np.max(
                                gds_client_scores[me]
                            )
                        ),
                        3
                    )

                else:

                    self.max_gds[me] = 0.0

                # ========================================================
                # Historical drift rate
                # ========================================================

                self.gds_rate_history[me].append(
                    self.gds_rate[me]
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

                self.gds_list[me].append(self.gds[me])
                self.gds_pvalue_list[me].append(self.gds_pvalue[me])

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
                    f"gds {self.gds[me]} "
                    f"data_shift_score {self.data_shift_score[me]} "
                    f"heterogeneity_degree "
                    f"{self.heterogeneity_degree[me]}"
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

            self._save_shift_detection_metrics(
                server_round
            )

            self._save_shift_detection_curve(
                server_round
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
            # GENERAL DATA-SHIFT DETECTION
            #
            # The client computes a single scalar data-shift score from
            # complementary local signals.  The server does not classify
            # the shift as LABEL_SHIFT, CONCEPT_DRIFT, or COMBINED_SHIFT.
            #
            # A shift is detected when the aggregated score reaches the
            # common threshold.  This same decision drives adaptation for
            # every simulated shift scenario.
            # ============================================================

            shift_detected = [
                False
            ] * self.ME

            data_shift_threshold = float(self.data_shift_threshold)

            for me in range(self.ME):

                current_score = float(
                    np.clip(
                        self.data_shift_score[me],
                        0.0,
                        1.0
                    )
                )

                shift_detected[me] = (
                    current_score >= data_shift_threshold
                )

                self.data_shift_detected[me] = (
                    shift_detected[me]
                )

                print(
                    f"[DATA SHIFT DETECTOR] "
                    f"round={t} "
                    f"model={me} "
                    f"score={current_score:.6f} "
                    f"threshold={data_shift_threshold:.6f} "
                    f"state="
                    f"{'DATA_SHIFT' if shift_detected[me] else 'NO_SHIFT'}"
                )

            # ============================================================
            # Determine whether a NEW adaptation must be started
            #
            # IMPORTANT:
            #
            # The clients responsible for detecting the shift are the
            # clients that trained in the previous round. They have
            # already processed the new data.
            #
            # Therefore, detection at round t only initializes the
            # adaptation state. It does NOT cause those same clients to
            # be selected again.
            # ============================================================

            newly_detected_model = -1

            for me in range(self.ME):

                if not shift_detected[me]:
                    continue

                # --------------------------------------------------------
                # A persistent significant CD test is not a new event.
                # A new adaptation is started only on the transition
                # from NO_SHIFT to a shift state.
                # --------------------------------------------------------
                previous_state = bool(
                    self.previous_detector_state.get(
                        me, False
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
                        f"round={t} model={me} "
                        f"state={'DATA_SHIFT' if current_state else 'NO_SHIFT'} "
                        f"previous={'DATA_SHIFT' if previous_state else 'NO_SHIFT'} "
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
                #
                # The actual adaptation selection below excludes the
                # clients that generated the current shift evidence.
                # --------------------------------------------------------

                self.last_drift_round[me] = t

                self.in_adaptation[me] = True

                self.adaptation_until[me] = (
                        t + self.min_drift_interval
                )

                self.data_drift_model = me

                newly_detected_model = me

                print(
                    f"[ADAPTATION START] "
                    f"round={t} "
                    f"model={me} "
                    f"detected_clients="
                    f"{self.selected_clients_m[me]}"
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
                return super().select_clients(
                    t
                )

            # ============================================================
            # Active adaptation
            # ============================================================

            adaptation_model = (
                self.data_drift_model
            )

            # ============================================================
            # Clients that already participated in the round that
            # produced the shift detection.
            #
            # These clients already trained with the new data and must
            # NOT be selected again for adaptation.
            # ============================================================

            already_adapted_clients = set(
                self.selected_clients_m[
                    adaptation_model
                ]
            )

            print(
                f"[ADAPTATION] "
                f"round={t} "
                f"model={adaptation_model} "
                f"already_adapted="
                f"{sorted(already_adapted_clients)}"
            )

            # ============================================================
            # Build the pool of clients that are still available for
            # adaptation.
            #
            # IMPORTANT:
            #
            # Do not use the complete client set here because that could
            # select again the clients that detected the shift.
            # ============================================================

            available_clients = [
                client_id
                for client_id in self.clients_ids
                if client_id not in already_adapted_clients
            ]

            # ============================================================
            # Also respect the existing uniform-selection pool when
            # possible.
            # ============================================================

            uniform_available_clients = [
                client_id
                for client_id in self.clients_ids_uniform_selection
                if client_id not in already_adapted_clients
            ]

            # ============================================================
            # Prefer clients still present in the uniform-selection pool.
            # ============================================================

            if len(uniform_available_clients) > 0:

                candidate_clients = (
                    uniform_available_clients
                )

            else:

                candidate_clients = (
                    available_clients
                )

            # ============================================================
            # Select clients for the current adaptation round.
            # ============================================================

            selected_clients = []

            if len(candidate_clients) > 0:
                remaining = min(
                    self.num_training_clients,
                    len(candidate_clients)
                )

                selected_clients = sorted(
                    random.sample(
                        candidate_clients,
                        remaining
                    )
                )

            # ============================================================
            # Remove selected clients from the uniform-selection pool.
            # ============================================================

            self.clients_ids_uniform_selection = [
                client_id
                for client_id in self.clients_ids_uniform_selection
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
            # Diagnostics
            # ============================================================

            print(
                f"[ADAPTATION SELECTION] "
                f"round={t} "
                f"model={adaptation_model} "
                f"selected={selected_clients} "
                f"remaining="
                f"{len(available_clients) - len(selected_clients)}"
            )

            # ============================================================
            # Adaptation completion
            #
            # The adaptation phase finishes when there are no more
            # clients available for adaptation or when the configured
            # adaptation interval has elapsed.
            # ============================================================

            adaptation_finished = False

            if (
                    t >= self.adaptation_until[
                adaptation_model
            ]
            ):

                adaptation_finished = True

                print(
                    f"[ADAPTATION END] "
                    f"round={t} "
                    f"model={adaptation_model} "
                    f"reason=interval_elapsed"
                )

            elif len(available_clients) == 0:

                adaptation_finished = True

                print(
                    f"[ADAPTATION END] "
                    f"round={t} "
                    f"model={adaptation_model} "
                    f"reason=no_available_clients"
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

                return super().select_clients(
                    t
                )

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
                    # Generic Data Shift
                    # ----------------------------------------------------

                    "gds": self.gds[me],

                    "gds_pvalue": self.gds_pvalue[me],

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
                    f"GDS={self.gds[me]:.6f}"
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
            ]["gds"] = (
                self.gds[me]
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
            ]["Drift clients"] = (
                self.gds_clients[me]
            )

            metrics_aggregated[
                me
            ]["Drift rate"] = (
                self.gds_rate[me]
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
                f"{self.ls[me]:.6f} | "
                f"GDS="
                f"{self.gds[me]:.6f} | "
                f"Drift clients="
                f"{self.gds_clients[me]} | "
                f"Drift rate="
                f"{self.gds_rate[me]}"
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
                "Drift Clients",
                "Drift Rate",
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
                self.previous_detector_state[me] = bool(
                    self.data_shift_detected[me]
                )

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
                    self.gds_clients[me],
                    self.gds_rate[me],
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

    def detect_change_ema(self, signal, alpha=0.1, threshold=3.0):
        try:
            """
            Detecta mudanças repentinas usando EMA (Exponential Moving Average).

            Args:
                signal (list ou np.array): sequência de valores reais.
                alpha (float): fator de suavização da EMA (0<alpha<=1).
                threshold (float): múltiplos do desvio padrão do resíduo para detectar mudança.

            Returns:
                indices (list): pontos onde foram detectadas mudanças.
                ema (np.array): valores da EMA ao longo do tempo.
            """
            signal = np.array(signal)
            ema = np.zeros_like(signal, dtype=float)
            ema[0] = signal[0]

            # Calcula EMA
            for t in range(1, len(signal)):
                ema[t] = alpha * signal[t] + (1 - alpha) * ema[t - 1]

            # Resíduo
            residuals = signal - ema
            std = np.std(residuals)

            # Detecta mudanças quando resíduo "explode"
            change_points = [i for i, r in enumerate(residuals) if abs(r) > threshold * std]

            return change_points, ema
        except Exception as e:
            print("detect_change_ema error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))