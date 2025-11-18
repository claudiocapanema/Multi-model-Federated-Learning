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
import copy
import time
import numpy as np
from flcore.clients.client_multifedavg_with_multifedpredictv0 import ClientMultiFedAvgWithMultiFedPredictv0
from flcore.clients.client_multifedavg_with_fedpredict import ClientMultiFedAvgWithFedPredict
from flcore.clients.client_multifedavg_with_fedpredict_dynamic import ClientMultiFedAvgWithFedPredictDynamic
from flcore.servers.server_multifedavg import MultiFedAvg
import sys
from fedpredict import fedpredict_server, fedpredict_layerwise_similarity
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
import flwr
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


def aggregate(results: list[tuple[NDArrays, int]], homogeneity_degree: float, current_parameters: list[tuple[NDArrays, int]], t: int) -> NDArrays:
    try:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples for (_, num_examples) in results)

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]
        weighted_weights = []
        for i, r in enumerate(results):
            weights, num_examples = r
            client_update = []
            for j, layer in enumerate(weights):
                original_layer = current_parameters[j]
                update = layer - original_layer
                client_update.append(update * num_examples)

            weighted_weights.append(client_update)

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        # if t <= 59:
        #     homogeneity_degree = 1
        if t in [1, 20, 40, 60, 80]:
            homogeneity_degree = 1
        weighted_weights = [np.array(original_layer + layer) for original_layer, layer in zip(current_parameters, weights_prime)]
        return weighted_weights
    except Exception as e:
        print("aggregate error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def get_weights(net):
    try:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    except Exception as e:
        print("get_weights error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


class MultiFedAvgWithMultiFedPredictv0(MultiFedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.test_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                   "# training clients", "training clients and models", "Model size", "fc", "il", "dh", "ps", "Alpha", "gw", "lw"]
        self.results_test_metrics = {me: {metric: [] for metric in self.test_metrics_names} for me in range(self.ME)}
        self.compression = ""
        self.similarity_list_per_layer = {me: {} for me in range(self.ME)}
        self.initial_similarity = 0
        self.current_similarity = 0
        self.model_shape_mefl = [None] * self.ME
        self.similarity_between_layers_per_round = {me: {} for me in range(self.ME)}
        self.similarity_between_layers_per_round_and_client = {me: {} for me in range(self.ME)}
        self.mean_similarity_per_round = {me: {} for me in range(self.ME)}
        self.df = [0] * self.ME
        self.prediction_layer = {me: {"non_iid": 0, "parameters": []} for me in range(self.ME)}
        self.homogeneity_degree = {me: 0 for me in range(self.ME)}
        self.fc = {me: 0 for me in range(self.ME)}
        self.il = {me: 0 for me in range(self.ME)}
        self.ps =  {me: 0 for me in range(self.ME)}
        self.lw = {me: [] for me in range(self.ME)}
        self.gw = {me: [] for me in range(self.ME)}
        self.increased_training_intensity = {me: 0 for me in range(self.ME)}
        self.max_number_of_rounds_data_drift_adaptation = 5
        self.similarity = {me: 0 for me in range(self.ME)}
        self.t_hat = [1] * self.ME
        self.reduced_training_intensity_flag = [False] * self.ME

    def set_clients(self):

        try:
            client_class = ClientMultiFedAvgWithMultiFedPredictv0
            for i in range(self.total_clients):
                client = client_class(self.args,
                                id=i,
                                   model=copy.deepcopy(self.global_model))
                self.clients.append(client)

        except Exception as e:
            print("set_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate fit results using weighted average."""
        try:
            # MultiFedAvg

            self.selected_clients_m = [[] for me in range(self.ME)]

            trained_models = []

            results_mefl = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                parameter, num_examples, result = results[i]
                me = result["me"]
                if me not in trained_models:
                    trained_models.append(me)
                client_id = result["client_id"]
                self.selected_clients_m[me].append(client_id)
                results_mefl[me].append(results[i])

            aggregated_ndarrays_mefl = {me: None for me in range(self.ME)}
            aggregated_ndarrays_mefl = {me: [] for me in range(self.ME)}
            weights_results_mefl = {me: [] for me in range(self.ME)}
            # parameters_aggregated_mefl = {me: [] for me in range(self.ME)}

            print(f"modelos treinados rodada {server_round} trained models {trained_models}")
            for me in trained_models:
                # Convert results
                weights_results = [
                    (parameters, num_examples)
                    for parameters, num_examples, fit_res in results_mefl[me]
                ]
                aggregated_ndarrays_mefl[me] = aggregate(weights_results, self.homogeneity_degree[me], self.parameters_aggregated_mefl[me], server_round)
                if len(weights_results) > 1:
                    aggregated_ndarrays_mefl[me] = aggregate(weights_results, self.homogeneity_degree[me], self.parameters_aggregated_mefl[me], server_round)
                elif len(weights_results) == 1:
                    aggregated_ndarrays_mefl[me] = results_mefl[me][0][0]

            for me in trained_models:
                self.parameters_aggregated_mefl[me] = aggregated_ndarrays_mefl[me]

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated_mefl = {me: [] for me in range(self.ME)}
            for me in trained_models:
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(num_examples, metrics) for _, num_examples, metrics in results_mefl[me]]
                    metrics_aggregated_mefl[me] = self.fit_metrics_aggregation_fn(fit_metrics)

            print("""finalizou aggregated fit""")

            self.parameters_aggregated_mefl = self.parameters_aggregated_mefl
            self.metrics_aggregated_mefl = metrics_aggregated_mefl

            parameters_aggregated_mefl, metrics_aggregated_mefl = self.parameters_aggregated_mefl, self.metrics_aggregated_mefl
            if server_round == 1:
                for me in range(self.ME):
                    self.model_shape_mefl[me] = [i.shape for i in parameters_aggregated_mefl[me]]

            clients_parameters_mefl = {me: [] for me in range(self.ME)}
            fc_list = {me: [] for me in range(self.ME)}
            il_list = {me: [] for me in range(self.ME)}
            ps_list = {me: [] for me in range(self.ME)}
            similarity_list = {me: [] for me in range(self.ME)}
            num_samples_list = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                parameter, num_examples, result = results[i]
                me = result["me"]
                fc = result["non_iid"]["fc"]
                il = result["non_iid"]["il"]
                ps = result["non_iid"]["ps"]
                similarity = result["non_iid"]["similarity"]
                fc_list[me].append(fc)
                il_list[me].append(il)
                ps_list[me].append(ps)
                similarity_list[me].append(similarity)
                num_samples_list[me].append(num_examples)
                clients_parameters_mefl[me].append(results[i][0])

            for me in trained_models:
                self.fc[me] = self._weighted_average(fc_list[me], num_samples_list[me])
                self.il[me] = self._weighted_average(il_list[me], num_samples_list[me])
                self.ps[me] = self._weighted_average(ps_list[me], num_samples_list[me])
                self.similarity[me] = self._weighted_average(il_list[me], num_samples_list[me])
                self.homogeneity_degree[me] = round((self.fc[me] + (1 - self.il[me])) / 2, 2)
                # if self.homogeneity_degree[me] > self.prediction_layer[me]["non_iid"]:
                print(f"round {server_round} fc {self.fc[me]} il {self.il[me]} similarity {self.similarity[me]} ps {self.ps[me]} homogeneity_degree {self.homogeneity_degree[me]}")
                # n_layers = 2 * 1
                # # n_layers = 2 * 2 # melhor cnn
                # # n_layers = 1 # melhor gru
                # if server_round <= 59:
                # # if server_round <= 29 or server_round >= 60:
                #     print(f"Rodada {server_round} substituiu. Novo {self.homogeneity_degree[me]} antigo {self.prediction_layer[me]['non_iid']} diferença: {self.homogeneity_degree[me] - self.prediction_layer[me]["non_iid"]}")
                #     self.prediction_layer[me]["non_iid"] = self.homogeneity_degree[me]
                #     # label shift
                #     self.prediction_layer[me]["parameters"] = parameters_aggregated_mefl[me][-n_layers:]
                #     # concept drift
                #     # self.prediction_layer[me]["parameters"] = parameters_aggregated_mefl[me][:n_layers]

                # label shift
                # parameters_aggregated_mefl[me][-n_layers:] = self.prediction_layer[me]["parameters"]
                # concept drift
                # parameters_aggregated_mefl[me][:n_layers] = self.prediction_layer[me]["parameters"]



            flag = False
            if server_round == 1:
                flag = True
            print("Flag: ", flag)
            for me in range(self.ME):
                if "dls" in self.compression:
                    if flag:
                        self.similarity_between_layers_per_round_and_client[me][server_round], \
                        self.similarity_between_layers_per_round[me][server_round], self.mean_similarity_per_round[me][
                            server_round], self.similarity_list_per_layer[me], self.df[me] = fedpredict_layerwise_similarity(
                            parameters_aggregated_mefl[me], clients_parameters_mefl[me], self.similarity_list_per_layer[me])
                    else:
                        self.similarity_between_layers_per_round_and_client[me][server_round], \
                        self.similarity_between_layers_per_round[me][
                            server_round], self.mean_similarity_per_round[me][
                            server_round], self.similarity_list_per_layer[me] = \
                        self.similarity_between_layers_per_round_and_client[me][server_round - 1], \
                        self.similarity_between_layers_per_round[me][
                            server_round - 1], self.mean_similarity_per_round[me][
                            server_round - 1], self.similarity_list_per_layer[me]
                else:
                    self.similarity_between_layers_per_round[me][server_round] = []
                    self.mean_similarity_per_round[me][server_round] = 0
                    self.similarity_between_layers_per_round_and_client[me][server_round] = []
                    self.df[me] = 1

            print(f"df: {self.df}")

            return parameters_aggregated_mefl, metrics_aggregated_mefl


        except Exception as e:
            print("aggregate_fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, t, parameters_aggregated_mefl):

        try:
            evaluate_results = []
            print("inicio s")
            for me in range(self.ME):
                clients_evaluate_list = []
                metrics = {"fc": self.fc[me], "il": self.il[me], "homogeneity_degree": self.homogeneity_degree[me], "ps": self.ps[me], "similarity": self.similarity[me]}
                for i in range(len(self.clients)):
                    client_dict = {}
                    client_dict["client"] = self.clients[i]
                    client_dict["cid"] = self.clients[i].client_id
                    client_dict["nt"] = t - self.clients[i].lt[me]
                    client_dict["lt"] = self.clients[i].lt[me]
                    clients_evaluate_list.append((self.clients[i], EvaluateIns(ndarrays_to_parameters(parameters_aggregated_mefl[me]), client_dict)))
                print(f"submetidos t: {self.t_hat[me]} T: {self.number_of_rounds} df: {self.df[me]}")
                clients_compressed_parameters = fedpredict_server(
                    global_model_parameters=parameters_aggregated_mefl[me], client_evaluate_list=clients_evaluate_list,
                    t=t, T=self.number_of_rounds, df=self.df[me], compression=self.compression, fl_framework="flwr", k_ratio=0.3)
                for i in range(len(self.clients)):
                    evaluate_results.append(self.clients[i].evaluate(me, t, parameters_to_ndarrays(clients_compressed_parameters[i][1].parameters), metrics))
                    # evaluate_results.append(self.clients[i].evaluate(me, t,
                    #     clients_compressed_parameters[i][1].config["parameters"]))

            loss_aggregated_mefl, metrics_aggregated_mefl = self.aggregate_evaluate(server_round=t,
                                                                                    results=evaluate_results,
                                                                                    failures=[])
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate evaluation losses using weighted average."""
        try:

            print("""inicio aggregate evaluate {}""".format(server_round))

            results_mefl = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                parameters, num_examples, result = results[i]
                me = result[2]["me"]
                results_mefl[me].append(result)


            # Aggregate loss
            # print("""metricas recebidas rodada {}: {}""".format(server_round, results_mefl[0]))
            loss_aggregated_mefl = {me: 0. for me in range(self.ME)}
            for me in results_mefl.keys():
                loss_aggregated = weighted_loss_avg(
                    [
                        (num_examples, loss)
                        for loss, num_examples, metrics in results_mefl[me]
                    ]
                )
                loss_aggregated_mefl[int(me)] = loss_aggregated

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated_mefl = {me: {} for me in range(self.ME)}
            if self.evaluate_metrics_aggregation_fn:
                for me in results_mefl.keys():
                    self.gw[me] = [metrics["gw"] for _, _, metrics in results_mefl[me]]
                    self.lw[me] = [metrics["lw"] for _, _, metrics in results_mefl[me]]
                    eval_metrics = [(num_examples, metrics) for loss, num_examples, metrics in results_mefl[me]]
                    metrics_aggregated_mefl[int(me)] = self.evaluate_metrics_aggregation_fn(eval_metrics)

            mode = "w"

            for me in range(self.ME):
                self.add_metrics(server_round, metrics_aggregated_mefl, me)
                self._save_results(mode, me)


            return loss_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            print("aggregate_evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def add_metrics(self, server_round, metrics_aggregated, me):
        try:
            metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
            metrics_aggregated[me]["# training clients"] = self.n_trained_clients
            metrics_aggregated[me]["training clients and models"] = self.selected_clients_m[me]
            metrics_aggregated[me]["fc"] = self.fc[me]
            metrics_aggregated[me]["il"] = self.il[me]
            metrics_aggregated[me]["dh"] = self.homogeneity_degree[me]
            metrics_aggregated[me]["ps"] = self.ps[me]
            metrics_aggregated[me]["gw"] = self.gw[me]
            metrics_aggregated[me]["lw"] = self.lw[me]

            print("gw lw: ", me, self.gw[me])

            for metric in metrics_aggregated[me]:
                self.results_test_metrics[me][metric].append(metrics_aggregated[me][metric])
        except Exception as e:
            print("add_metrics error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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

            print("arquivo nome: ", file_path)

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
            return round(float(values), 2)

        except Exception as e:
            print("_weighted_average error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))