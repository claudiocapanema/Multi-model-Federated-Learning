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
from flcore.clients.client_multifedavg_with_multifedpredict import ClientMultiFedAvgWithMultiFedPredict
from flcore.clients.client_multifedavg_with_fedpredict import ClientMultiFedAvgWithFedPredict
from flcore.clients.client_multifedavg_with_fedpredict_dynamic import ClientMultiFedAvgWithFedPredictDynamic
from flcore.servers.server_multifedavg_with_multifedpredict_v0 import MultiFedAvgWithMultiFedPredictv0
import sys
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
        return {"Accuracy": sum(accuracies) / sum(examples), "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
                "Loss": sum(loss) / sum(examples), "Round (t)": metrics[0][1]["Round (t)"], "Model size": metrics[0][1]["Model size"]}
    except Exception as e:
        print("weighted_average_fit error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


class MultiFedAvgWithMultiFedPredict(MultiFedAvgWithMultiFedPredictv0):
    def __init__(self, args, times, version):
        super().__init__(args, times)
        self.t_hat = [1] * self.ME
        self.reduced_training_intensity_flag = [False] * self.ME
        self.train_accuracy_list = {me: [] for me in range(self.ME)}
        self.max_number_of_rounds_data_drift_adaptation = len(self.clients) //  self.num_training_clients
        self.increased_training_intensity = [0] * self.ME
        self.reduced_training_intensity_flag = [False] * self.ME
        self.last_round_increased_training_intensity = [0] * self.ME
        self.version = version
        # self.clients_ids = [i.client_id for i in self.clients]
        # self.clients_ids_uniform_selection = dict(copy.deepcopy(self.clients_ids))
        self.train_losses = {me: [] for me in range(self.ME)}
        self.fit_metrics_aggregation_fn = weighted_average_fit
        self.drift_flag = {me: [] for me in range(self.ME)}
        self.reduced = {me: [] for me in range(self.ME)}
        self.reduction_fraction_list = {me: [] for me in range(self.ME)}
        self.ps_list = {me: [] for me in range(self.ME)}

    def set_clients(self):

        try:
            client_class = ClientMultiFedAvgWithMultiFedPredict
            for i in range(self.total_clients):
                client = client_class(self.args,
                                id=i,
                                   model=copy.deepcopy(self.global_model))
                self.clients.append(client)

            self.clients_ids = [i.client_id for i in self.clients]
            self.clients_ids_uniform_selection = [i for i in copy.deepcopy(self.clients_ids)]

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
            self.drift_flag = {me: [] for me in range(self.ME)}
            self.reduced = {me: [] for me in range(self.ME)}

            results_mefl = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                parameter, num_examples, result = results[i]
                me = result["me"]
                if me not in trained_models:
                    trained_models.append(me)
                client_id = result["client_id"]
                self.selected_clients_m[me].append(client_id)
                results_mefl[me].append(results[i])
                self.drift_flag[me].append(result["non_iid"]["drift"])
                self.reduced[me].append(result["non_iid"]["reduced"])

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
                aggregated_ndarrays_mefl[me] = self.aggregate(weights_results, self.homogeneity_degree[me], self.parameters_aggregated_mefl[me], server_round)
                if len(weights_results) > 1:
                    aggregated_ndarrays_mefl[me] = self.aggregate(weights_results, self.homogeneity_degree[me], self.parameters_aggregated_mefl[me], server_round)
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
                    self.train_losses[me].append(metrics_aggregated_mefl[me]["Loss"])
                    print(f"Teste data shift modelo {me} rodada {server_round} teste {self.drift_flag[me]} reduced {self.reduced[me]}")
                else:
                    print("nao tem")

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
                alpha = result["alpha"]
                me = result["me"]
                client_id = result["client_id"]
                fc = result["non_iid"]["fc"]
                il = result["non_iid"]["il"]
                ps = result["non_iid"]["ps"]
                similarity = result["non_iid"]["similarity"]
                self.client_metrics[client_id][me][alpha]["fc"] = fc
                self.client_metrics[client_id][me][alpha]["il"] = il
                self.client_metrics[client_id][me][alpha]["similarity"] = similarity
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

    def aggregate(self, results: list[tuple[NDArrays, int]], homogeneity_degree: float,
                  current_parameters: list[tuple[NDArrays, int]], t: int) -> NDArrays:
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

            if self.version in ["iti"]:
                homogeneity_degree = 1
            weighted_weights = [np.array(original_layer + homogeneity_degree * layer) for original_layer, layer in
                                zip(current_parameters, weights_prime)]
            return weighted_weights
        except Exception as e:
            print("aggregate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def detect_drift_ks(self, losses, window=5, alpha=0.05):

        try:
            """
            Detecta drift usando teste de Kolmogorov-Smirnov.
            Apenas aumentos são considerados drift.

            Args:
                losses (list[float]): histórico de valores de loss.
                window (int): tamanho da janela de comparação.
                alpha (float): nível de significância.

            Returns:
                bool: True se houve drift, False caso contrário.
            """
            if len(losses) < window + 1:
                return False  # histórico insuficiente

            history = np.array(losses[-(window + 1):-1])
            last = np.array([losses[-1]] * len(history))  # simula distribuição do último valor

            stat, p_value = ks_2samp(history, last)

            # KS detecta diferença significativa (p < alpha)
            # mas só consideramos se for aumento
            return (p_value < alpha) and (losses[-1] > history.mean())

        except Exception as e:
            print("detect_drift_ks client error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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
            g.manual_seed(t)
            random.seed(t)
            np.random.seed(t)
            torch.manual_seed(t)

            if self.version in ["dh"]:
                return super().select_clients(t)

            data_drift_model = -1
            #  or self.drift_flag or self.reduced[me].count(True) > self.reduced[me].count(False)
            drift_degree = [0] * self.ME
            drift_ps = [0] * self.ME
            for me in range(self.ME):
                # step 1
                if len(self.reduced[me]) > 0:
                    # reduction_fraction = self.reduced[me].count(True) / len(self.reduced[me])
                    n_clients_reduced = self.reduced[me].count(True)
                    n_clients = len(self.reduced[me])
                    reduction_probability = self.binomial(n_clients_reduced, n_clients)
                    reduction_fraction_list = copy.deepcopy(self.reduction_fraction_list[me])
                    reduction_fraction_list.append(reduction_probability)
                    if self.detect_drift_ks(reduction_fraction_list, window=5, alpha=0.1):
                        drift_degree[me] = reduction_probability
                    else:
                        drift_degree[me] = 0
                        self.reduction_fraction_list[me].append(reduction_probability)
                    self.reduction_fraction_list[me].append(reduction_probability)
                # step 2
                if len(self.ps_list[me]) == 0:
                    self.ps_list[me].append(self.ps[me])
                else:
                    ps_list = copy.deepcopy(self.ps_list[me])
                    ps_list.append(self.ps[me])
                    if self.detect_drift_ks(ps_list, window=5, alpha=0.1):
                        drift_ps[me] = self.ps[me]
                    else:
                        self.ps_list[me].append(self.ps[me])
                        # drift_ps[me] = 0
                        drift_ps[me] = self.ps[me]
                #  or t in {0: [10, 40, 70], 1: [20, 50, 80], 2: [30, 60, 90]}[me]
                print(f"Rodada {t} modelo {me} resultado drift degree = {drift_degree[me]} ps = {drift_ps[me]}")
                if (drift_degree[me] >= 0.5 or self.ps[me] > 0) or self.increased_training_intensity[me] > 0:
                        data_drift_model = me
                        if drift_degree[me] > 0.5 and drift_ps[me] > 0:
                            data_shift_untrained_clients = (self.total_clients - len(self.reduced[me])) * drift_degree[me]
                            rounds_needed = data_shift_untrained_clients // self.num_training_clients
                            self.max_number_of_rounds_data_drift_adaptation = rounds_needed


            print(f"##rodada {t} data_drift_model = {data_drift_model} drift_ps {drift_ps} drift degree = {drift_degree}")
            if data_drift_model > -1:
                self.increased_training_intensity[data_drift_model] += 1

                # selected_clients = list(np.random.choice(self.clients, self.num_training_clients, replace=False))
                # selected_clients = [i.client_id for i in selected_clients]
                to_remove = [i for i in sorted(random.sample(list(self.clients_ids_uniform_selection), self.num_training_clients))]
                self.clients_ids_uniform_selection = [x for x in self.clients_ids_uniform_selection if x not in to_remove]
                selected_clients = list(to_remove)
                if self.increased_training_intensity[data_drift_model] == self.max_number_of_rounds_data_drift_adaptation:
                    self.increased_training_intensity[data_drift_model] = 0
                    self.clients_ids_uniform_selection = [i for i in copy.deepcopy(self.clients_ids)]
            else:
                sc = super().select_clients(t)

                for me in range(self.ME):
                    if len(sc[me]) > 0:
                        self.t_hat[me] = t
                        self.reduced_training_intensity_flag[me] = False
                    elif len(sc[me]) == 0 and not self.reduced_training_intensity_flag[me]:
                        self.t_hat[me] = t - 1
                        # self.t_hat[me] = t
                        self.reduced_training_intensity_flag[me] = True

                return sc

            sc = []
            for me in range(self.ME):
                if me == data_drift_model:
                    sc.append(selected_clients)
                else:
                    sc.append([])

            for me in range(self.ME):
                if len(sc[me]) > 0:
                    self.t_hat[me] = t
                    self.reduced_training_intensity_flag[me] = False
                elif len(sc[me]) == 0 and not self.reduced_training_intensity_flag[me]:
                    self.t_hat[me] = t - 1
                    self.reduced_training_intensity_flag[me] = True

            self.n_trained_clients = sum([len(i) for i in sc])

            return sc

        except Exception as e:
            print("select_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, t, parameters_aggregated_mefl): # v0

        try:
            evaluate_results = []
            print("inicio s")
            for me in range(self.ME):
                clients_evaluate_list = []
                freeze = True if len(self.selected_clients_m[me]) == 0 else False
                metrics = {"fc": self.fc[me], "il": self.il[me], "homogeneity_degree": self.homogeneity_degree[me], "ps": self.ps[me], "similarity": self.similarity[me], "freeze": freeze, "freeze_round": self.t_hat[me]}
                for i in range(len(self.clients)):
                    client_dict = {}
                    client_dict["client"] = self.clients[i]
                    client_dict["cid"] = self.clients[i].client_id
                    # client_dict["nt"] = self.t_hat[me] - self.clients[i].lt[me]
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

    # def evaluate(self, t, parameters_aggregated_mefl):
    #
    #     try:
    #         evaluate_results = []
    #         print("inicio s")
    #         for me in range(self.ME):
    #             clients_evaluate_list = []
    #             metrics = {"fc": self.fc[me], "il": self.il[me], "homogeneity_degree": self.homogeneity_degree[me], "ps": self.ps[me], "similarity": self.similarity[me]}
    #             for i in range(len(self.clients)):
    #                 client_dict = {}
    #                 client_dict["client"] = self.clients[i]
    #                 client_dict["cid"] = self.clients[i].client_id
    #                 client_dict["nt"] = t - self.clients[i].lt[me]
    #                 client_dict["lt"] = self.clients[i].lt[me]
    #                 clients_evaluate_list.append((self.clients[i], EvaluateIns(ndarrays_to_parameters(parameters_aggregated_mefl[me]), client_dict)))
    #             print(f"submetidos t: {self.t_hat[me]} T: {self.number_of_rounds} df: {self.df[me]}")
    #             clients_compressed_parameters = fedpredict_server(
    #                 global_model_parameters=parameters_aggregated_mefl[me], client_evaluate_list=clients_evaluate_list,
    #                 t=t, T=self.number_of_rounds, df=self.df[me], compression=self.compression, fl_framework="flwr", k_ratio=0.3)
    #             for i in range(len(self.clients)):
    #                 evaluate_results.append(self.clients[i].evaluate(me, t, parameters_to_ndarrays(clients_compressed_parameters[i][1].parameters), metrics))
    #                 # evaluate_results.append(self.clients[i].evaluate(me, t,
    #                 #     clients_compressed_parameters[i][1].config["parameters"]))
    #
    #         loss_aggregated_mefl, metrics_aggregated_mefl = self.aggregate_evaluate(server_round=t,
    #                                                                                 results=evaluate_results,
    #                                                                                 failures=[])
    #     except Exception as e:
    #         print("evaluate error")
    #         print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    # def add_metrics(self, server_round, metrics_aggregated, me):
    #     try:
    #         metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
    #         metrics_aggregated[me]["# training clients"] = self.n_trained_clients
    #         metrics_aggregated[me]["training clients and models"] = self.selected_clients_m[me]
    #         metrics_aggregated[me]["fc"] = self.fc[me]
    #         metrics_aggregated[me]["il"] = self.il[me]
    #         metrics_aggregated[me]["dh"] = self.homogeneity_degree[me]
    #         metrics_aggregated[me]["ps"] = self.ps[me]
    #
    #         for metric in metrics_aggregated[me]:
    #             self.results_test_metrics[me][metric].append(metrics_aggregated[me][metric])
    #     except Exception as e:
    #         print("add_metrics error")
    #         print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _save_data_metrics(self):

        try:
            for me in range(self.ME):
                algo = self.dataset[me] + "_" + self.strategy_name
                result_path = self.get_result_path("test")
                file_path = result_path + "{}_metrics.csv".format(algo)
                rows = []
                head = ["cid", "me", "Alpha", "fc", "il", "ps", "dh"]
                self._write_header(file_path, head, mode='w')
                for cid in range(0, self.total_clients):
                    for alpha in [0.1, 1.0, 10.0]:
                        fc = self.client_metrics[cid][me][alpha]["fc"]
                        il = self.client_metrics[cid][me][alpha]["il"]
                        if fc is not None and il is not None:
                            dh = (fc + (1 - il)) / 2
                        else:
                            dh = None
                        row = [cid, me, alpha, fc, il, self.client_metrics[cid][me][alpha]["similarity"], dh]
                        rows.append(row)

                self._write_outputs(file_path, rows)

                print(f"rows {rows}")

        except Exception as e:
            print("_save_data_metrics error")
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