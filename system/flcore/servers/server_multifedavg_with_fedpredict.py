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
from flcore.servers.server_multifedavg_with_multifedpredict import MultiFedAvgWithMultiFedPredict
import sys
from fedpredict import fedpredict_server, fedpredict_layerwise_similarity
import flwr
from flwr.common import (
    EvaluateIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

def get_weights(net):
    try:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    except Exception as e:
        print("get_weights error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


class MultiFedAvgWithFedPredict(MultiFedAvgWithMultiFedPredict):
    def __init__(self, args, times):
        super().__init__(args, times)

    def set_clients(self):

        try:
            client_class = ClientMultiFedAvgWithFedPredict
            for i in range(self.total_clients):
                client = client_class(self.args,
                                id=i,
                                   model=copy.deepcopy(self.global_model))
                self.clients.append(client)

        except Exception as e:
            print("set_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results,
    #     failures,
    # ):
    #     """Aggregate fit results using weighted average."""
    #     try:
    #
    #
    #
    #         parameters_aggregated_mefl, metrics_aggregated_mefl = super().aggregate_fit(server_round, results, failures)
    #         if server_round == 1:
    #             for me in range(self.ME):
    #                 self.model_shape_mefl[me] = [i.shape for i in parameters_aggregated_mefl[me]]
    #
    #         clients_parameters_mefl = {me: [] for me in range(self.ME)}
    #         fc_list = {me: [] for me in range(self.ME)}
    #         il_list = {me: [] for me in range(self.ME)}
    #         similarity_list = {me: [] for me in range(self.ME)}
    #         num_samples_list = {me: [] for me in range(self.ME)}
    #         for i in range(len(results)):
    #             parameter, num_examples, result = results[i]
    #             me = result["me"]
    #             fc = result["non_iid"]["fc"]
    #             il = result["non_iid"]["il"]
    #             similarity = result["non_iid"]["similarity"]
    #             fc_list[me].append(fc)
    #             il_list[me].append(il)
    #             similarity_list[me].append(similarity)
    #             num_samples_list[me].append(num_examples)
    #             clients_parameters_mefl[me].append(results[i][0])
    #         print("verificar")
    #         print(fc_list[0],  num_samples_list[0],  self.fc[0])
    #         for me in range(self.ME):
    #             print(self._weighted_average(fc_list[me], num_samples_list[me]))
    #             self.fc[me] = self._weighted_average(fc_list[me], num_samples_list[me])
    #             self.il[me] = self._weighted_average(il_list[me], num_samples_list[me])
    #             self.similarity[me] = self._weighted_average(il_list[me], num_samples_list[me])
    #             self.homogeneity_degree[me] = (self.fc[me] + (1 - self.il[me])) / 2
    #             # if self.homogeneity_degree[me] > self.prediction_layer[me]["non_iid"]:
    #             if server_round <= 59:
    #                 print(f"Rodada {server_round} substituiu. Novo {self.homogeneity_degree[me]} antigo {self.prediction_layer[me]['non_iid']} diferença: {self.homogeneity_degree[me] - self.prediction_layer[me]["non_iid"]}")
    #                 self.prediction_layer[me]["non_iid"] = self.homogeneity_degree[me]
    #                 self.prediction_layer[me]["parameters"] = parameters_aggregated_mefl[me][-2:]
    #
    #             # parameters_aggregated_mefl[me][-2:] = self.prediction_layer[me]["parameters"]
    #
    #
    #
    #         flag = False
    #         if server_round == 1:
    #             flag = True
    #         print("Flag: ", flag)
    #         for me in range(self.ME):
    #             if "dls" in self.compression:
    #                 if flag:
    #                     self.similarity_between_layers_per_round_and_client[me][server_round], \
    #                     self.similarity_between_layers_per_round[me][server_round], self.mean_similarity_per_round[me][
    #                         server_round], self.similarity_list_per_layer[me], self.df[me] = fedpredict_layerwise_similarity(
    #                         parameters_aggregated_mefl[me], clients_parameters_mefl[me], self.similarity_list_per_layer[me])
    #                 else:
    #                     self.similarity_between_layers_per_round_and_client[me][server_round], \
    #                     self.similarity_between_layers_per_round[me][
    #                         server_round], self.mean_similarity_per_round[me][
    #                         server_round], self.similarity_list_per_layer[me] = \
    #                     self.similarity_between_layers_per_round_and_client[me][server_round - 1], \
    #                     self.similarity_between_layers_per_round[me][
    #                         server_round - 1], self.mean_similarity_per_round[me][
    #                         server_round - 1], self.similarity_list_per_layer[me]
    #             else:
    #                 self.similarity_between_layers_per_round[me][server_round] = []
    #                 self.mean_similarity_per_round[me][server_round] = 0
    #                 self.similarity_between_layers_per_round_and_client[me][server_round] = []
    #                 self.df[me] = 1
    #
    #         print(f"df: {self.df}")
    #
    #         return parameters_aggregated_mefl, metrics_aggregated_mefl
    #
    #
    #     except Exception as e:
    #         print("aggregate_fit error")
    #         print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
    #
    #
    #
    # def evaluate(self, t, parameters_aggregated_mefl):
    #
    #     try:
    #         evaluate_results = []
    #         print("inicio s")
    #         for me in range(self.ME):
    #             clients_evaluate_list = []
    #             metrics = {"fc": self.fc[me], "il": self.il[me], "homogeneity_degree": self.homogeneity_degree[me], "similarity": self.similarity[me]}
    #             for i in range(len(self.clients)):
    #                 client_dict = {}
    #                 client_dict["client"] = self.clients[i]
    #                 client_dict["cid"] = self.clients[i].client_id
    #                 client_dict["nt"] = t - self.clients[i].lt[me]
    #                 client_dict["lt"] = self.clients[i].lt[me]
    #                 clients_evaluate_list.append((self.clients[i], EvaluateIns(ndarrays_to_parameters(parameters_aggregated_mefl[me]), client_dict)))
    #             print(f"submetidos t: {t} T: {self.number_of_rounds} df: {self.df[me]}")
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
    #
    # def _get_results(self, train_test, mode, me):
    #
    #     try:
    #         algo = self.dataset[me] + "_" + self.strategy_name
    #
    #         result_path = self.get_result_path(train_test)
    #
    #         if not os.path.exists(result_path):
    #             os.makedirs(result_path)
    #
    #         compression = self.compression
    #         if len(compression) > 0:
    #             compression = "_" + compression
    #         file_path = result_path + "{}{}.csv".format(algo, compression)
    #
    #         if train_test == 'test':
    #
    #             header = self.test_metrics_names
    #             # print(self.rs_test_acc[me])
    #             # print(self.rs_test_auc[me])
    #             # print(self.rs_train_loss[me])
    #             list_of_metrics = []
    #             for metric in self.results_test_metrics[me]:
    #                 # print(me, len(self.results_test_metrics[me][metric]))
    #                 length = len(self.results_test_metrics[me][metric])
    #                 list_of_metrics.append(self.results_test_metrics[me][metric])
    #
    #             data = []
    #             for i in range(length):
    #                 row = []
    #                 for j in range(len(list_of_metrics)):
    #                     row.append(list_of_metrics[j][i])
    #
    #                 data.append(row)
    #
    #         else:
    #             if mode == '':
    #                 header = self.train_metrics_names
    #                 list_of_metrics = []
    #                 for metric in self.results_train_metrics[me]:
    #                     # print(me, len(self.results_train_metrics[me][metric]))
    #                     length = len(self.results_train_metrics[me][metric])
    #                     list_of_metrics.append(self.results_train_metrics[me][metric])
    #
    #                 data = []
    #                 # print("""tamanho: {}    {}""".format(length, list_of_metrics))
    #                 for i in range(length):
    #                     row = []
    #                     for j in range(len(list_of_metrics)):
    #                         if len(list_of_metrics[j]) > 0:
    #                             row.append(list_of_metrics[j][i])
    #                         else:
    #                             row.append(0)
    #
    #                     data.append(row)
    #
    #         # print("File path: " + file_path)
    #         print(data)
    #
    #         return file_path, header, data
    #     except Exception as e:
    #         print("get_results error")
    #         print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
    #
    # def _weighted_average(self, values, weights):
    #
    #     try:
    #         values = np.array([i * j for i, j in zip(values, weights)])
    #         values = np.sum(values) / np.sum(weights)
    #         return float(values)
    #
    #     except Exception as e:
    #         print("_weighted_average error")
    #         print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate fit results using weighted average."""
        try:

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

            for me in trained_models:
                # Convert results
                weights_results = [
                    (parameters, num_examples)
                    for parameters, num_examples, fit_res in results_mefl[me]
                ]
                aggregated_ndarrays_mefl[me] = aggregate(weights_results)
                if len(weights_results) > 1:
                    aggregated_ndarrays_mefl[me] = aggregate(weights_results)
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

            return self.parameters_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            print("aggregate_fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, t, parameters_aggregated_mefl):

        try:
            evaluate_results = []
            print("inicio s")
            for me in range(self.ME):
                clients_evaluate_list = []
                metrics = {"fc": self.fc[me], "il": self.il[me], "homogeneity_degree": self.homogeneity_degree[me], "ps": self.ps[me], "similarity": self.similarity[me],}
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