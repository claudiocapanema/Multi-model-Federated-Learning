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

import copy
import time
import numpy as np
from flcore.clients.client_multifedavg_with_multifedpredict import ClientMultiFedAvgWithMultiFedPredict
from flcore.servers.server_multifedavg import MultiFedAvg
import sys
from fedpredict import fedpredict_server, fedpredict_layerwise_similarity


class MultiFedAvgWithMultiFedPredict(MultiFedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.compression = "dls_compredict"
        self.similarity_list_per_layer = {me: {} for me in range(self.ME)}
        self.initial_similarity = 0
        self.current_similarity = 0
        self.model_shape_mefl = [None] * self.ME
        self.similarity_between_layers_per_round = {me: {} for me in range(self.ME)}
        self.similarity_between_layers_per_round_and_client = {me: {} for me in range(self.ME)}
        self.mean_similarity_per_round = {me: {} for me in range(self.ME)}
        self.df = [0] * self.ME

    def set_clients(self):

        try:
            for i in range(self.total_clients):
                client = ClientMultiFedAvgWithMultiFedPredict(self.args,
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



            parameters_aggregated_mefl, metrics_aggregated_mefl = super().aggregate_fit(server_round, results, failures)
            if server_round == 1:
                for me in range(self.ME):
                    self.model_shape_mefl[me] = [i.shape for i in parameters_aggregated_mefl[me]]

            clients_parameters_mefl = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                parameter, num_examples, result = results[i]
                me = result["me"]
                clients_parameters_mefl[me].append(results[i][0])

            flag = False
            if server_round == 1:
                flag = True
            print("Flag: ", flag)
            for me in range(self.ME):
                if "dls" in self.compression:
                    if flag:
                        self.similarity_between_layers_per_round_and_client[me][server_round], \
                        self.similarity_between_layers_per_round[me][server_round], self.mean_similarity_per_round[me][
                            server_round], self.similarity_list_per_layer[me] = fedpredict_layerwise_similarity(
                            parameters_aggregated_mefl[me], clients_parameters_mefl[me], self.similarity_list_per_layer[me])
                        self.df[me] = float(max(0, abs(np.mean(self.similarity_list_per_layer[me][0]) - np.mean(
                            self.similarity_list_per_layer[me][len(parameters_aggregated_mefl[me]) - 2]))))
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
            for me in range(self.ME):
                clients_evaluate_list = []
                for i in range(len(self.clients)):
                    client_dict = {}
                    client_dict["client"] = self.clients[i]
                    client_dict["cid"] = self.clients[i].client_id
                    client_dict["nt"] = t - self.clients[i].lt[me]
                    client_dict["lt"] = self.clients[i].lt[me]
                    clients_evaluate_list.append(client_dict)
                clients_compressed_parameters = fedpredict_server(
                    global_model_parameters=parameters_aggregated_mefl[me], client_evaluate_list=clients_evaluate_list,
                    t=t, T=self.number_of_rounds, df=self.df[me], compression="dls_compredict")

                for i in range(len(self.clients)):
                    evaluate_results.append(self.clients[i].evaluate(me, t, clients_compressed_parameters[i]["parameters"]))

            loss_aggregated_mefl, metrics_aggregated_mefl = self.aggregate_evaluate(server_round=t,
                                                                                    results=evaluate_results,
                                                                                    failures=[])
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))