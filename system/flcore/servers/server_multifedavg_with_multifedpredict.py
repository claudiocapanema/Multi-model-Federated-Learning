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
from fedpredict import fedpredict_server


class MultiFedAvgWithMultiFedPredict(MultiFedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.model_shape_mefl = [None] * self.ME

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
                clients_compressed_parameters = fedpredict_server(global_model_parameters=parameters_aggregated_mefl[me], client_evaluate_list=clients_evaluate_list,
                                     t=t, T=self.number_of_rounds, df=0, model_shape=self.model_shape_mefl[me],
                                     compression="dls")

                for i in range(len(self.clients)):
                    evaluate_results.append(self.clients[i].evaluate(me, t, clients_compressed_parameters[i]["parameters"]))

            loss_aggregated_mefl, metrics_aggregated_mefl = self.aggregate_evaluate(server_round=t,
                                                                                    results=evaluate_results,
                                                                                    failures=[])
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))