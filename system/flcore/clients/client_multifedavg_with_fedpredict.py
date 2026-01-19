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
import copy
import torch
import numpy as np
import time
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import sys
from flcore.clients.client_multifedavg_with_multifedpredict import ClientMultiFedAvgWithMultiFedPredict
from fedpredict import fedpredict_client_torch
from .utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
from numpy.linalg import norm
import pickle

def cosine_similarity(p_1, p_2):

    # compute cosine similarity
    try:
        p_1_size = np.array(p_1).shape
        p_2_size = np.array(p_2).shape
        if p_1_size != p_2_size:
            raise Exception(f"Input sizes have different shapes: {p_1_size} and {p_2_size}. Please check your input data.")

        return np.dot(p_1, p_2) / (norm(p_1) * norm(p_2))
    except Exception as e:
        print("cosine_similairty error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class ClientMultiFedAvgWithFedPredict(ClientMultiFedAvgWithMultiFedPredict):
    def __init__(self, args, id, model, fold_id):
        try:
            super().__init__(args, id,  model, fold_id)
        except Exception as e:
            print("__init__ error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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

    def evaluate(self, me, t, global_model, metrics):
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
            combined_model = fedpredict_client_torch(local_model=self.model[me], global_model=global_model,
                                                     t=t, T=self.T, nt=nt, device=self.device,
                                                     global_model_original_shape=self.model_shape_mefl[me])
            loss, metrics = test(combined_model, self.valloader[me], self.device, self.client_id, t,
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
