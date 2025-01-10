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

import time
from functools import reduce
import numpy as np
import torch
from flcore.clients.clientmultifedyogi_global_model_eval_with_fedpredict import clientMultiFedYogiGlobalModelEvalWithFedPredict
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np

from torch.nn.parameter import Parameter

import numpy.typing as npt

from io import BytesIO
from dataclasses import dataclass

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]


class MultiFedYogiGlobalModelEvalWithFedPredict(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMultiFedYogiGlobalModelEvalWithFedPredict)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        eta = 1e-2
        eta_l = 0.0316
        beta_1 = 0.9
        beta_2 = 0.99
        tau = 1e-3
        # self.load_model()
        self.Budget = []
        self.current_weights = None
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_t: Optional[NDArrays] = None
        self.v_t: Optional[NDArrays] = None


    def train(self):
        self._get_models_size()
        for t in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients(t)
            # self.send_models()
            print(self.selected_clients)

            for m in range(len(self.selected_clients)):


                for i in range(len(self.selected_clients[m])):
                    self.clients[self.selected_clients[m][i]].train(m, t, self.global_model[m])

                if t%self.eval_gap == 0:
                    print(f"\n-------------Round number: {t}-------------")
                    print("\nEvaluate global model for ", self.dataset[m])
                    self.evaluate(m, t=t)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and t%self.dlg_gap == 0:
                self.call_dlg(t)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        for m in range(self.M):
            self.save_results(m)
            self.save_global_model(m)

    def aggregate(self, results: List[Tuple[NDArrays, float]], m: int) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples, cid in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer.detach().cpu().numpy() * num_examples for layer in weights.parameters()] for weights, num_examples, cid in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        """Yogi"""

        delta_t: NDArrays = [
            x - y.data.detach().cpu().numpy() for x, y in zip(weights_prime, self.global_model[m].parameters())
        ]

        # m_t
        print("beta: ", self.beta_1, self.beta_2)
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
            for x, y in zip(self.v_t, delta_t)
        ]
        # print("ffl: ", "mt", self.m_t, "vt", self.v_t)
        new_weights = [
            x.data.detach().cpu().numpy() + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.global_model[m].parameters(), self.m_t, self.v_t)
        ]

        # self.current_weights = new_weights

        weights_prime = [Parameter(torch.Tensor(i.tolist())) for i in new_weights]

        return weights_prime

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        for m in range(len(self.uploaded_models)):
            # self.global_model[m] = copy.deepcopy(self.uploaded_models[m][0])
            # for param in self.global_model[m].parameters():
            #     param.data.zero_()
            parameters_tuple = []
            for cid, w, client_model in zip(self.uploaded_ids[m], self.uploaded_weights[m], self.uploaded_models[m]):
                # self.add_parameters(w, client_model, m)
                parameters_tuple.append((client_model, w, cid))

            agg_parameters = self.aggregate(parameters_tuple, m)

            for server_param, client_param in zip(self.global_model[m].parameters(), agg_parameters):
                # print(": ", server_param.data.shape, client_param.data.clone().shape)
                server_param.data = client_param.data.clone()
