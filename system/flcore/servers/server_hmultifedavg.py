# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import copy
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
import sys
import math
import numpy as np
import random
import os

import pandas as pd
from flcore.clients.client_hmultifedavg import HMultiFedAvgClient, MultiFedAvgClient
from flcore.servers.server_multifedavg import MultiFedAvg
from threading import Thread

from functools import reduce
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

import torch
from torch.nn.parameter import Parameter

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

import logging
logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]


class HMultiFedAvg(MultiFedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.homogeneity_degree = [None] * self.ME
        self.fc = [None] * self.ME
        self.il = [None] * self.ME
        self.alternated_model_index = None
        self.clients_non_iid_degree()

    def set_clients(self):

        try:
            for i in range(self.total_clients):
                client = HMultiFedAvgClient(self.args, id=i, model=copy.deepcopy(self.global_model))
                self.clients.append(client)

        except Exception as e:
            print("set_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def select_clients(self, t):

        try:
            g = torch.Generator()
            g.manual_seed(t)
            random.seed(t)
            np.random.seed(t)
            torch.manual_seed(t)
            selected_clients = list(np.random.choice(self.clients, self.num_training_clients, replace=False))
            selected_clients = [i.client_id for i in selected_clients]

            if self.experiment_id == 1:
                middle = int(self.number_of_rounds * 0.5)
            elif self.experiment_id == 2:
                middle = int(1)
            elif self.experiment_id == 3:
                middle = int(self.number_of_rounds * 0.3)
            elif self.experiment_id == 4:
                middle = int(self.number_of_rounds * 0.7)
            heterogeneous_models = np.argwhere(self.homogeneity_degree <= 0.36).flatten()
            homogeneous_models = np.argwhere(self.homogeneity_degree > 0.36).flatten()
            equal_number_of_clients = int(len(selected_clients) / self.ME)
            if t >= middle and len(heterogeneous_models) > 0:
                if self.alternated_model_index is None:
                    self.alternated_model_index = 0
                else:
                    self.alternated_model_index += 1
                    if self.alternated_model_index >= len(heterogeneous_models):
                        self.alternated_model_index = 0

                training_intensity = [0] * self.ME
                for me in homogeneous_models:
                    training_intensity[me] = equal_number_of_clients

                if self.alternated_model_index is not None:
                    me = heterogeneous_models[self.alternated_model_index]
                    print("antes: ", equal_number_of_clients, len(heterogeneous_models), me, heterogeneous_models, training_intensity)
                    training_intensity[me] = int(equal_number_of_clients * len(heterogeneous_models))


                sc = []
                i = 0
                for me in range(self.ME):

                    n_clients = training_intensity[me]
                    if n_clients > 0:
                        sc.append(selected_clients[i: i + n_clients])
                    else:
                        sc.append([])
            else:
                n = len(selected_clients) // self.ME
                sc = np.array_split(selected_clients, self.ME)

            self.n_trained_clients = sum([len(i) for i in sc])

            return sc

        except Exception as e:
            print("select_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def clients_non_iid_degree(self):

        num_samples = {me: [] for me in range(self.ME)}
        fc = {me: [] for me in range(self.ME)}
        il = {me: [] for me in range(self.ME)}
        for client in self.clients:

            for me in range(self.ME):

                num_samples[me].append(client.num_examples[me])
                fc[me].append(client.fc_ME[me])
                il[me].append(client.il_ME[me])

        for me in range(self.ME):
            fc[me] = self._weighted_average(fc[me], num_samples[me])
            il[me] = self._weighted_average(il[me], num_samples[me])

            self.homogeneity_degree[me] = (fc[me] + (1 - il[me])) / 2
            self.fc[me] = fc[me]
            self.il[me] = il[me]
        print(f"fc {fc} il {il}  homogeneity degree {self.homogeneity_degree}")
        self.homogeneity_degree = np.array(self.homogeneity_degree)

    def _weighted_average(self, values, weights):

        try:
            values = np.array([i * j for i, j in zip(values, weights)])
            values = np.sum(values) / np.sum(weights)
            return float(values)

        except Exception as e:
            logger.error("_weighted_average error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
