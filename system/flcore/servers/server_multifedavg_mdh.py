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
from flcore.clients.client_multifedavgmdh import MultiFedAvgMDHClient, MultiFedAvgClient
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


class MultiFedAvgMDH(MultiFedAvg):
    def __init__(self, args, times, fold_id):
        super().__init__(args, times, fold_id)

        self.homogeneity_degree = [None] * self.ME
        self.fc = [None] * self.ME
        self.il = [None] * self.ME

        # Maximum number of heterogeneous models that can be
        # trained simultaneously.
        #
        # Q_MAX = 1 reproduces the original MultiFedAvg-MDH.
        # Q_MAX >= h allows all heterogeneous models to be trained
        # in the same round.
        self.q_max = max(1, int(getattr(args, "q_max", 1)))

        self.alternated_model_index = None

        self.clients_non_iid_degree()

        print(f"MultiFedAvg-MDH Q_MAX = {self.q_max}")

    def set_clients(self):

        try:
            for i in range(self.total_clients):
                client = MultiFedAvgMDHClient(self.args, id=i, model=copy.deepcopy(self.global_model), fold_id=self.fold_id)
                self.clients.append(client)

        except Exception as e:
            print("set_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def select_clients(self, t):

        try:
            # ---------------------------------------------------------
            # 1. Deterministic client sampling
            # ---------------------------------------------------------
            g = torch.Generator()
            g.manual_seed(t)

            random.seed(t)
            np.random.seed(t)
            torch.manual_seed(t)

            selected_clients = list(
                np.random.choice(
                    self.clients,
                    self.num_training_clients,
                    replace=False
                )
            )

            selected_clients = [
                client.client_id
                for client in selected_clients
            ]

            # ---------------------------------------------------------
            # 2. Identify heterogeneous and homogeneous models
            # ---------------------------------------------------------
            threshold = 0.36

            heterogeneous_models = (
                np.argwhere(
                    self.homogeneity_degree <= threshold
                ).flatten().tolist()
            )

            homogeneous_models = (
                np.argwhere(
                    self.homogeneity_degree > threshold
                ).flatten().tolist()
            )

            # ---------------------------------------------------------
            # 3. Standard MultiFedAvg allocation
            #
            # This branch is preserved when:
            #   - MDH alternation has not started yet
            #   - no heterogeneous models exist
            # ---------------------------------------------------------
            if t < 50 or len(heterogeneous_models) == 0:
                sc = np.array_split(
                    selected_clients,
                    self.ME
                )

                sc = [
                    list(client_group)
                    for client_group in sc
                ]

                self.n_trained_clients = sum(
                    len(client_group)
                    for client_group in sc
                )

                return sc

            # ---------------------------------------------------------
            # 4. Determine the number of heterogeneous models
            #    trained simultaneously
            # ---------------------------------------------------------
            h = len(heterogeneous_models)

            q = min(h, self.q_max)

            # ---------------------------------------------------------
            # 5. Determine which heterogeneous models are active
            #
            # The active set rotates deterministically.
            #
            # Example:
            #
            # h = 10, q = 3
            #
            # Round 50 -> [0, 1, 2]
            # Round 51 -> [3, 4, 5]
            # Round 52 -> [6, 7, 8]
            # Round 53 -> [9, 0, 1]
            # ...
            # ---------------------------------------------------------
            start_position = ((t - 50) * q) % h

            active_heterogeneous_models = []

            for offset in range(q):
                position = (start_position + offset) % h
                active_heterogeneous_models.append(
                    heterogeneous_models[position]
                )

            # ---------------------------------------------------------
            # 6. Preserve the total number of selected clients
            #
            # The original MDH defines the total budget associated
            # with heterogeneous models as:
            #
            #       budget = h * C / ME
            #
            # We preserve this principle.
            #
            # When q models are active, this heterogeneous budget
            # is divided among them.
            # ---------------------------------------------------------
            total_selected_clients = len(selected_clients)

            heterogeneous_budget = int(
                round(
                    h
                    * total_selected_clients
                    / self.ME
                )
            )

            # Safety bounds
            heterogeneous_budget = max(
                0,
                min(
                    heterogeneous_budget,
                    total_selected_clients
                )
            )

            homogeneous_budget = (
                    total_selected_clients
                    - heterogeneous_budget
            )

            # ---------------------------------------------------------
            # 7. Determine client counts for heterogeneous models
            #
            # The budget is distributed as evenly as possible
            # among q active heterogeneous models.
            # ---------------------------------------------------------
            if q > 0:

                base_heterogeneous = (
                        heterogeneous_budget // q
                )

                heterogeneous_remainder = (
                        heterogeneous_budget % q
                )

                heterogeneous_client_counts = []

                for i in range(q):

                    n_clients = base_heterogeneous

                    if i < heterogeneous_remainder:
                        n_clients += 1

                    heterogeneous_client_counts.append(
                        n_clients
                    )

            else:

                heterogeneous_client_counts = []

            # ---------------------------------------------------------
            # 8. Determine client counts for homogeneous models
            #
            # The remaining client budget is distributed as evenly
            # as possible among homogeneous models.
            # ---------------------------------------------------------
            number_homogeneous = len(
                homogeneous_models
            )

            if number_homogeneous > 0:

                base_homogeneous = (
                        homogeneous_budget
                        // number_homogeneous
                )

                homogeneous_remainder = (
                        homogeneous_budget
                        % number_homogeneous
                )

                homogeneous_client_counts = {}

                for i, me in enumerate(
                        homogeneous_models
                ):

                    n_clients = base_homogeneous

                    if i < homogeneous_remainder:
                        n_clients += 1

                    homogeneous_client_counts[me] = (
                        n_clients
                    )

            else:

                homogeneous_client_counts = {}

            # ---------------------------------------------------------
            # 9. Build the allocation
            # ---------------------------------------------------------
            training_intensity = [0] * self.ME

            # Homogeneous models
            for me in homogeneous_models:
                training_intensity[me] = (
                    homogeneous_client_counts[me]
                )

            # Active heterogeneous models
            for i, me in enumerate(
                    active_heterogeneous_models
            ):
                training_intensity[me] = (
                    heterogeneous_client_counts[i]
                )

            # ---------------------------------------------------------
            # 10. Assign unique clients sequentially
            #
            # Important:
            # Unlike the previous implementation, the pointer i
            # is incremented after every allocation so that the
            # same selected client is never assigned to multiple
            # models in the same round.
            # ---------------------------------------------------------
            sc = [[] for _ in range(self.ME)]

            client_pointer = 0

            # Keep model order deterministic
            for me in range(self.ME):

                n_clients = training_intensity[me]

                if n_clients <= 0:
                    continue

                client_group = selected_clients[
                    client_pointer:
                    client_pointer + n_clients
                ]

                sc[me] = list(client_group)

                client_pointer += n_clients

            # ---------------------------------------------------------
            # 11. Sanity check
            # ---------------------------------------------------------
            self.n_trained_clients = sum(
                len(client_group)
                for client_group in sc
            )

            if self.n_trained_clients != total_selected_clients:
                raise RuntimeError(
                    "Invalid MultiFedAvg-MDH client allocation: "
                    f"expected {total_selected_clients} clients, "
                    f"but allocated {self.n_trained_clients}."
                )

            # ---------------------------------------------------------
            # 12. Logging
            # ---------------------------------------------------------
            print(
                f"[Round {t}] "
                f"heterogeneous={heterogeneous_models}, "
                f"active={active_heterogeneous_models}, "
                f"Q_MAX={self.q_max}, "
                f"training_intensity={training_intensity}"
            )

            return sc

        except Exception as e:

            print("select_clients error")

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            raise

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
