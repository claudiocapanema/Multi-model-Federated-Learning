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
import sys
import time
import numpy as np
from flcore.servers.server_multifedavg import MultiFedAvg
from threading import Thread
import torch
import torch.nn as nn
from flcore.clients.client_adaptive_fedavg import AdaptiveFedAvgClient
import copy

NUM_CLIENTS = 10
PARTICIPATION_RATE = 0.3  # 30%
CLIENTS_PER_ROUND = max(1, int(NUM_CLIENTS * PARTICIPATION_RATE))
NUM_ROUNDS = 20            # total FL rounds (increase for fuller sim)
LOCAL_EPOCHS = 1            # epochs per client update
BATCH_SIZE = 32
LR_SERVER_INIT = 1       # eta0 from paper
LR_DECAY = 0.99             # d (server-side decay per round)
L_EST = 20                  # number of forward-pass samples for LR estimation per client
BETA1 = 0.7                 # EMA decay for mean loss
BETA2 = 0.3                 # EMA decay for variance
BETA3 = 0.9                 # EMA decay for variance ratio
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose drift type and parameters
DRIFT_TYPE = "sudden"       # "sudden" or "incremental" or None
DRIFT_ROUND = 30            # r0: when drift starts
# sudden drift parameters (swap pairs)
CLASS_SWAP_PAIRS = [(0, 1), (2, 3)]
# incremental drift parameters
TD = 20                     # duration of incremental drift
SIGMA_MAX = 5.0             # gaussian blur max sigma

# For reproducibility
SEED = 42


class AdaptiveFedAvg(MultiFedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.lr_dict = {'EMNIST': 0.01,
                        'MNIST': 0.01,
                        'CIFAR10': 0.01,
                        'GTSRB': 0.01,
                        'WISDM-W': 0.001,
                        'WISDM-P': 0.001,
                        'ImageNet100': 0.01,
                        'ImageNet': 0.1,
                        'ImageNet10': 0.01,
                        "ImageNet_v2": 0.01,
                        "Gowalla": 0.001,
                        "wikitext": 0.001}

    def set_clients(self):

        try:
            for i in range(self.total_clients):
                client = AdaptiveFedAvgClient(self.args,
                                              id=i,
                                              model=copy.deepcopy(self.global_model))
                self.clients.append(client)

        except Exception as e:
            print("set_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def update_learning_rate(self, t, me):

        try:
            eta0 = self.lr_dict[self.args.dataset[me]]
            eta_r = eta0 * (LR_DECAY ** (t - 1))
            if eta_r > eta0:  # just in case
                eta_r = eta0
            for cid in range(self.total_clients):
                # Prepare client dataset with drift applied according to current round and selected DRIFT_TYPE
                label_map = None
                blur_sigma = 0.0

                # if DRIFT_TYPE == "sudden" and t >= DRIFT_ROUND:
                #     # apply class-swap mapping to ALL client data permanently (as in paper)
                #     # build mapping dict once: swap pairs
                #     label_map = {}
                #     for a, b in CLASS_SWAP_PAIRS:
                #         label_map[a] = b
                #         label_map[b] = a
                # elif DRIFT_TYPE == "incremental" and t >= DRIFT_ROUND:
                #     # compute sigma increasing linearly from 0 to SIGMA_MAX over TD rounds
                #     t_since = t - DRIFT_ROUND
                #     if t_since >= TD:
                #         blur_sigma = SIGMA_MAX
                #     else:
                #         blur_sigma = (t_since / TD) * SIGMA_MAX

                l_k = self.metrics_aggregated_mefl[me]["Loss"]

                # Update EMAs for client cid following equations in paper
                self.clients[cid].rounds_seen[me] += 1
                # Mr = lr * (1 - beta1) + Mr-1 * beta1
                self.clients[cid].M[me] = l_k * (1.0 - BETA1) + self.clients[cid].M[me] * BETA1
                # bias-corrected M_hat = M / (1 - beta1^r)
                denom1 = (1 - (BETA1 ** self.clients[cid].rounds_seen[me])) if self.clients[cid].rounds_seen[me] > 0 else 1.0
                M_hat = self.clients[cid].M[me] / (denom1 + 1e-12)

                # Vr = (lr - M_{r-1})^2 * (1 - beta2) + Vr-1 * beta2
                # note: paper uses Mr-1 inside; using M_hat_prev to follow bias-corrected prev
                prev_M = self.clients[cid].M_hat_prev[me] if self.clients[cid].rounds_seen[me] > 1 else 0.0
                self.clients[cid].V[me] = ((l_k - prev_M) ** 2) * (1.0 - BETA2) + self.clients[cid].V[me] * BETA2
                denom2 = (1 - (BETA2 ** self.clients[cid].rounds_seen[me])) if self.clients[cid].rounds_seen[me] > 0 else 1.0
                V_hat = self.clients[cid].V[me] / (denom2 + 1e-12)

                # Rr = (V_hat / V_hat_prev) * (1 - beta3) + R_{r-1} * beta3  (if V_hat_prev == 0 use 1*(1-beta3)+R_prev*beta3)
                if self.clients[cid].V_hat_prev[me] == 0.0:
                    ratio = 1.0
                else:
                    ratio = (V_hat / (self.clients[cid].V_hat_prev[me] + 1e-12))
                self.clients[cid].R[me] = ratio * (1.0 - BETA3) + self.clients[cid].R[me] * BETA3
                denom3 = (1 - (BETA3 ** self.clients[cid].rounds_seen[me])) if self.clients[cid].rounds_seen[me] > 0 else 1.0
                R_hat = self.clients[cid].R[me] / (denom3 + 1e-12)

                # compute local lr: eta_lr = min(eta0, eta_r * R_hat)
                eta_lr = min(eta0, eta_r * (R_hat if R_hat > 0 else 1.0))

                # store bias-corrected prev values for next round
                self.clients[cid].M_hat_prev[me] = M_hat
                self.clients[cid].V_hat_prev[me] = V_hat

                # store computed local lr in client state for use when selected to train
                self.clients[cid].local_lr[me] = float(eta_lr)
                self.clients[cid].optimizer[me] = self.clients[cid].get_optimizer(dataset_name=self.args.dataset[me], me=me)
                print(f'rodada {t} learning rate: {eta_lr} cliente {cid}')

        except Exception as e:
            print("update_learning_rate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
    ):
        """Aggregate fit results using weighted average."""
        try:

            result = super().aggregate_fit(server_round, results, failures)
            for me in range(self.ME):
                self.update_learning_rate(server_round, me)
            return result

        except Exception as e:
            print("aggregate_fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


