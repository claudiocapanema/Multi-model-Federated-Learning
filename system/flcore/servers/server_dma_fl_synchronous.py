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
import copy
import numpy as np
from flcore.servers.server_multifedavg import MultiFedAvg
from flcore.clients.client_dma_fl_synchronous import DMAFLSynchronousClient


class DMAFLSynchronous(MultiFedAvg):
    def __init__(self, args, times, fold_id):
        super().__init__(args, times, fold_id)

    def set_clients(self):

        try:
            for i in range(self.total_clients):
                client = DMAFLSynchronousClient(self.args,
                                id=i,
                                   model=copy.deepcopy(self.global_model),
                                fold_id=self.fold_id)
                self.clients.append(client)

        except Exception as e:
            print("set_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def select_clients(self, t):

        try:
            seed = t // self.ME
            step = t % self.ME
            np.random.seed(seed+self.fold_id)

            selected_clients_ids = []
            sc = []
            for me in range(self.ME):
                # Update client's stored last loss to current for next round comparison

                drift_scores = [c.drift_score[me] for c in self.clients]

                # ------------- Decide which clients are "high drift" -------------
                drift_threshold = 0.05
                high_drift_flags = [(score >= drift_threshold) for score in drift_scores]
                # show summary
                print("Drift scores (per client):", ["{:.3f}".format(s) for s in drift_scores])
                print("High-drift flags:", high_drift_flags)

                # ------------- Selection / scheduling: prioritize high-drift clients -------------
                # We build a selection pool where high-drift clients appear more times => higher selection prob.
                clients_probability = []
                for idx, c in enumerate(self.clients):
                    weight = 3 if high_drift_flags[idx] else 1  # simple multiplicative factor
                    if c.client_id in selected_clients_ids:
                        weight = 0
                    clients_probability.append(weight)

                clients_probability = np.array(clients_probability)
                clients_probability = clients_probability / np.sum(clients_probability)
                # choose clients_per_round unique clients (if pool smaller, fallback)

                print("tam: ", len(self.clients), len(clients_probability))

                selected_clients = list(np.random.choice(self.clients, self.num_training_clients//self.ME, p=clients_probability, replace=False))
                selected_clients = [i.client_id for i in selected_clients]
                selected_clients_ids += selected_clients
                self.current_num_join_clients = self.num_training_clients

                sc.append(selected_clients)

            self.n_trained_clients = sum([len(i) for i in sc])

            print("Selecionados: ", t, "\n", sc)
            print("----------")

            return sc

        except Exception as e:
            print("select_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


