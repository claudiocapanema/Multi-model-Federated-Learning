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
import numpy as np
from flcore.clients.clientavg import clientAVG
from flcore.servers.serveravg import MultiFedAvg
from threading import Thread


class FedFairMMFL(MultiFedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.fairness_weight = args.fairness_weight

    def select_clients(self, t):
        np.random.seed(t)
        if t == 1:
            return super().select_clients(t)
        else:
            if self.random_join_ratio:
                self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_join_clients
            np.random.seed(t)
            selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
            selected_clients_m = [[] for i in range(self.M)]

            for client in selected_clients:
                client_losses = []
                for metrics_m in client.train_metrics_list_dict:
                    client_losses.append(metrics_m['Loss'] * metrics_m['Samples'])
                client_losses = np.array(client_losses)
                client_losses = (np.power(client_losses, self.fairness_weight - 1)) / np.sum(client_losses)
                client_losses = client_losses / np.sum(client_losses)
                print("probal: ", client_losses)
                m = np.random.choice([i for i in range(self.M)], p=client_losses)
                selected_clients_m[m].append(client.id)

        print("Modelos clientes: ", selected_clients_m)

        return selected_clients_m
