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
import sys
import numpy as np
from flcore.servers.server_multifedavg import MultiFedAvg
from threading import Thread


class FedFairMMFL(MultiFedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.fairness_weight = 2

    def select_clients(self, t):
        try:
            np.random.seed(t)
            if t == 1:
                return super().select_clients(t)
            else:
                selected_clients = list(np.random.choice(self.clients, self.num_training_clients, replace=False))
                selected_clients = [i for i in selected_clients]

                selected_clients_me = []
                for me in range(self.ME):
                    selected_clients_me.append([])

                for client in selected_clients:
                    client_losses = []
                    for me in range(self.ME):
                        client_losses.append(client.loss_ME[me] * len(client.trainloader[me].dataset))
                    client_losses = np.array(client_losses)
                    client_losses = (np.power(client_losses, self.fairness_weight - 1)) / np.sum(client_losses)
                    client_losses = client_losses / np.sum(client_losses)
                    print("probal: ", client_losses)
                    m = np.random.choice([i for i in range(self.ME)], p=client_losses)
                    selected_clients_me[m].append(client.client_id)

            print("Modelos clientes: ", selected_clients_me)

            return selected_clients_me

        except Exception as e:
            print("select_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
