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
from flcore.servers.serveravg import Server
from threading import Thread


class MultiFedAvgRR(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

    def select_clients(self, t):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        seed = t // self.M
        np.random.seed(seed)
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        n = len(selected_clients) // self.M
        selected_clients = [selected_clients[i:i+n] for i in range(0, len(selected_clients), n)]

        # print(np.array(selected_clients).shape)
        # print(np.array(selected_clients)[0])
        # print(n)
        # exit()

        return selected_clients
