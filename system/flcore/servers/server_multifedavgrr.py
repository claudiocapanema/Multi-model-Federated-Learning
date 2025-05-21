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


class MultiFedAvgRR(MultiFedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)

    def select_clients(self, t):

        try:
            seed = t // self.ME
            step = t % self.ME
            np.random.seed(seed)
            selected_clients = list(np.random.choice(self.clients, self.num_training_clients, replace=False))
            selected_clients = [i.client_id for i in selected_clients]
            self.current_num_join_clients = self.num_training_clients

            n = len(selected_clients) // self.ME
            sc = np.array_split(selected_clients, self.ME)
            new_selected_clients = [[] for i in range(len(sc))]
            if step > 0:
                for i in range(len(sc)):
                    if i + step >= len(sc):
                        diff = len(sc) - i - step
                    else:
                        diff = i + step
                    new_selected_clients[i] = sc[diff]
            # sc = [np.array(selected_clients[0:6])]
            # sc.append(np.array(selected_clients[6:]))

                sc = np.array(new_selected_clients)

            self.n_trained_clients = sum([len(i) for i in sc])

            print("Selecionados: ", t, "\n", sc)
            print("----------")

            return sc

        except Exception as e:
            print("select_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


