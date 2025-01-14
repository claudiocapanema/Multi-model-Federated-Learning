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
from flcore.clients.clientavg_with_fedpredict_train import clientAVGWithFedPredictTrain
from flcore.servers.serverbase import Server
from threading import Thread


class MultiFedAvgWithFedPredictTrain(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVGWithFedPredictTrain)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.client_class_count = {m: {i: [] for i in range(self.num_clients)} for m in range(self.M)}
        self.clients_training_count = {m: [0 for i in range(self.num_clients)] for m in range(self.M)}
        self.current_training_class_count = [np.array([0.0 for i in range(self.num_classes[j])]) for j in range(self.M)]
        for m in range(self.M):
            for i in range(self.num_clients):
                self.client_class_count[m][i] = self.clients[i].train_class_count[m] / np.sum(self.clients[i].train_class_count[m])
                self.current_training_class_count[m] += self.clients[i].train_class_count[m] / np.sum(self.clients[i].train_class_count[m])

            self.current_training_class_count[m] = np.round(
                self.current_training_class_count[m] / np.sum(self.current_training_class_count[m]), 2)

        print(self.current_training_class_count)


    def train(self):
        self._get_models_size()
        for t in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients(t)
            # self.send_models()
            print(self.selected_clients)

            for m in range(len(self.selected_clients)):


                for i in range(len(self.selected_clients[m])):
                    self.clients[self.selected_clients[m][i]].train(m, self.global_model[m], t, self.current_training_class_count[m])

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

            if self.num_new_clients > 0:
                self.eval_new_clients = True
                self.set_new_clients(clientAVGWithFedPredictTrain)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(m, t=t)
