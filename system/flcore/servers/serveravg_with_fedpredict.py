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
from flcore.clients.clientavg_with_fedpredict import ClientAvgWithFedPredict
from flcore.servers.serverbase import Server
from threading import Thread


class MultiFedAvgWithFedPredict(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.current_round = 1
        # select slow clients
        self.set_slow_clients()
        self.set_clients(ClientAvgWithFedPredict)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for t in range(1, self.global_rounds + 1):
            s_t = time.time()
            self.current_round = t
            self.selected_clients = self.select_clients(t)
            # self.send_models()
            for m in range(len(self.selected_clients)):
                if t % self.eval_gap == 0:
                    print(f"\n-------------Round number: {t}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(m, t=t)

                for t in range(len(self.selected_clients[m])):
                    self.clients[self.selected_clients[m][t]].train(m, t, self.global_model[m])

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and t % self.dlg_gap == 0:
                self.call_dlg(t)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        for m in range(self.M):
            self.save_results(m)
            self.save_global_model(m)

            if self.num_new_clients > 0:
                self.eval_new_clients = True
                self.set_new_clients(ClientAvgWithFedPredict)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(m, t=t)

    def test_metrics(self, m):
        if self.eval_new_clients and self.num_new_clients > 0:
            test_clients = self.new_clients
        else:
            test_clients = self.clients

        num_samples = []
        acc = []
        loss = []
        auc = []
        balanced_acc = []
        micro_fscore = []
        weighted_fscore = []
        macro_fscore = []
        for c in test_clients:
            test_acc, test_loss, test_num, test_auc, test_balanced_acc, test_micro_fscore, test_macro_fscore, test_weighted_fscore = c.test_metrics(m=m, global_model=self.global_model[m], t=self.current_round, T=self.global_rounds)
            acc.append(test_acc * test_num)
            auc.append(test_auc * test_num)
            num_samples.append(test_num)
            loss.append(test_loss * test_num)
            balanced_acc.append(test_balanced_acc * test_num)
            micro_fscore.append(test_micro_fscore * test_num)
            weighted_fscore.append(test_weighted_fscore * test_num)
            macro_fscore.append(test_macro_fscore * test_num)

        ids = [c.id for c in test_clients]

        decimals = 5
        acc = round(sum(acc) / sum(num_samples), decimals)
        auc = round(sum(auc) / sum(num_samples), decimals)
        loss = round(sum(loss) / sum(num_samples), decimals)
        balanced_acc = round(sum(balanced_acc) / sum(num_samples), decimals)
        micro_fscore = round(sum(micro_fscore) / sum(num_samples), decimals)
        weighted_fscore = round(sum(weighted_fscore) / sum(num_samples), decimals)
        macro_fscore = round(sum(macro_fscore) / sum(num_samples), decimals)

        return {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc, 'AUC': auc,
                "Loss": loss, "Balanced accuracy": balanced_acc, "Micro f1-score": micro_fscore,
                "Weighted f1-score": weighted_fscore, "Macro f1-score": macro_fscore}
