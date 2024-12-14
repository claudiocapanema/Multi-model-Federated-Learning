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

import copy
import time
import numpy as np
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
        self._get_models_size()
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

    def test_metrics(self, m, t):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            test_clients = self.new_clients
        else:
            test_clients = self.clients

        num_samples = []
        accs_w = []
        std_accs_w = []
        loss_w = []
        std_losses_w = []
        auc_w = []
        balanced_acc_w = []
        micro_fscore_w = []
        weighted_fscore_w = []
        macro_fscore_w = []
        accs = []
        std_accs = []
        loss = []
        std_losses = []
        auc = []
        balanced_acc = []
        micro_fscore = []
        weighted_fscore = []
        macro_fscore = []
        alpha_list = []
        for i in range(len(test_clients)):
            # if i in self.selected_clients[m] or t == 1:
            c = test_clients[i]
            test_acc, test_loss, test_num, test_auc, test_balanced_acc, test_micro_fscore, test_macro_fscore, test_weighted_fscore, alpha = c.test_metrics(
               m=m, global_model=copy.deepcopy(self.global_model[m].to(self.device)), t=t, T=self.args.global_rounds)
            self.clients_test_metrics[i]["Accuracy"][m].append(test_acc)
            self.clients_test_metrics[i]["Loss"][m].append(test_loss)
            self.clients_test_metrics[i]["Balanced accuracy"][m].append(test_balanced_acc)
            self.clients_test_metrics[i]["Micro f1-score"][m].append(test_micro_fscore)
            self.clients_test_metrics[i]["Samples"][m].append(test_num)
            # print("test_weighted_fscore: ", test_weighted_fscore, len(self.clients_test_metrics[i]["Weighted f1-score"][m]))
            self.clients_test_metrics[i]["Weighted f1-score"][m].append(test_weighted_fscore)
            self.clients_test_metrics[i]["Macro f1-score"][m].append(test_macro_fscore)
            self.clients_test_metrics[i]["Round"][m].append(t)
            # accs.append(test_acc*test_num)
            # auc.append(test_auc*test_num)
            # num_samples.append(test_num)
            # loss.append(test_loss*test_num)
            # balanced_acc.append(test_balanced_acc*test_num)
            # micro_fscore.append(test_micro_fscore*test_num)
            # weighted_fscore.append(test_weighted_fscore*test_num)
            # macro_fscore.append(test_macro_fscore*test_num)
            accs_w.append(test_acc * test_num)
            std_accs_w.append(test_acc * test_num)
            auc_w.append(test_auc * test_num)
            num_samples.append(test_num)
            loss_w.append(test_loss * test_num)
            std_losses_w.append(test_loss * test_num)
            balanced_acc_w.append(test_balanced_acc * test_num)
            micro_fscore_w.append(test_micro_fscore * test_num)
            weighted_fscore_w.append(test_weighted_fscore * test_num)
            macro_fscore_w.append(test_macro_fscore * test_num)

            accs.append(test_acc)
            std_accs.append(test_acc)
            auc.append(test_auc)
            # num_samples.append(test_num)
            loss.append(test_loss)
            std_losses.append(test_loss)
            balanced_acc.append(test_balanced_acc)
            micro_fscore.append(test_micro_fscore)
            weighted_fscore.append(test_weighted_fscore)
            macro_fscore.append(test_macro_fscore)
            alpha_list.append(alpha)

            # print(test_num)
            # exit()

        ids = [c.id for c in test_clients]

        decimals = 5
        acc_w = round(sum(accs_w) / sum(num_samples), decimals)
        std_acc_w = np.round(np.std(np.array(std_accs_w) / sum(num_samples)), decimals)
        auc_w = round(sum(auc_w) / sum(num_samples), decimals)
        loss_w = round(sum(loss_w) / sum(num_samples), decimals)
        std_loss_w = np.round(np.std(np.array(std_losses_w) / sum(num_samples)), decimals)
        balanced_acc_w = round(sum(balanced_acc_w) / sum(num_samples), decimals)
        micro_fscore_w = round(sum(micro_fscore_w) / sum(num_samples), decimals)
        weighted_fscore_w = round(sum(weighted_fscore_w) / sum(num_samples), decimals)
        macro_fscore_w = round(sum(macro_fscore_w) / sum(num_samples), decimals)

        acc = round(np.mean(accs), decimals)
        std_acc = np.round(np.std(np.array(std_accs)), decimals)
        auc = round(np.mean(auc), decimals)
        loss = round(np.mean(loss), decimals)
        std_loss = np.round(np.std(np.array(std_losses)), decimals)
        balanced_acc = round(np.mean(balanced_acc), decimals)
        micro_fscore = round(np.mean(micro_fscore), decimals)
        weighted_fscore = round(np.mean(weighted_fscore), decimals)
        macro_fscore = round(np.mean(macro_fscore), decimals)

        server_metrics = {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc, "Std Accuracy": std_acc, 'AUC': auc,
                          "Loss": loss, "Std loss": std_loss, "Balanced accuracy": balanced_acc,
                          "Micro f1-score": micro_fscore,
                          "Weighted f1-score": weighted_fscore, "Macro f1-score": macro_fscore, "Alpha": alpha_list[0]}

        server_metrics_weighted = {'ids': ids, 'num_samples': num_samples, 'Accuracy': acc_w, "Std Accuracy": std_acc_w,
                                   'AUC': auc_w,
                                   "Loss": loss_w, "Std loss": std_loss_w, "Balanced accuracy": balanced_acc_w,
                                   "Micro f1-score": micro_fscore_w,
                                   "Weighted f1-score": weighted_fscore_w, "Macro f1-score": macro_fscore_w,
                                   "Alpha": alpha_list[0]}

        return server_metrics, server_metrics_weighted
