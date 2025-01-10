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

import random
import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientProto(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        self.protos = [None] * self.M
        self.global_protos = [None] * self.M
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda


    def train(self, m, t, global_protos):
        self.global_protos[m] = global_protos
        trainloader = self.trainloader[m]
        start_time = time.time()

        # self.model.to(self.device)
        self.model[m].train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                if type(y) == tuple:
                    y = torch.from_numpy(np.array(list(y), dtype=np.int32))
                y = y.type(torch.LongTensor).to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output, rep = self.model[m].forward_kd(x)
                loss = self.loss(output, y)

                if self.global_protos[m] is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[m][y_c]) != type([]) and self.global_protos[m][y_c] is not None:
                            proto_new[i, :] = self.global_protos[m][y_c]
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                self.optimizer[m].zero_grad()
                loss.backward()
                self.optimizer[m].step()

        # self.model.cpu()
        # rep = self.model.base(x)
        # print(torch.sum(rep!=0).item() / rep.numel())

        # self.collect_protos()
        self.protos[m] = agg_func(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    # def test_metrics(self, m, t, T, global_protos):
    #     g = torch.Generator()
    #     g.manual_seed(self.id)
    #     random.seed(self.id)
    #     np.random.seed(self.id)
    #     torch.manual_seed(self.id)
    #     if bool(self.args.concept_drift):
    #         alpha = float(
    #             self.experiment_config_df.query("""Dataset == '{}' and Round == {}""".format(self.dataset[m], t))[
    #                 "Alpha"])
    #         if self.alpha[m] != alpha:
    #             self.alpha[m] = alpha
    #             self.update_loader[m] = True
    #             self.testloaderfull[m] = self.load_test_data(m, t, batch_size=self.batch_size)
    #             self.trainloader[m], self.train_class_count[m] = self.load_train_data(m, t,
    #                                                                                   batch_size=self.batch_size[m])
    #             self.train_samples[m] = 0
    #             for sample in self.trainloader[m]:
    #                 self.train_samples[m] += len(sample)
    #             threshold = np.sum(self.train_class_count[m]) / len(self.train_class_count[m])
    #             self.imbalance_level[m] = len(np.argwhere(self.train_class_count[m] < threshold)) / len(
    #                 self.train_class_count[m])
    #             print("fc do cliente ", self.id, self.fraction_of_classes[m], self.imbalance_level[m])
    #         else:
    #             self.update_loader[m] = False
    #     testloaderfull = self.testloaderfull[m]
    #     g = torch.Generator()
    #     g.manual_seed(self.id)
    #     random.seed(self.id)
    #     np.random.seed(self.id)
    #     torch.manual_seed(self.id)
    #     self.model[m].to(self.device)
    #     self.model[m].eval()
    #
    #     test_acc = 0
    #     test_loss = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []
    #     contab = 0
    #
    #     if self.global_protos[m] is not None:
    #         with torch.no_grad():
    #             for x, y in testloaderfull:
    #                 if type(x) == type([]):
    #                     x[0] = x[0].to(self.device)
    #                 else:
    #                     x = x.to(self.device)
    #                 y = y.to(self.device)
    #                 if type(y) == tuple:
    #                     y = torch.from_numpy(np.array(list(y), dtype=np.int32))
    #                 y = y.type(torch.LongTensor).to(self.device)
    #                 output, rep = self.model[m].forward_kd(x)
    #
    #                 print("y: ", y.shape[0], y.shape)
    #                 output = float('inf') * torch.ones(y.shape[0], self.num_classes[m]).to(self.device)
    #                 for i, r in enumerate(rep):
    #                     for j, pro in self.global_protos[m]:
    #                         if type(pro) != type([]):
    #                             output[i, j] = self.loss_mse(r, pro)
    #
    #                 loss = self.loss(output, y)
    #                 test_loss += loss.item() * y.shape[0]
    #
    #                 test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #                 test_num += y.shape[0]
    #
    #                 y_prob.append(output.detach().cpu().numpy())
    #                 nc = self.num_classes[m]
    #                 if self.num_classes[m] == 2:
    #                     nc += 1
    #                 lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #                 if self.num_classes[m] == 2:
    #                     lb = lb[:, :2]
    #                 y_true.append(lb)
    #
    #                 # self.model.cpu()
    #                 # self.save_model(self.model, 'model')
    #
    #             test_loss = test_loss / test_num
    #
    #             y_prob = np.concatenate(y_prob, axis=0)
    #             y_true = np.concatenate(y_true, axis=0)
    #             test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    #
    #             y_prob = y_prob.argmax(axis=1)
    #             y_true = y_true.argmax(axis=1)
    #
    #             test_acc = test_acc / test_num
    #             test_balanced_acc = metrics.balanced_accuracy_score(y_true, y_prob)
    #             test_micro_fscore = metrics.f1_score(y_true, y_prob, average='micro')
    #             test_macro_fscore = metrics.f1_score(y_true, y_prob, average='macro')
    #             test_weighted_fscore = metrics.f1_score(y_true, y_prob, average='weighted')
    #
    #             self.test_loss[m].append(test_loss)
    #
    #             self.test_metrics_list_dict[m] = {'ids': self.id, 'Accuracy': test_acc, 'AUC': test_auc,
    #                                               "Loss": test_loss, "Samples": test_num,
    #                                               "Balanced accuracy": test_balanced_acc,
    #                                               "Micro f1-score": test_micro_fscore,
    #                                               "Weighted f1-score": test_weighted_fscore,
    #                                               "Macro f1-score": test_macro_fscore}
    #
    #             return (test_acc,
    #                     test_loss,
    #                     test_num,
    #                     test_auc,
    #                     test_balanced_acc,
    #                     test_micro_fscore,
    #                     test_macro_fscore,
    #                     test_weighted_fscore,
    #                     self.current_alpha[m])
    #     else:
    #         return 0, 1e-5, 0, 0, 0, 0, 0, 0, 0

    # def train_metrics(self, m, t, T, global_protos):
    #     g = torch.Generator()
    #     g.manual_seed(self.id)
    #     random.seed(self.id)
    #     np.random.seed(self.id)
    #     torch.manual_seed(self.id)
    #     if bool(self.args.concept_drift):
    #         alpha = float(
    #             self.experiment_config_df.query("""Dataset == '{}' and Round == {}""".format(self.dataset[m], t))[
    #                 "Alpha"])
    #         if self.alpha[m] != alpha:
    #             self.alpha[m] = alpha
    #             self.update_loader[m] = True
    #             self.testloaderfull[m] = self.load_test_data(m, t, batch_size=self.batch_size[m])
    #             self.trainloader[m], self.train_class_count[m] = self.load_train_data(m, t,
    #                                                                                   batch_size=self.batch_size[m])
    #             self.train_samples[m] = 0
    #             for sample in self.trainloader[m]:
    #                 self.train_samples[m] += len(sample)
    #             threshold = np.sum(self.train_class_count[m]) / len(self.train_class_count[m])
    #             self.imbalance_level[m] = len(np.argwhere(self.train_class_count[m] < threshold)) / len(
    #                 self.train_class_count[m])
    #             print("fc do cliente ", self.id, self.fraction_of_classes[m], self.imbalance_level[m])
    #         else:
    #             self.update_loader[m] = False
    #     trainloader = self.trainloader[m]
    #     g = torch.Generator()
    #     g.manual_seed(self.id)
    #     random.seed(self.id)
    #     np.random.seed(self.id)
    #     torch.manual_seed(self.id)
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model[m].eval()
    #
    #     train_acc = 0
    #     train_num = 0
    #     train_loss = 0
    #     y_prob = []
    #     y_true = []
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             if type(y) == tuple:
    #                 y = torch.from_numpy(np.array(list(y), dtype=np.int32))
    #             y = y.type(torch.LongTensor).to(self.device)
    #             y = y.to(self.device)
    #             output, rep = self.model[m].forward_kd(x)
    #             loss = self.loss(output, y)
    #
    #             if self.global_protos is not None:
    #                 proto_new = copy.deepcopy(rep.detach())
    #                 for i, yy in enumerate(y):
    #                     y_c = yy.item()
    #                     if type(self.global_protos[y_c]) != type([]):
    #                         proto_new[i, :] = self.global_protos[y_c].data
    #                 loss += self.loss_mse(proto_new, rep) * self.lamda
    #             train_num += y.shape[0]
    #             train_loss += loss.item() * y.shape[0]
    #
    #             train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #
    #             y_prob.append(output.detach().cpu().numpy())
    #             nc = self.num_classes[m]
    #             if self.num_classes[m] == 2:
    #                 nc += 1
    #             lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #             if self.num_classes[m] == 2:
    #                 lb = lb[:, :2]
    #             y_true.append(lb)
    #
    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')
    #
    #     train_loss = train_loss / train_num
    #
    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = np.concatenate(y_true, axis=0)
    #     test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    #
    #     y_prob = y_prob.argmax(axis=1)
    #     y_true = y_true.argmax(axis=1)
    #
    #     train_acc = train_acc / train_num
    #
    #     train_balanced_acc = metrics.balanced_accuracy_score(y_true, y_prob)
    #     train_micro_fscore = metrics.f1_score(y_true, y_prob, average='micro')
    #     train_macro_fscore = metrics.f1_score(y_true, y_prob, average='macro')
    #     train_weighted_fscore = metrics.f1_score(y_true, y_prob, average='weighted')
    #
    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')
    #
    #     self.train_metrics_list_dict[m] = {'ids': self.id, 'Accuracy': train_acc, "Loss": train_loss,
    #                                        "Samples": train_num,
    #                                        'Balanced accuracy': train_balanced_acc,
    #                                        'Micro f1-score': train_micro_fscore,
    #                                        'Macro f1-score': train_macro_fscore,
    #                                        'Weighted f1-score': train_weighted_fscore}
    #
    #     return train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore, \
    #     self.current_alpha[m]


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos