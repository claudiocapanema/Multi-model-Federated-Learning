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
import torch
import numpy as np
import time
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientavg import clientAVG
from utils.privacy import *
from fedpredict import fedpredict_client_torch


class ClientAvgWithFedPredict(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.last_training_round = 0
        self.combined_model = copy.deepcopy(self.model)

    def train(self, m, t, global_model):
        self.last_training_round = t
        # nt = t - self.last_training_round
        # print("nt: ", nt, t, self.last_training_round)
        # combined_model = fedpredict_client_torch(local_model=self.model[m], global_model=global_model,
        #                                          t=t, T=100, nt=nt)
        # self.set_parameters(m, combined_model)
        return super().train(m, global_model)

    def set_parameters_combined_model(self, m, model):
        for new_param, old_param in zip(model.parameters(), self.combined_model[m].parameters()):
            old_param.data = new_param.data.clone()

    def test_metrics(self, m, global_model, t, T):
        nt = t - self.last_training_round
        print("nt: ", nt, t, self.last_training_round)
        combined_model = fedpredict_client_torch(local_model=self.model[m], global_model=global_model,
                                  t=t, T=100, nt=nt)
        self.set_parameters_combined_model(m, combined_model)
        testloaderfull = self.load_test_data(m)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        # self.set_parameters(m, model)
        self.combined_model[m].to(self.device)
        self.combined_model[m].eval()

        test_acc = 0
        test_loss = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.combined_model[m](x)
                loss = self.loss(output, y)
                test_loss += loss.item() * y.shape[0]

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes[m]
                if self.num_classes[m] == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes[m] == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        test_loss = test_loss / test_num

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)

        test_acc = test_acc / test_num
        test_balanced_acc = metrics.balanced_accuracy_score(y_true, y_prob)
        test_micro_fscore = metrics.f1_score(y_true, y_prob, average='micro')
        test_macro_fscore = metrics.f1_score(y_true, y_prob, average='macro')
        test_weighted_fscore = metrics.f1_score(y_true, y_prob, average='weighted')

        # print(self.dataset[m], " micro: ", test_micro_fscore, test_acc)

        return (test_acc,
                test_loss,
                test_num,
                test_auc,
                test_balanced_acc,
                test_micro_fscore,
                test_macro_fscore,
                test_weighted_fscore)