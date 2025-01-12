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
import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
import random
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientKD(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        self.mentee_learning_rate = args.mentee_learning_rate

        self.global_model = copy.deepcopy(args.model)
        self.optimizer_g = [None] * self.M
        self.optimizer_w = [None] * self.M
        self.W_h = [None] * self.M
        self.last_not_nan = []
        for m in range(self.M):
            if self.dataset[m] in ['ExtraSensory', 'WISDM-W', 'WISDM-P']:
                self.optimizer_g.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.001))
                self.optimizer_w.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.001))
            elif self.dataset[m] in ["Tiny-ImageNet", "ImageNet", "ImageNet_v2"]:
                self.optimizer_g.append(torch.optim.Adam(self.model[m].parameters(), lr=0.0005))
                self.optimizer_w.append(torch.optim.Adam(self.model[m].parameters(), lr=0.0005))
            elif self.dataset[m] in ["EMNIST", "CIFAR10"]:
                self.optimizer_g.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
                self.optimizer_w.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
            else:
                self.optimizer_g.append(torch.optim.SGD(self.model[m].parameters(), lr=0.004))
                self.optimizer_w.append(torch.optim.SGD(self.model[m].parameters(), lr=0.004))

            self.feature_dim = list(args.model[m].parameters())[-2].shape[1]
            self.W_h[m] = nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(self.device)

        self.optimizer = []
        self.learning_rate_scheduler = []
        for m in range(self.M):
            if self.dataset[m] in ['ExtraSensory', 'WISDM-W', 'WISDM-P']:
                self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.001))
                # self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.0001)) # loss constante não aprende
                # self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
            elif self.dataset[m] in ["Tiny-ImageNet", "ImageNet", "ImageNet_v2"]:
                self.optimizer.append(torch.optim.Adam(self.model[m].parameters(), lr=0.0005))
            elif self.dataset[m] in ["EMNIST", "CIFAR10"]:
                self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
            else:
                self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.005))

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()

        self.compressed_param = {}
        self.energy = None
        self.past_train_metrics = None
        self.last_not_nan = torch.from_numpy(np.array([]))


    def train(self, m, t, global_model):
        trainloader = self.trainloader[m]
        self.model[m].to(self.device)
        global_model.to(self.device)
        self.set_parameters(m, global_model)
        self.model[m].train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            train_acc_student = 0
            train_acc_teacher = 0
            train_loss_student = 0
            train_num = 0
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                if type(y) == tuple:
                    y = torch.from_numpy(np.array(y).astype(int))
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                self.optimizer[m].zero_grad()
                output_student, rep_g, output_teacher, rep = self.model[m].forward_kd(x)
                outputs_S1 = F.log_softmax(output_student, dim=1)
                outputs_S2 = F.log_softmax(output_teacher, dim=1)
                outputs_T1 = F.softmax(output_student, dim=1)
                outputs_T2 = F.softmax(output_teacher, dim=1)

                loss_student = self.loss(output_student, y)
                loss_teacher = self.loss(output_teacher, y)
                loss = torch.nn.KLDivLoss()(outputs_S1, outputs_T2) / (loss_student + loss_teacher)
                loss += torch.nn.KLDivLoss()(outputs_S2, outputs_T1) / (loss_student + loss_teacher)
                L_h = self.MSE(rep, self.W_h[m](rep_g)) / (loss_student + loss_teacher)
                loss += loss_student + loss_teacher + L_h
                train_loss_student += loss.item()

                loss.backward()
                self.optimizer[m].step()

                train_acc_student += (torch.sum(torch.argmax(output_student, dim=1) == y)).item()
                train_acc_teacher += (torch.sum(torch.argmax(output_teacher, dim=1) == y)).item()

        # self.model.cpu()

        # self.decomposition()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_g.step()
            self.learning_rate_scheduler_W.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    # def set_parameters(self, global_param, energy):
    #     # recover
    #     for k in global_param.keys():
    #         if len(global_param[k]) == 3:
    #             # use np.matmul to support high-dimensional CNN param
    #             global_param[k] = np.matmul(global_param[k][0] * global_param[k][1][..., None, :], global_param[k][2])
    #
    #     for name, old_param in self.global_model.named_parameters():
    #         if name in global_param:
    #             old_param.data = torch.tensor(global_param[name], device=self.device).data.clone()
    #     self.energy = energy

    def get_parameters_of_model(self):
        try:
            parameters = [i.detach().numpy() for i in self.model[m].student.parameters()]
            return parameters
        except Exception as e:
            print("get parameters of model")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    # def train_metrics(self):
    #     trainloader = self.load_train_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()
    #
    #     train_num = 0
    #     losses = 0
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             rep = self.model.base(x)
    #             rep_g = self.global_model.base(x)
    #             output = self.model.head(rep)
    #             output_g = self.global_model.head(rep_g)
    #
    #             CE_loss = self.loss(output, y)
    #             CE_loss_g = self.loss(output_g, y)
    #             L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
    #             L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)
    #
    #             loss = CE_loss + L_d + L_h
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]
    #
    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')
    #
    #     return losses, train_num
    
    def decomposition(self):
        self.compressed_param = {}
        for name, param in self.global_model.named_parameters():
            param_cpu = param.detach().cpu().numpy()
            # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
            if param_cpu.shape[0]>1 and len(param_cpu.shape)>1 and 'embeddings' not in name:
                u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
                # support high-dimensional CNN param
                if len(u.shape)==4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    sigma = np.transpose(sigma, (2, 0, 1))
                    v = np.transpose(v, (2, 3, 0, 1))
                threshold=0
                if np.sum(np.square(sigma))==0:
                    compressed_param_cpu=param_cpu
                else:
                    for singular_value_num in range(len(sigma)):
                        if np.sum(np.square(sigma[:singular_value_num]))>self.energy*np.sum(np.square(sigma)):
                            threshold=singular_value_num
                            break
                    u=u[:, :threshold]
                    sigma=sigma[:threshold]
                    v=v[:threshold, :]
                    # support high-dimensional CNN param
                    if len(u.shape)==4:
                        u = np.transpose(u, (2, 3, 0, 1))
                        sigma = np.transpose(sigma, (1, 2, 0))
                        v = np.transpose(v, (2, 3, 0, 1))
                    compressed_param_cpu=[u,sigma,v]
            elif 'embeddings' not in name:
                compressed_param_cpu=param_cpu

            self.compressed_param[name] = compressed_param_cpu

    def set_parameters(self, m, global_model):
        try:
            for new_param, old_param in zip(global_model.parameters(), self.model[m].student.parameters()):
                old_param.data = new_param.data.clone()
        except Exception as e:
            print("set parameters to model")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def test_metrics(self, m, t, T, global_model):

        g = torch.Generator()
        g.manual_seed(self.id)
        random.seed(self.id)
        np.random.seed(self.id)
        torch.manual_seed(self.id)
        if bool(self.args.concept_drift):
            alpha = float(
                self.experiment_config_df.query("""Dataset == '{}' and Round == {}""".format(self.dataset[m], t))[
                    "Alpha"])
            if self.alpha[m] != alpha:
                self.alpha[m] = alpha
                self.update_loader[m] = True
                self.testloaderfull[m] = self.load_test_data(m, t, batch_size=self.batch_size)
                self.trainloader[m], self.train_class_count[m] = self.load_train_data(m, t,
                                                                                      batch_size=self.batch_size[m])
                self.train_samples[m] = 0
                for sample in self.trainloader[m]:
                    self.train_samples[m] += len(sample)
                threshold = np.sum(self.train_class_count[m]) / len(self.train_class_count[m])
                self.imbalance_level[m] = len(np.argwhere(self.train_class_count[m] < threshold)) / len(
                    self.train_class_count[m])
                print("fc do cliente ", self.id, self.fraction_of_classes[m], self.imbalance_level[m])
            else:
                self.update_loader[m] = False
        testloaderfull = self.testloaderfull[m]
        g = torch.Generator()
        g.manual_seed(self.id)
        random.seed(self.id)
        np.random.seed(self.id)
        torch.manual_seed(self.id)
        self.model[m].to(self.device)
        self.model[m].eval()

        test_acc = 0
        test_loss = 0
        test_num = 0
        y_prob = []
        y_true = []
        contab = 0

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                contab = contab + 1
                if type(y) == tuple:
                    y = torch.from_numpy(np.array(list(y), dtype=np.int32))
                y = y.type(torch.LongTensor).to(self.device)
                output = self.model[m](x)
                if torch.isnan(output).any() and len(output) == self.batch_size[m]:
                    output = self.last_not_nan[:len(y)]
                elif torch.isnan(output).any():
                    output = torch.from_numpy(np.array(self.last_not_nan.detach().cpu().numpy().tolist()*self.batch_size[m])[:len(y)]).to(self.device)
                else:
                    self.last_not_nan = output
                # print("saida: ", torch.isnan(output).any(), output)
                # print("true: ", torch.isnan(y).any())
                print("sai: ", output.shape, y.shape, x.shape)
                if output.shape[0] != y.shape[0]:
                    continue
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

        if test_num == 0:
            test_num = 1

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        test_loss = test_loss / test_num

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # print("y true: ", y_true, " y prob: ", y_prob)
        # print("entrada: ", x[:2])
        test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)

        test_acc = test_acc / test_num
        test_balanced_acc = metrics.balanced_accuracy_score(y_true, y_prob)
        test_micro_fscore = metrics.f1_score(y_true, y_prob, average='micro')
        test_macro_fscore = metrics.f1_score(y_true, y_prob, average='macro')
        test_weighted_fscore = metrics.f1_score(y_true, y_prob, average='weighted')

        self.test_loss[m].append(test_loss)

        self.test_metrics_list_dict[m] = {'ids': self.id, 'Accuracy': test_acc, 'AUC': test_auc,
                                          "Loss": test_loss, "Samples": test_num,
                                          "Balanced accuracy": test_balanced_acc, "Micro f1-score": test_micro_fscore,
                                          "Weighted f1-score": test_weighted_fscore,
                                          "Macro f1-score": test_macro_fscore}

        return (test_acc,
                test_loss,
                test_num,
                test_auc,
                test_balanced_acc,
                test_micro_fscore,
                test_macro_fscore,
                test_weighted_fscore,
                self.current_alpha[m])

    def train_metrics(self, m, t):

        try:
            g = torch.Generator()
            g.manual_seed(self.id)
            random.seed(self.id)
            np.random.seed(self.id)
            torch.manual_seed(self.id)
            if bool(self.args.concept_drift):
                alpha = float(
                    self.experiment_config_df.query("""Dataset == '{}' and Round == {}""".format(self.dataset[m], t))[
                        "Alpha"])
                if self.alpha[m] != alpha:
                    self.alpha[m] = alpha
                    self.update_loader[m] = True
                    self.testloaderfull[m] = self.load_test_data(m, t, batch_size=self.batch_size[m])
                    self.trainloader[m], self.train_class_count[m] = self.load_train_data(m, t, batch_size=self.batch_size[m])
                    self.train_samples[m] = 0
                    for sample in self.trainloader[m]:
                        self.train_samples[m] += len(sample)
                    threshold = np.sum(self.train_class_count[m]) / len(self.train_class_count[m])
                    self.imbalance_level[m] = len(np.argwhere(self.train_class_count[m] < threshold)) / len(
                        self.train_class_count[m])
                    print("fc do cliente ", self.id, self.fraction_of_classes[m], self.imbalance_level[m])
                else:
                    self.update_loader[m] = False
            trainloader = self.trainloader[m]
            g = torch.Generator()
            g.manual_seed(self.id)
            random.seed(self.id)
            np.random.seed(self.id)
            torch.manual_seed(self.id)
            # self.model = self.load_model('model')
            # self.model.to(self.device)
            self.model[m].eval()

            train_acc = 0
            train_num = 0
            train_loss = 0
            y_prob = []
            y_true = []
            with torch.no_grad():
                for x, y in trainloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    if type(y) == tuple:
                        y = torch.from_numpy(np.array(list(y), dtype=np.int32))
                    y = y.type(torch.LongTensor).to(self.device)
                    output = self.model[m](x)

                    if torch.isnan(output).any() and len(output) == self.batch_size[m]:
                        output = self.last_not_nan[:len(y)]
                        train_num = 1
                        continue
                    elif torch.isnan(output).any():
                        output = torch.from_numpy(np.array(self.last_not_nan.detach().cpu().numpy().tolist()*self.batch_size[m])[:len(y)]).to(self.device)
                        train_num = 1
                        continue
                    else:
                        self.last_not_nan = output
                    loss = self.loss(output, y)
                    train_num += y.shape[0]
                    train_loss += loss.item() * y.shape[0]

                    train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                    y_prob.append(output.detach().cpu().numpy())
                    nc = self.num_classes[m]
                    if self.num_classes[m] == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes[m] == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

            train_loss = train_loss / train_num

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

            y_prob = y_prob.argmax(axis=1)
            y_true = y_true.argmax(axis=1)

            train_acc = train_acc / train_num

            train_balanced_acc = metrics.balanced_accuracy_score(y_true, y_prob)
            train_micro_fscore = metrics.f1_score(y_true, y_prob, average='micro')
            train_macro_fscore = metrics.f1_score(y_true, y_prob, average='macro')
            train_weighted_fscore = metrics.f1_score(y_true, y_prob, average='weighted')

            # self.model.cpu()
            # self.save_model(self.model, 'model')

            self.train_metrics_list_dict[m] = {'ids': self.id, 'Accuracy': train_acc, "Loss": train_loss, "Samples": train_num,
                                               'Balanced accuracy': train_balanced_acc, 'Micro f1-score': train_micro_fscore,
                                               'Macro f1-score': train_macro_fscore, 'Weighted f1-score': train_weighted_fscore}

            self.past_train_metrics = train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore, self.current_alpha[m]
            return train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore, self.current_alpha[m]

        except Exception as e:
            print("train metrics")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            return self.past_train_metrics



    #
    # def train_metrics(self, m, t):
    #
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
    #             self.trainloader[m], self.train_class_count[m] = self.load_train_data(m, t, batch_size=self.batch_size[m])
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
    #             output = self.model[m].teacher(x)
    #             loss = self.loss(output, y)
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
    #     self.train_metrics_list_dict[m] = {'ids': self.id, 'Accuracy': train_acc, "Loss": train_loss, "Samples": train_num,
    #                                        'Balanced accuracy': train_balanced_acc, 'Micro f1-score': train_micro_fscore,
    #                                        'Macro f1-score': train_macro_fscore, 'Weighted f1-score': train_weighted_fscore}
    #
    #     return train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore, self.current_alpha[m]
