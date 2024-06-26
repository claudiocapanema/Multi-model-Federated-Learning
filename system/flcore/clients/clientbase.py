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
import sys
import random
import ast
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.M = len(args.dataset)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.testloaderfull = []
        self.trainloader = []
        self.train_class_count = [np.array([0 for j in range(self.num_classes[i])]) for i in range(self.M)]
        self.test_loss = [[] for m in range(self.M)]

        # check BatchNorm
        self.has_BatchNorm = False
        for m in range(self.M):
            print(self.model)
            self.testloaderfull.append(self.load_test_data(m))
            train, self.train_class_count[m] = self.load_train_data(m)
            self.trainloader.append(train)
            for layer in self.model[m].children():
                if isinstance(layer, nn.BatchNorm2d):
                    self.has_BatchNorm = True
                    break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = []
        self.learning_rate_scheduler = []
        for m in range(self.M):
            if self.dataset[m] in ['ExtraSensory', 'WISDM-WATCH', 'WISDM-P']:
                self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.001))
            elif self.dataset[m] == "Tiny-ImageNet":
                self.optimizer.append(torch.optim.Adam(self.model[m].parameters(), lr=0.001))
            else:
                self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=self.learning_rate))
                self.learning_rate_scheduler.append(torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer[m],
                    gamma=args.learning_rate_decay_gamma
                ))
        self.learning_rate_decay = args.learning_rate_decay

        self.test_metrics_list_dict = [{} for m in range(self.M)]
        self.train_metrics_list_dict = [{} for m in range(self.M)]

        for m in range(self.M):
            trainloader, self.train_class_count[m] = self.load_train_data(m)

            print("Cliente: ", self.id, " modelo: ", m, " train class count: ", self.train_class_count[m])

    def load_wisdm(self, m, mode="train", batch_size=32, dataset_name=None):

        try:
            dir_path = "../dataset/WISDM-W/" + "clients_" + str(self.args.num_clients) + "/alpha_" + str(self.args.alpha[m]) + "/" + "client_" + str(
                self.id) + "/"
            filename_train = dir_path + """train/idx_train_{}.csv""".format(self.id)
            filename_test = dir_path + "test/idx_test_{}.csv""".format(self.id)
            cid = self.id
            filename_train = filename_train.replace("pickle", "csv")
            filename_test = filename_test.replace("pickle", "csv")

            train = pd.read_csv(filename_train)
            test = pd.read_csv(filename_test)

            df = pd.concat([train, test], ignore_index=True)
            x = np.array([ast.literal_eval(i) for i in df['X'].tolist()], dtype=np.float32)
            y = np.array([i for i in df['Y'].to_numpy().astype(np.int32)])

            for i in range(len(x)):
                row = x[i]
                indexes = row[:, 0].argsort(kind='mergesort')
                row = row[indexes]
                x[i] = row

            last_timestamp = []
            for i in range(len(x)):

                last_timestamp.append(x[i, -1, 0])

            indexes = np.array(last_timestamp).argsort(kind='heapsort')

            x = x[indexes]
            y = y[indexes]

            new_x = []
            for i in range(len(x)):
                row = x[i]
                if dataset_name != 'Cologne':
                    new_x.append(row[:, [1, 2, 3, 4, 5, 6]])
                else:
                    new_x.append(row[:, [1, 2]])

            x = np.array(new_x)

            p = np.unique(y, return_counts=True)
            c = p[0]
            total = np.sum(p[1])
            p = p[1]/total

            size = int(len(x) * 0.8)
            x_train, x_test = x[:size], x[size:]
            y_train, y_test = y[:size], y[size:]
            unique_count = {i: 0 for i in range(self.args.num_classes[m])}
            unique, count = np.unique(y, return_counts=True)
            data_unique_count_dict = dict(zip(unique, count))
            for class_ in data_unique_count_dict:
                unique_count[class_] = data_unique_count_dict[class_]
            unique_count = np.array(list(unique_count.values()))
            # print("Tamanho original dataset: ", len(x_train))

            training_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).to(dtype=torch.float32), torch.from_numpy(y_train))
            validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).to(dtype=torch.float32), torch.from_numpy(y_test))

            random.seed(cid)
            np.random.seed(cid)
            torch.manual_seed(cid)

            def seed_worker(worker_id):
                np.random.seed(cid)
                random.seed(cid)

            g = torch.Generator()
            g.manual_seed(cid)

            trainLoader = DataLoader(training_dataset, batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            testLoader = DataLoader(validation_dataset, batch_size, drop_last=False, shuffle=False)

            if mode == "train":
                return trainLoader, unique_count
            else:
                return testLoader

        except Exception as e:
            print("load WISDM")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_train_data(self, m, batch_size=None):
        if self.dataset[m] == "WISDM-W":
            return self.load_wisdm(m, mode='train')
        else:
            if batch_size == None:
                batch_size = self.batch_size
            train_data, unique_count = read_client_data(m, self.id, args=self.args, is_train=True)
            return DataLoader(train_data, batch_size, drop_last=True, shuffle=True), unique_count

    def load_test_data(self, m, batch_size=None):
        if self.dataset[m] == "WISDM-W":
            return self.load_wisdm(m, mode='test')
        else:
            if batch_size == None:
                batch_size = self.batch_size
            test_data = read_client_data(m, self.id, args=self.args, is_train=False)
            return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, m, model):
        for new_param, old_param in zip(model.parameters(), self.model[m].parameters()):
            old_param.data = new_param.data.clone()

    def set_parameters_all_models(self, models):

        for m in range(len(models)):
            model = models[m]
            for new_param, old_param in zip(model.parameters(), self.model[m].parameters()):
                old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self, m, model):
        testloaderfull = self.testloaderfull[m]
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        #self.set_parameters(m, model)
        self.model[m].to(self.device)
        self.model[m].eval()

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
                y = y.type(torch.LongTensor).to(self.device)
                output = self.model[m](x)
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

        self.test_loss[m].append(test_loss)

        self.test_metrics_list_dict[m] = {'ids': self.id, 'Accuracy': test_acc, 'AUC': test_auc,
                "Loss": test_loss, "Samples": test_num, "Balanced accuracy": test_balanced_acc, "Micro f1-score": test_micro_fscore,
                "Weighted f1-score": test_weighted_fscore, "Macro f1-score": test_macro_fscore}
        
        return (test_acc,
                test_loss,
                test_num,
                test_auc,
                test_balanced_acc,
                test_micro_fscore,
                test_macro_fscore,
                test_weighted_fscore)

    def train_metrics(self, m):
        trainloader, self.train_class_count[m] = self.load_train_data(m)
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
                y = y.type(torch.LongTensor).to(self.device)
                output = self.model[m](x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                train_loss += loss.item() * y.shape[0]

                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]

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

        return train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
