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
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data_v2, read_gtsrb


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, **kwargs):
        g = torch.Generator()
        g.manual_seed(id)
        random.seed(id)
        np.random.seed(id)
        torch.manual_seed(id)
        self.args = args
        self.M = len(args.dataset)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.alpha = [args.alpha[i] for i in range(self.M)]
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.current_alpha = copy.deepcopy(self.alpha)
        self.num_classes = args.num_classes
        self.train_samples = [0] * self.M
        self.batch_size = [0] * self.M
        for m in range(self.M):
            if self.dataset[m] == "EMNIST":
                self.batch_size[m] = 128
            else:
                self.batch_size[m] = 64

        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.testloaderfull = [0 for m in range(self.M)]
        self.trainloader = [0 for m in range(self.M)]
        self.update_loader = [False for j in range(self.M)]
        self.train_class_count = [np.array([0 for j in range(self.num_classes[i])]) for i in range(self.M)]
        self.test_loss = [[] for m in range(self.M)]
        self.concept_drift = bool(self.args.concept_drift)
        self.alpha_end= self.args.alpha_end
        self.rounds_concept_drift = self.args.rounds_concept_drift
        if self.concept_drift:
            self.experiment_config_df = pd.read_csv(
                """../concept_drift_configs/rounds_{}/datasets_{}/concept_drift_rounds_{}_{}/alpha_initial_{}_{}/alpha_end_{}_{}/config.csv""".format(self.args.global_rounds,
                                                                                                                                                    self.dataset,
                                                                                                                                                    self.rounds_concept_drift[0],
                                                                                                                                                    self.rounds_concept_drift[1],
                                                                                                                                                    self.alpha[0],
                                                                                                                                                    self.alpha[1],
                                                                                                                                                    self.alpha_end[0],
                                                                                                                                                    self.alpha_end[1]))
        self.fraction_of_classes = [0 for m in range(self.M)]
        self.imbalance_level = [0 for m in range(self.M)]
        # check BatchNorm
        self.has_BatchNorm = False
        for m in range(self.M):
            print(self.model)
            self.testloaderfull[m] = self.load_test_data(m, 1, batch_size=self.batch_size[m])
            self.trainloader[m], self.train_class_count[m] = self.load_train_data(m, 1, batch_size=self.batch_size[m])
            self.train_samples[m] = 0
            for i, sample in enumerate(self.trainloader[m]):
                self.train_samples[m] += len(sample)
            print("no zero: ", np.count_nonzero(self.train_class_count[m]), self.train_class_count[m])
            self.fraction_of_classes[m] = np.count_nonzero(self.train_class_count[m]) / len(self.train_class_count[m])
            threshold = np.sum(self.train_class_count[m]) / len(self.train_class_count[m])
            self.imbalance_level[m] = len(np.argwhere(self.train_class_count[m] < threshold)) / len(self.train_class_count[m])
            print("fc do cliente ", self.id, self.fraction_of_classes[m], self.imbalance_level[m])
        

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
            if self.dataset[m] in ['ExtraSensory', 'WISDM-W', 'WISDM-P']:
                self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.001))
                # self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.0001)) # loss constante não aprende
                # self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
            elif self.dataset[m] in ["Tiny-ImageNet", "ImageNet", "ImageNet_v2"]:
                self.optimizer.append(torch.optim.Adam(self.model[m].parameters(), lr=0.0005))
            elif self.dataset[m] in ["EMNIST", "CIFAR10"]:
                self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
            else:
                self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=self.learning_rate))

        self.learning_rate_decay = args.learning_rate_decay

        self.test_metrics_list_dict = [{} for m in range(self.M)]
        self.train_metrics_list_dict = [{} for m in range(self.M)]


        #
        # for m in range(self.M):
        #     trainloader, self.train_class_count[m] = self.load_train_data(m, 1)
        #     self.fc[m] = np.nonzero(self.train_class_count[m])/len(self.train_class_count[m])

        print("Cliente: ", self.id, " modelo: ", m, " train class count: ", self.train_class_count[m])

    def load_wisdm(self, m, name, alpha, mode="train", batch_size=32, dataset_name=None):

        try:
            dir_path = "../dataset/" + name + "/" + "clients_" + str(self.args.num_clients) + "/alpha_" + str(alpha) + "/"
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
            print("Wisdm tamanho original dataset: ", len(x_train))

            training_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).to(dtype=torch.float32), torch.from_numpy(y_train).to(dtype=torch.int32))
            validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).to(dtype=torch.float32), torch.from_numpy(y_test).to(dtype=torch.int32))

            random.seed(cid)
            np.random.seed(cid)
            torch.manual_seed(cid)

            def seed_worker(worker_id):
                np.random.seed(cid)
                random.seed(cid)

            g = torch.Generator()
            g.manual_seed(cid)
            np.random.seed(cid)
            random.seed(cid)

            trainLoader = DataLoader(training_dataset, batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            testLoader = DataLoader(validation_dataset, batch_size, drop_last=False, shuffle=False)

            if mode == "train":
                return trainLoader, unique_count
            else:
                return testLoader

        except Exception as e:
            print("load WISDM client base")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_imagenet(self, name, m, alpha, t, mode="train", batch_size=32, ):

        try:
            dir_path = "../dataset/" + name + "/" + "clients_" + str(self.args.num_clients) + "/alpha_" + str(alpha) + "/"
            traindir = """../dataset/ImageNet/rawdata/ImageNet/train/"""
            filename_train = dir_path + """train/idx_train_{}.pickle""".format(self.id)
            filename_test = dir_path + "test/idx_test_{}.pickle""".format(self.id)

            random.seed(self.id)
            np.random.seed(self.id)
            torch.manual_seed(self.id)
            g = torch.Generator()
            g.manual_seed(self.id)

            transmforms = {'train': transforms.Compose(
                    [

                        transforms.Resize((32, 32)),
                        transforms.RandomRotation(10),  # Rotates the image to a specified angel
                        transforms.ToTensor(),
                        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                        # transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
                        # transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                        # transforms.RandomRotation(10),  # Rotates the image to a specified angel
                        # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                        # # Performs actions like zooms, change shear angles.
                        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                        # transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                ), 'test': transforms.Compose(
                    [

                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                        # transforms.Resize((32, 32)),
                        # transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )}[mode]

            training_dataset = datasets.ImageFolder(
                traindir,
                transmforms
            )

            validation_dataset = datasets.ImageFolder(
                traindir,
                transmforms
            )

            # np.random.seed(self.id)

            dataset_image = []
            dataset_samples = []
            dataset_label = []
            dataset_samples.extend(training_dataset.samples)
            dataset_image.extend(training_dataset.imgs)
            dataset_label.extend(training_dataset.targets)

            with open(filename_train, 'rb') as handle:
                idx_train = pickle.load(handle)

            with open(filename_test, 'rb') as handle:
                idx_test = pickle.load(handle)

            # print("tipo: ", type(training_dataset.imgs), type(training_dataset.targets), type(training_dataset.samples))
            imgs = training_dataset.imgs
            x_train = []
            x_test = []
            y_train = []
            y_test = []
            for i in range(1):
                x_train += training_dataset.samples
                x_test += training_dataset.samples
                y_train += training_dataset.targets
                y_test += training_dataset.targets

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            x_train = x_train[idx_train]
            y_train = y_train[idx_train]
            x_test = x_test[idx_test]
            y_test = y_test[idx_test]

            training_dataset.samples = list(x_train)
            training_dataset.targets = list(y_train)
            validation_dataset.samples = list(x_test)
            validation_dataset.targets = list(y_test)

            y = np.array(list(y_train) + list(y_test))


            # validation_dataset.imgs = list(imgs_test)

            def seed_worker(worker_id):
                np.random.seed(self.id)
                random.seed(self.id)
                torch.manual_seed(self.id)
                g = torch.Generator()
                g.manual_seed(self.id)

            g = torch.Generator()
            g.manual_seed(self.id)
            random.seed(self.id)
            np.random.seed(self.id)
            torch.manual_seed(self.id)

            unique_count = {i: 0 for i in range(self.args.num_classes[m])}
            unique, count = np.unique(y, return_counts=True)
            data_unique_count_dict = dict(zip(unique, count))
            for class_ in data_unique_count_dict:
                unique_count[class_] = data_unique_count_dict[class_]
            unique_count = np.array(list(unique_count.values()))

            trainLoader = DataLoader(training_dataset, batch_size, shuffle=True, worker_init_fn=seed_worker,
                                     generator=g)
            testLoader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)

            if self.id == 10:

                print("Cliente 10 rodada ", t, " alpha: ", alpha, " dataset: ", name, " value counts: ", pd.Series(y_test).value_counts())

            if mode == "train":
                return trainLoader, unique_count
            else:
                return testLoader

        except Exception as e:
            print("load ImageNet client base")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_train_data(self, m, t, batch_size=32):

        try:
            alpha = self.alpha[m]
            if bool(self.args.concept_drift):
                alpha = float(self.experiment_config_df.query("""Dataset == '{}' and Round == {}""".format(self.dataset[m], t))["Alpha"])
                self.current_alpha[m] = alpha
            if self.dataset[m] in ["WISDM-W", "WISDM-P"]:
                return self.load_wisdm(m=m, alpha=alpha, name=self.dataset[m],  mode='train')
            elif "ImageNet" in self.dataset[m]:
                return self.load_imagenet(self.dataset[m], m=m, alpha=alpha, t=t, mode='train')
            if self.dataset[m] == "GTSRB":
                return read_gtsrb(name=self.dataset[m], args=self.args, m=m, cid=self.id, t=t, mode='train')
            else:
                return read_client_data_v2(m, self.dataset[m], self.id, batch_size=batch_size, args=self.args, mode="train")
        except Exception as e:
            print("load train data")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_test_data(self, m, t, batch_size=None):
        try:
            alpha = self.alpha[m]
            if bool(self.args.concept_drift):
                alpha = float(
                    self.experiment_config_df.query("""Dataset == '{}' and Round == {}""".format(self.dataset[m], t))[
                        "Alpha"])
                self.current_alpha[m] = alpha
            if self.dataset[m] in ["WISDM-W", "WISDM-P"]:
                return self.load_wisdm(m=m, name=self.dataset[m], alpha=alpha, mode='test')
            elif "ImageNet" in self.dataset[m]:
                return self.load_imagenet(name=self.dataset[m], m=m, alpha=alpha, t=t, mode='test')
            if self.dataset[m] == "GTSRB":
                return read_gtsrb(name=self.dataset[m], m=m, cid=self.id, args=self.args, t=t, mode='test')
            else:
                return read_client_data_v2(m, self.dataset[m], self.id, batch_size=batch_size, args=self.args, mode="test")
        except Exception as e:
            print("load test data")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        
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
                test_weighted_fscore,
                self.current_alpha[m])

    def train_metrics(self, m, t):

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

        return train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore, self.current_alpha[m]

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
