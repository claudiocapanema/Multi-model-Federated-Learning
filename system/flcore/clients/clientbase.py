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
import time
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
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from custom_federated_dataset import CustomFederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomResizedCrop, RandomAffine, ColorJitter,  Normalize, ToTensor, RandomRotation, Lambda


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
        self.algorithm = args.strategy
        self.dataset = args.dataset
        self.alpha = [args.alpha[i] for i in range(self.M)]
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.current_alpha = copy.deepcopy(self.alpha)
        self.num_classes = args.num_classes
        self.train_samples = [0] * self.M
        self.batch_size = [0] * self.M
        self.num_clients = args.total_clients
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
                """../concept_drift_configs/rounds_{}/datasets_{}/concept_drift_rounds_{}_{}/alpha_initial_{}_{}/alpha_end_{}_{}/config.csv""".format(self.args.number_of_rounds,
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
            self.trainloader[m], self.testloaderfull[m] = self.load_data(self.dataset[m], self.alpha[m], self.id, self.total_clients, self.batch_size[m])
            self.trainloader[m], self.train_class_count[m] = self.load_train_data(m, 1, batch_size=self.batch_size[m])
            self.train_samples[m] = 0
            for i, sample in enumerate(self.trainloader[m]):
                self.train_samples[m] += len(sample)
            print("no zero: ", np.count_nonzero(self.train_class_count[m]), self.train_class_count[m])
            self.fraction_of_classes[m] = np.count_nonzero(self.train_class_count[m]) / len(self.train_class_count[m])
            threshold = np.sum(self.train_class_count[m]) / len(self.train_class_count[m])
            self.imbalance_level[m] = len(np.argwhere(self.train_class_count[m] < threshold)) / len(self.train_class_count[m])
            print("fc do cliente ", self.id, self.fraction_of_classes[m], self.imbalance_level[m])
        

        self.train_slow = 0
        self.send_slow = 0
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = []
        self.learning_rate_scheduler = []
        for m in range(self.M):
            if self.dataset[m] in ['ExtraSensory', 'WISDM-W', 'WISDM-P']:
                # self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.0001))
                # self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.0001)) # loss constante não aprende
                # self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.0005)) # 102 rounds
                # self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.001)) antes

                # self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
                self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.001))
            elif self.dataset[m] in ['Gowalla']:
                # self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.0001))
                # self.optimizer.append(torch.optim.RMSprop(self.model[m].parameters(), lr=0.0001)) # loss constante não aprende
                # self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.001))  # bom para alpha 10# 101 e 102 rounds
                if float(self.alpha[m]) == 0.1:
                    self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
                elif float(self.alpha[m]) == 1.0:
                    self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
                elif float(self.alpha[m]) > 1.0:
                    self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
            elif self.dataset[m] in ["Tiny-ImageNet", "ImageNet", "ImageNet_v2"]:
                # self.optimizer.append(torch.optim.Adam(self.model[m].parameters(), lr=0.0001)) # bom para alpha 0.1
                # self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.001))  # aprende pouco mas não dá overfitting
                if float(self.alpha[m]) == 0.1:
                    self.optimizer.append(torch.optim.Adam(self.model[m].parameters(), lr=0.001))
                elif float(self.alpha[m]) == 1.0:
                    self.optimizer.append(torch.optim.Adam(self.model[m].parameters(), lr=0.001)) # funciona bem
                elif float(self.alpha[m]) > 1.0:
                    self.optimizer.append(torch.optim.Adam(self.model[m].parameters(), lr=0.001)) # sgd 0.01 bom 3% diferença

            elif self.dataset[m] in ["EMNIST", "CIFAR10"]:
                self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=0.01))
            else:
                self.optimizer.append(torch.optim.SGD(self.model[m].parameters(), lr=self.learning_rate))

        self.learning_rate_decay = args.learning_rate_decay

        self.test_metrics_list_dict = [{} for m in range(self.M)]
        self.train_metrics_list_dict = [[0] * 8 for m in range(self.M)]
        self.last_training_round = 0

    def load_data(self, dataset_name: str, alpha: float, partition_id: int, num_partitions: int, batch_size: int,
                  data_sampling_percentage: int, get_from_volume: bool = True):
        try:

            DATASET_INPUT_MAP = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image",
                                 "Gowalla": "sequence", "WISDM-W": "sequence", "ImageNet": "image"}
            # Only initialize `FederatedDataset` once
            print(
                """Loading {} {} {} {} {} {} data.""".format(dataset_name, partition_id, num_partitions, batch_size,
                                                             data_sampling_percentage, alpha))
            global fds
            if not get_from_volume:

                if dataset_name not in fds:
                    partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",

                                                       alpha=alpha, min_partition_size=10,

                                                       self_balancing=True)
                    fds[dataset_name] = FederatedDataset(
                        dataset=
                        {"EMNIST": "claudiogsc/emnist_balanced", "CIFAR10": "uoft-cs/cifar10", "MNIST": "ylecun/mnist",
                         "GTSRB": "claudiogsc/GTSRB", "Gowalla": "claudiogsc/Gowalla-State-of-Texas",
                         "WISDM-W": "claudiogsc/WISDM-W", "ImageNet": "claudiogsc/ImageNet-15_household_objects"}[
                            dataset_name],
                        partitioners={"train": partitioner},
                        seed=42
                    )
            else:
                # dts = dt.load_from_disk(f"datasets/{dataset_name}")
                partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                                   alpha=alpha, min_partition_size=10,
                                                   self_balancing=True)
                print("dataset from volume")
                fd = CustomFederatedDataset(
                    dataset=
                    {"EMNIST": "claudiogsc/emnist_balanced", "CIFAR10": "uoft-cs/cifar10", "MNIST": "ylecun/mnist",
                     "GTSRB": "claudiogsc/GTSRB", "Gowalla": "claudiogsc/Gowalla-State-of-Texas",
                     "WISDM-W": "claudiogsc/WISDM-W", "ImageNet": "claudiogsc/ImageNet-15_household_objects"}[
                        dataset_name],
                    partitioners={"train": partitioner},
                    path=f"datasets/{dataset_name}",
                    seed=42
                )
                fds[dataset_name] = fd

            attempts = 0
            while True:
                attempts += 1
                try:
                    time.sleep(random.randint(1, 1))
                    partition = fds[dataset_name].load_partition(partition_id)
                    print("""Loaded dataset {} in the {} attempt for client {}""".format(dataset_name, attempts,
                                                                                               partition_id))
                    break
                except Exception as e:
                    print(
                        """Tried to load dataset {} for the {} time for the client {} error""".format(dataset_name,
                                                                                                      attempts,
                                                                                                      partition_id))
                    print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
                    time.sleep(1)
            # Divide data on each node: 80% train, 20% test
            test_size = 1 - data_sampling_percentage
            partition_train_test = partition.train_test_split(test_size=test_size, seed=42)

            if dataset_name in ["CIFAR10", "MNIST", "EMNIST", "GTSRB", "ImageNet", "WISDM-W", "Gowalla"]:
                pytorch_transforms = {"CIFAR10": Compose(
                    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                    "MNIST": Compose([ToTensor(), RandomRotation(10),
                                      Normalize([0.5], [0.5])]),
                    "EMNIST": Compose([ToTensor(), RandomRotation(10),
                                       Normalize([0.5], [0.5])]),
                    "GTSRB": Compose(
                        [

                            Resize((32, 32)),
                            RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                            RandomRotation(10),  # Rotates the image to a specified angel
                            RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                            # Performs actions like zooms, change shear angles.
                            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                            ToTensor(),
                            Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                        ]
                    ),
                    "ImageNet": Compose(
                        [

                            Resize(32),
                            RandomHorizontalFlip(),
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                            # transforms.Resize((32, 32)),
                            # transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]
                    ),
                    "WISDM-W": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32))),
                    "Gowalla": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32))),

                }[dataset_name]

            # import torchvision.datasets as datasets
            # datasets.EMNIST
            key = DATASET_INPUT_MAP[dataset_name]

            def apply_transforms(batch):
                """Apply transforms to the partition from FederatedDataset."""

                batch[key] = [pytorch_transforms(img) for img in batch[key]]
                # print("""bath key: {}""".format(batch[key]))
                return batch

            if dataset_name in ["CIFAR10", "MNIST", "EMNIST", "GTSRB", "ImageNet", "WISDM-W", "Gowalla"]:
                partition_train_test = partition_train_test.with_transform(apply_transforms)
            trainloader = DataLoader(
                partition_train_test["train"], batch_size=batch_size, shuffle=True
            )
            testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
            return trainloader, testloader

        except Exception as e:
            print("load_data error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
        
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
        self.set_parameters(m, global_model)
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
        g.manual_seed(t)
        random.seed(t)
        np.random.seed(t)
        torch.manual_seed(t)
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
                if self.dataset[m] == "Gowalla":
                    x = torch.tensor(x, dtype=torch.long).to(self.device)
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

        self.test_metrics_list_dict[m] = (test_acc,
                test_loss,
                test_num,
                test_auc,
                test_balanced_acc,
                test_micro_fscore,
                test_macro_fscore,
                test_weighted_fscore,
                self.current_alpha[m])
        
        return (test_acc,
                test_loss,
                test_num,
                test_auc,
                test_balanced_acc,
                test_micro_fscore,
                test_macro_fscore,
                test_weighted_fscore,
                self.current_alpha[m])

    def train_metrics(self, m, global_model, t):

        g = torch.Generator()
        g.manual_seed(t)
        random.seed(t)
        np.random.seed(t)
        torch.manual_seed(t)
        self.set_parameters(m, global_model)
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
                if self.dataset[m] == "Gowalla":
                    x = torch.tensor(x, dtype=torch.long).to(self.device)
                if type(y) == tuple:
                    y = torch.from_numpy(np.array(list(y), dtype=np.int32))
                y = y.type(torch.LongTensor).to(self.device)
                output = self.model[m](x)
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

        self.train_metrics_list_dict[m] = train_acc, train_loss, train_num, train_balanced_acc, train_micro_fscore, train_macro_fscore, train_weighted_fscore, self.current_alpha[m]

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
