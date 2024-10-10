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
import random
import ast
import pickle
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import subprocess
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.datasets as datasets


def read_data(dataset, idx, args, alpha, is_train=True):
    if is_train:
        path = """../dataset/{}/clients_{}/alpha_{}/train/""".format(dataset, args.num_clients,alpha)
        train_data_dir = path

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()


        return train_data

    else:
        path = """../dataset/{}/clients_{}/alpha_{}/test/""".format(dataset, args.num_clients, alpha)
        test_data_dir = path

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

def load_wisdm(m, name, cid, args, mode="train", batch_size=32, dataset_name=None):

    try:
        dir_path = "../dataset/" + name + "/" + "clients_" + str(args.num_clients) + "/alpha_" + str(args.alpha[m]) + "/"
        filename_train = dir_path + """train/idx_train_{}.csv""".format(cid)
        filename_test = dir_path + "test/idx_test_{}.csv""".format(cid)

        filename_train = filename_train.replace("pickle", "csv")
        filename_test = filename_test.replace("pickle", "csv")

        train = pd.read_csv(filename_train)
        test = pd.read_csv(filename_test)

        df = pd.concat([train, test], ignore_index=True)
        # print("lll: ", df['X'].tolist())
        # for i in range(len(df['X'])):
        #     print(df['X'].tolist()[i])
        #     print(ast.literal_eval(df['X'].tolist()[i]))
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
        total = np.sum(p[1])
        p = p[1]/total

        size = int(len(x) * 0.8)
        x_train, x_test = x[:size], x[size:]
        y_train, y_test = y[:size], y[size:]
        unique_count = {i: 0 for i in range(args.num_classes[m])}
        unique, count = np.unique(y, return_counts=True)
        data_unique_count_dict = dict(zip(unique, count))
        for class_ in data_unique_count_dict:
            unique_count[class_] = data_unique_count_dict[class_]
        unique_count = np.array(list(unique_count.values()))
        print("Tamanho original dataset: ", len(x_train))

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
        testLoader = DataLoader(validation_dataset, batch_size, shuffle=False)

        if mode == "train":
            return trainLoader, unique_count
        else:
            return testLoader, unique_count

    except Exception as e:
        print("load WISDM")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_imagenet(self, m, mode="train", batch_size=32, dataset_name=None):

        try:
            dir_path = "../dataset/ImageNet/" + "clients_" + str(self.args.num_clients) + "/alpha_" + str(self.args.alpha[m]) + "/" + "client_" + str(
                self.id) + "/"
            traindir = """/home/claudio/Documentos/pycharm_projects/Multi-model-Federated-Learning/dataset/ImageNet/clients_40/alpha_5.0/rawdata/ImageNet/train/"""
            filename_train = dir_path + """train/idx_train_{}.pickle""".format(self.id)
            filename_test = dir_path + "test/idx_test_{}.picle""".format(self.id)

            transmforms = {'train': transforms.Compose(
                    [

                        transforms.Resize((32, 32)),
                        transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                        transforms.RandomRotation(10),  # Rotates the image to a specified angel
                        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                        # Performs actions like zooms, change shear angles.
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                    ]
                ), 'test': transforms.Compose(
                    [

                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
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

            np.random.seed(self.id)

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


            # validation_dataset.imgs = list(imgs_test)

            def seed_worker(worker_id):
                np.random.seed(self.cid)
                random.seed(self.cid)

            g = torch.Generator()
            g.manual_seed(self.cid)

            unique_count = {i: 0 for i in range(self.args.num_classes[m])}
            unique, count = np.unique(y, return_counts=True)
            data_unique_count_dict = dict(zip(unique, count))
            for class_ in data_unique_count_dict:
                unique_count[class_] = data_unique_count_dict[class_]
            unique_count = np.array(list(unique_count.values()))

            trainLoader = DataLoader(training_dataset, batch_size, shuffle=True, worker_init_fn=seed_worker,
                                     generator=g)
            testLoader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)

            if mode == "train":
                return trainLoader, unique_count
            else:
                return testLoader

        except Exception as e:
            print("load ImageNet")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def load_imagenet(m, cid, args, mode="train", batch_size=32, dataset_name=None):

    try:
        dir_path = "../dataset/ImageNet/" + "clients_" + str(args.num_clients) + "/alpha_" + str(args.alpha[m]) + "/"
        traindir = """/home/claudio/Documentos/pycharm_projects/Multi-model-Federated-Learning/dataset/ImageNet/clients_40/alpha_5.0/rawdata/ImageNet/train/"""
        filename_train = dir_path + """train/idx_train_{}.pickle""".format(cid)
        filename_test = dir_path + "test/idx_test_{}.pickle""".format(cid)

        transmforms = {'train': transforms.Compose(
                [

                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                    transforms.RandomRotation(10),  # Rotates the image to a specified angel
                    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                    # Performs actions like zooms, change shear angles.
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            ), 'test': transforms.Compose(
                [

                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

        np.random.seed(cid)

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

        def seed_worker(worker_id):
            np.random.seed(cid)
            random.seed(cid)

        g = torch.Generator()
        g.manual_seed(cid)

        unique_count = {i: 0 for i in range(args.num_classes[m])}
        unique, count = np.unique(y, return_counts=True)
        data_unique_count_dict = dict(zip(unique, count))
        for class_ in data_unique_count_dict:
            unique_count[class_] = data_unique_count_dict[class_]
        unique_count = np.array(list(unique_count.values()))

        trainLoader = DataLoader(training_dataset, batch_size, shuffle=True, worker_init_fn=seed_worker,
                                 generator=g)
        testLoader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)

        if mode == "train":
            return trainLoader, unique_count
        else:
            return testLoader

    except Exception as e:
        print("load ImageNet")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def read_client_data(m, idx, args={}, is_train=True):
    dataset = args.dataset[m]
    alpha = args.alpha[m]
    num_classes = args.num_classes[m]
    unique_count = {i: 0 for i in range(num_classes)}
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)
    elif "ImageNet" in dataset:
        return load_imagenet(m, idx, args)
    elif dataset in ["WISDM-W", "WISDM-P"]:
        return load_wisdm(m, dataset, idx, args)

    if is_train:
        train_data = read_data(dataset, idx, args, alpha, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        y = train_data['y']
        print("client ", idx, len(y))
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        unique, count = np.unique(y, return_counts=True)
        data_unique_count_dict = dict(zip(unique, count))
        for class_ in data_unique_count_dict:
            unique_count[class_] = data_unique_count_dict[class_]
        return train_data, np.array(list(unique_count.values()))
    else:
        test_data = read_data(dataset, idx, args, alpha, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, args={}, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, args, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, args, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, args={}, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, args, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, args, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

