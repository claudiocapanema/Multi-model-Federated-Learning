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
import torchvision
from torch.utils.data import TensorDataset, DataLoader


def read_data(dataset, idx, args, alpha, is_train=True):
    if is_train:
        path = """../dataset/{}/clients_{}/alpha_{}/train/""".format(dataset, args.num_clients, alpha)
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

def read_client_data_v2(m, name, cid, args, mode="train", batch_size=32):

    try:
        dir_path = """../dataset/{}/clients_{}/alpha_{}/""".format(name, args.num_clients, args.alpha[m])
        filename_train = dir_path + """train/idx_train_{}.pickle""".format(cid)
        filename_test = dir_path + "test/idx_test_{}.pickle""".format(cid)
        dataset_dir_path = """../dataset/{}/""".format(name)

        with open(filename_train, 'rb') as handle:
            idx_train = pickle.load(handle)

        with open(filename_test, 'rb') as handle:
            idx_test = pickle.load(handle)

        if name == "CIFAR10":
            trainset, testset = read_cifar10(dir_path)

            x = trainset.data
            x = np.concatenate((x, testset.data))
            y = trainset.targets
            y = np.concatenate((y, testset.targets))
            x_train = x[idx_train]
            x_test = x[idx_test]
            y_train = y[idx_train]
            y_test = y[idx_test]

            trainset.data = x_train
            trainset.targets = y_train
            testset.data = x_test
            testset.targets = y_test

            print("""Cliente {} dados treino {} dados teste {}""".format(cid, x_train.shape, x_test.shape))
            print("""Quantidade de cada classe\nTreino {} \nTeste {}""".format(np.unique_counts(y_train),
                                                                               np.unique_counts(y_test)))
            # exit()

            # trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).to(dtype=torch.float32),
            #                                                   torch.from_numpy(y_train))
            # testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).to(dtype=torch.float32),
            #                                                     torch.from_numpy(y_test))

        elif name == "EMNIST":
            trainset, testset = read_emnist(dataset_dir_path)
            # data = torch.concatenate([trainset.data, testset.data])
            # # data = trainset.data
            # labels = torch.concatenate([trainset.targets, testset.targets])
            # # data_train = torch.index_select(data, 0, torch.tensor(idx_train))
            # # data_test = torch.index_select(data, 0, torch.tensor(idx_test))
            # data_train = data[idx_train]
            # data_test = data[idx_test]
            # labels_train = labels[idx_train]
            # labels_test = labels[idx_test]
            # # labels_train = torch.index_select(labels, 0, torch.tensor(idx_train))
            # # labels_test = torch.index_select(labels, 0, torch.tensor(idx_test))
            # print("""Cliente {} dados treino {} dados teste {}""".format(cid, data_train.shape, data_test.shape))
            #
            # y_train = np.concatenate([labels_train.numpy(), labels_test.numpy()])
            #
            # trainset.data = data_train
            # testset.data = data_test
            # trainset.labels = labels_train
            # testset.labels = labels_test

            trainset.data = torch.concatenate((trainset.data, testset.data), axis=0)

            trainset.targets = torch.concatenate((trainset.targets, testset.targets), axis=0)
            testset.data = trainset.data[idx_test]
            testset.targets = trainset.targets[idx_test]
            trainset.data = trainset.data[idx_train]
            trainset.targets = trainset.targets[idx_train]
            y = trainset.targets
            y_train = y

            print("""Cliente {} dados treino {} dados teste {}""".format(cid, trainset.data.shape, testset.data.shape))
            print("""Quantidade de cada classe\nTreino {} \nTeste {}""".format(np.unique_counts(trainset.targets), np.unique_counts(testset.targets)))
            print(type(trainset.data), type(testset.data), type(trainset.targets), type(testset.targets))

        # print(type(trainset.data), type(testset.data[0]), trainset.data)
        # exit()

        # x = np.array(trainset.data)
        # x = np.concatenate([x, testset.data], axis=0)
        # y = np.array(trainset.targets)
        # y = np.concatenate([y, testset.targets], axis=0)

        # trainset.data = trainset.data[:100]
        # trainset.target = trainset.targets[:100]
        # print(trainset.data.numpy().shape)
        # print(trainset.data)
        # exit()


        # dataset_image.extend(testset.data.numpy().tolist())
        # dataset_label.extend(trainset.targets.numpy().tolist())
        # dataset_label.extend(testset.targets.numpy().tolist())
        # x = np.array(dataset_image)
        # y = np.array(dataset_label)

        # x_train = x[idx_train]
        # y_train = y[idx_train]
        # x_test = x[idx_test]
        # y_test = y[idx_test]

        y = np.array(list(y_train))

        # print("""Cliente {} dados treino {} dados teste {}""".format(cid, x_train.shape, x_test.shape))
        # exit()

        # trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).to(dtype=torch.float32),
        #                                                   torch.from_numpy(y_train))
        # testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).to(dtype=torch.float32),
        #                                                     torch.from_numpy(y_test))

        # trainset.data = torch.from_numpy(np.array(x_train)).to(torch.float32)
        # trainset.target = np.array(y_train)
        # testset.data = torch.from_numpy(np.array(x_test)).to(torch.float32)
        # testset.target = np.array(y_test)


        g = torch.Generator()
        g.manual_seed(cid)

        unique_count = {i: 0 for i in range(args.num_classes[m])}
        unique, count = np.unique(y, return_counts=True)
        data_unique_count_dict = dict(zip(unique, count))
        for class_ in data_unique_count_dict:
            unique_count[class_] = data_unique_count_dict[class_]
        unique_count = np.array(list(unique_count.values()))

        np.random.seed(cid)

        def seed_worker(worker_id):
            np.random.seed(cid)
            random.seed(cid)

        trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                                 generator=g)
        testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker,
                                 generator=g)
        # if name == "EMNIST":
        #     x_train = np.expand_dims(x_train, axis=1)
        #     x_test = np.expand_dims(x_test, axis=1)
        # print("x train: ", x_train.shape, y_train.shape)


        # for i, (x, y) in enumerate(trainloader):
        #     if i == 0:
        #         print("aaaa_oi: ", x.shape, y.shape)
        #         exit()
        # trainset, trainloader, testset, testloader = create_torch_dataset_from_numpy(x_train, x_test, y_train, y_test, batch_size=batch_size, g=g, cid=cid)

        # print("test loader")
        # for x, y in testloader:
        #     print(x.shape, y.shape)
        #     exit()


        if mode == "train":
            return trainloader, unique_count
        else:
            return testloader

    except Exception as e:
        print("ead_client_data_v2")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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

def load_imagenet(m, cid, args, mode="train", batch_size=32, dataset_name=None):

    try:
        dir_path = "../dataset/ImageNet/" + "clients_" + str(args.num_clients) + "/alpha_" + str(args.alpha[m]) + "/"
        traindir = """../dataset/ImageNet/rawdata/ImageNet/train/"""
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
        print("load ImageNet data utils")
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


def read_cifar10(dir_path):
    transform = {'train': transforms.Compose(
        [transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
         transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
         transforms.RandomRotation(10),  # Rotates the image to a specified angel
         transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
         # Performs actions like zooms, change shear angles.
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
         transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # Normalize all the images
    ), 'test': transforms.Compose([
                                transforms.Resize((32, 32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )}

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=True, download=True, transform=transform["train"])
    testset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=False, download=True, transform=transform["test"])

    # trainset.data = np.moveaxis(trainset.data, 1, 3).tolist()
    # testset.data = np.moveaxis(testset.data, 1, 3).tolist()

    return trainset, testset


def read_emnist(dir_path):
    transform = {'train': transforms.Compose(
        [
            transforms.ToTensor(), transforms.RandomRotation(10),
             transforms.Normalize([0.5], [0.5])
        ]
    ), 'test': transforms.Compose(
        [
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
        ]
    )}

    trainset = torchvision.datasets.EMNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform["train"], split='balanced')
    testset = torchvision.datasets.EMNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform["test"], split='balanced')

    return trainset, testset

def read_gtsrb(m, name, cid, args, t, mode="train", batch_size=32):

    dir_path = """../dataset/{}/clients_{}/alpha_{}/""".format(name, args.num_clients, args.alpha[m])
    filename_train = dir_path + """train/idx_train_{}.pickle""".format(cid)
    filename_test = dir_path + "test/idx_test_{}.pickle""".format(cid)
    file_dir_path = "../dataset/GTSRB/rawdata/"

    print(file_dir_path + "Train")

    trainset = datasets.ImageFolder(
        file_dir_path + "Train",
        transforms.Compose(
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
        )
    )

    valset = datasets.ImageFolder(
        file_dir_path + "Train",
        transforms.Compose(
            [

                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ]
        )
    )

    np.random.seed(cid)

    dataset_image = []
    dataset_samples = []
    dataset_label = []
    dataset_samples.extend(trainset.samples)
    dataset_image.extend(trainset.imgs)
    dataset_label.extend(trainset.targets)
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    with open(filename_train, 'rb') as handle:
        idx_train = pickle.load(handle)

    with open(filename_test, 'rb') as handle:
        idx_test = pickle.load(handle)

    # imgs = trainset.imgs
    # x_train = []
    # x_test = []
    # y_train = []
    # y_test = []
    # for i in range(1):
    #     x_train += trainset.samples
    #     x_test += valset.samples
    #     y_train += trainset.targets
    #     y_test += valset.targets

    x_train = np.array(trainset.samples)
    y_train = np.array(trainset.targets)
    x_test = np.array(valset.samples)
    y_test = np.array(valset.targets)
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    print("""cliente {} treino {} teste {}""".format(cid, len(y_train), len(y_test)))

    trainset.samples = list(x_train)
    trainset.targets = list(y_train)
    valset.samples = list(x_test)
    valset.targets = list(y_test)

    g = torch.Generator()
    g.manual_seed(cid)

    unique_count = {i: 0 for i in range(args.num_classes[m])}
    unique, count = np.unique(np.concatenate([y_test, y_test]), return_counts=True)
    data_unique_count_dict = dict(zip(unique, count))
    for class_ in data_unique_count_dict:
        unique_count[class_] = data_unique_count_dict[class_]
    unique_count = np.array(list(unique_count.values()))

    np.random.seed(cid)

    def seed_worker(worker_id):
        np.random.seed(cid)
        random.seed(cid)

    g = torch.Generator()
    g.manual_seed(cid)

    trainloader = DataLoader(trainset, batch_size, shuffle=True, worker_init_fn=seed_worker,
                             generator=g)
    testloader = DataLoader(dataset=valset, batch_size=32, shuffle=False)

    if mode == "train":
        return trainloader, unique_count
    else:
        return testloader

# def create_torch_dataset_from_numpy(x_train, x_test, y_train, y_test, batch_size, g, cid):
#
#     def seed_worker(worker_id):
#         np.random.seed(cid)
#         random.seed(cid)
#
#     tensor_x_train = torch.Tensor(x_train)  # transform to torch tensor
#     tensor_y_train = torch.Tensor(y_train)
#
#
#     train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
#     trainLoader = DataLoader(train_dataset, batch_size, shuffle=True)
#     # , worker_init_fn=seed_worker, generator=g
#
#     tensor_x_test = torch.Tensor(x_test)  # transform to torch tensor
#     tensor_y_test = torch.Tensor(y_test)
#
#     test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
#     testLoader = DataLoader(test_dataset, batch_size, shuffle=False)
#
#     return train_dataset, trainLoader, test_dataset, testLoader

def create_torch_dataset_from_numpy(x_train, x_test, y_train, y_test, batch_size, g, cid):

    tensor_x_train = torch.from_numpy(x_train)  # transform to torch tensor
    tensor_y_train = torch.from_numpy(y_train)

    def seed_worker(worker_id):
        np.random.seed(cid)
        random.seed(cid)

    trainset = TensorDataset(tensor_x_train, tensor_y_train)
    trainloader = DataLoader(trainset, batch_size, shuffle=True)

    tensor_x_test = torch.from_numpy(x_test)  # transform to torch tensor
    tensor_y_test = torch.from_numpy(y_test)

    testset = TensorDataset(tensor_x_test, tensor_y_test)
    testloader = DataLoader(testset, batch_size, shuffle=False)

    return trainset, trainloader, testset, testloader
