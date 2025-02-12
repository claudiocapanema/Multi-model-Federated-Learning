from xml.sax.expatreader import version


import torch
import numpy as np
import random
from torch.utils.data import TensorDataset
import os
import torchvision
import torchvision.transforms as transforms
import subprocess
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.datasets as datasets
from PIL import Image
from torch.utils import data

def load_data_eminist(data_path):
    """Load Emnist (training and val set)."""

    # Load ImageNet and normalize
    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = datasets.ImageFolder(
        traindir,
        transform
    )

    valset = datasets.ImageFolder(
        valdir,
        transform
    )

    return trainset, valset

def load_data_imagenet(data_path):
    """Load ImageNet (training and val set)."""

    # Load ImageNet and normalize
    traindir = data_path
    valdir = traindir

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    valset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    return trainset, valset

def load_data_statefarm(data_path):
    """Load ImageNet (training and val set)."""

    # Load ImageNet and normalize
    traindir = os.path.join(data_path, "train")

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomRotation(degrees=60, expand=False),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    valset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    print(trainset.targets)

    return trainset, None

def load_data_gtsrb(data_path):
    """Load ImageNet (training and val set)."""

    # Load ImageNet and normalize
    traindir = os.path.join(data_path, "train")

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomRotation(degrees=60, expand=False),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    return trainset, None

def get_filepath(dir_root):
    ''''获取一个目录下所有文件的路径，并存储到List中'''
    file_paths = []
    for root, dirs, files in os.walk(dir_root):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

class DriverDataset(data.Dataset):

    def __init__(self, data_root, transform=None, train=True):
        self.train = train
        imgs_in = get_filepath(data_root)

        if transform is None and self.train:
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                         transforms.RandomRotation(degrees=60, expand=False),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
        else:
            self.transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
        self.imgs = imgs_in

        self.samples, self.labels = self.ler()

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = int(img_path.split('/')[-2][1])
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def ler(self):

        samples = []
        labels = []
        for i in range(len(self.imgs)):
            img_path = self.imgs[i]
            label = int(img_path.split('/')[-2][1])
            data = Image.open(img_path)
            samples.append(data)
            labels.append(label)

        return self.transforms(np.array(samples)), labels

    def get_targets(self):

        labels = []
        for i in range(len(self.imgs)):
            labels.append(self.imgs[i][1])

        return labels

    def __len__(self):
        return len(self.imgs)

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not os.listdir(self.root):
            print("entro")
            command = """cd {} \nwget http://cs231n.stanford.edu/tiny-imagenet-200.zip""".format(self.root)
            subprocess.Popen(command, shell=True).wait()
            command = """cd {} \nunzip tiny-imagenet-200.zip""".format(self.root)
            subprocess.Popen(command, shell=True).wait()
        elif not os.path.exists(self.root + "/tiny-imagenet-200/val/"):
            print("aaa")
            command = """cd {} \nunzip tiny-imagenet-200.zip""".format(self.root)
            subprocess.Popen(command, shell=True).wait()
        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

class ManageDatasets():

    def __init__(self):
        random.seed(0)

    # def load_UCIHAR(self):
    #     with open(f'data/UCI-HAR/{self.cid + 1}_train.pickle', 'rb') as train_file:
    #         train = pickle.load(train_file)
    #
    #     with open(f'data/UCI-HAR/{self.cid + 1}_test.pickle', 'rb') as test_file:
    #         test = pickle.load(test_file)
    #
    #     train['label'] = train['label'].apply(lambda x: x - 1)
    #     y_train = train['label'].values
    #     train.drop('label', axis=1, inplace=True)
    #     x_train = train.values
    #
    #     test['label'] = test['label'].apply(lambda x: x - 1)
    #     y_test = test['label'].values
    #     test.drop('label', axis=1, inplace=True)
    #     x_test = test.values
    #     print("exemplo ucihar: ", x_test.shape, x_train.shape)
    #
    #     return x_train, y_train, x_test, y_test
    #
    # def load_MotionSense(self):
    #     with open(f'data/motion_sense/{self.cid + 1}_train.pickle', 'rb') as train_file:
    #         train = pickle.load(train_file)
    #
    #     with open(f'data/motion_sense/{self.cid + 1}_test.pickle', 'rb') as test_file:
    #         test = pickle.load(test_file)
    #
    #     y_train = train['activity'].values
    #     train.drop('activity', axis=1, inplace=True)
    #     train.drop('subject', axis=1, inplace=True)
    #     train.drop('trial', axis=1, inplace=True)
    #     x_train = train.values
    #
    #     y_test = test['activity'].values
    #     test.drop('activity', axis=1, inplace=True)
    #     test.drop('subject', axis=1, inplace=True)
    #     test.drop('trial', axis=1, inplace=True)
    #     x_test = test.values
    #     print("exemplo motion: ", x_test.shape, x_train.shape)
    #
    #     return x_train, y_train, x_test, y_test

    def load_MNIST(self):

        dir_path = "data/MNIST/"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

            # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # Get EMNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        trainset = torchvision.datasets.MNIST(
            root=dir_path + "rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root=dir_path + "rawdata", train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        return dataset_image, dataset_label, np.array([]), np.array([])

    def load_CIFAR10(self):

        dir_path = "CIFAR10/"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

            # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # Get EMNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        trainset = torchvision.datasets.CIFAR10(
            root=dir_path + "rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=dir_path + "rawdata", train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        return dataset_image, dataset_label, np.array([]), np.array([])

    def load_CIFAR100(self):
        dir_path = "CIFAR100/"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

            # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # Get EMNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        trainset = torchvision.datasets.CIFAR100(
            root=dir_path + "rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(
            root=dir_path + "rawdata", train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        return dataset_image, dataset_label, np.array([]), np.array([])

    def load_imagenet(self, dataset_name):

        num_clients = 40
        alpha = 0.1
        dir_path = dataset_name + "/clients_" + str(num_clients) + "/alpha_" + str(alpha) + "/"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

            # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        trainset, valset = load_data_imagenet("""{}/rawdata/{}/train/""".format(dataset_name, dataset_name))

        # trainset = ImageFolder_custom(root=dir_path + '', transform=transform)
        # testset = ImageFolder_custom(root=dir_path + '', transform=transform)
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=len(trainset), shuffle=False)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=len(testset), shuffle=False)
        #
        # print("sam: ", trainset.classes)
        #
        # # for _, train_data in enumerate(trainloader, 0):
        # #     print("oi: ", train_data)
        # #     exit()
        # # for _, test_data in enumerate(testloader, 0):
        # #     testset.data, testset.targets = test_data
        # exit()
        np.random.seed(0)

        dataset_image = []
        dataset_label = []
        dataset_image.extend(trainset.imgs)
        # dataset_image.extend(valset.imgs)
        dataset_label.extend(trainset.targets)
        # dataset_label.extend(valset.targets)
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        print("rotulos: ", dataset_label, dataset_label[0])

        return dataset_image, dataset_label, np.array([]), np.array([])

    def load_statefarm(self):

        dir_path = "data/state-farm-distracted-driver-detection/imgs/"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

            # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        # if not os.listdir(dir_path):
        #     print("entro")
        #     command = """cd {} \nwget http://cs231n.stanford.edu/tiny-imagenet-200.zip""".format(dir_path)
        #     subprocess.Popen(command, shell=True).wait()
        #     command = """cd {} \nunzip tiny-imagenet-200.zip""".format(dir_path)
        #     subprocess.Popen(command, shell=True).wait()
        # elif not os.path.exists(dir_path + "tiny-imagenet-200/val/"):
        #     print("aaa")
        #     command = """cd {} \nunzip 'tiny-imagenet-200.zip'""".format(dir_path)
        #     subprocess.Popen(command, shell=True).wait()

        trainset, valset = load_data_statefarm(dir_path)

        # trainset = ImageFolder_custom(root=dir_path + '', transform=transform)
        # testset = ImageFolder_custom(root=dir_path + '', transform=transform)
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=len(trainset), shuffle=False)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=len(testset), shuffle=False)
        #
        # print("sam: ", trainset.classes)
        #
        # # for _, train_data in enumerate(trainloader, 0):
        # #     print("oi: ", train_data)
        # #     exit()
        # # for _, test_data in enumerate(testloader, 0):
        # #     testset.data, testset.targets = test_data
        # exit()
        np.random.seed(0)

        dataset_image = []
        dataset_label = []
        dataset_image.extend(trainset.samples)
        dataset_label.extend(trainset.targets)
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        print("rotulos: ", dataset_label, dataset_label[0])

        return dataset_image, dataset_label, np.array([]), np.array([])

    def load_gtsrb(self):

        file_dir_path = "GTSRB/rawdata/"

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

        np.random.seed(0)

        dataset_image = []
        dataset_samples = []
        dataset_label = []
        dataset_samples.extend(trainset.samples)
        dataset_image.extend(trainset.imgs)
        dataset_label.extend(trainset.targets)
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        return dataset_image, dataset_label, np.array([]), np.array([])

    def load_emnist(self):

        dir_path = "EMNIST/"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

            # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # Get EMNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        trainset = torchvision.datasets.EMNIST(
            root=dir_path + "rawdata", train=True, download=True, transform=transform, split='balanced')
        testset = torchvision.datasets.EMNIST(
            root=dir_path + "rawdata", train=False, download=True, transform=transform, split='balanced')
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        return dataset_image, dataset_label, np.array([]), np.array([])

    def slipt_dataset(self, x_train, y_train, x_test, y_test, n_clients):
        p_train = int(len(x_train) / n_clients)
        p_test = int(len(x_test) / n_clients)

        random.seed(self.cid)
        selected_train = random.sample(range(len(x_train)), p_train)

        random.seed(self.cid)
        selected_test = random.sample(range(len(x_test)), p_test)

        x_train = x_train[selected_train]
        y_train = y_train[selected_train]

        x_test = x_test[selected_test]
        y_test = y_test[selected_test]

        return x_train, y_train, x_test, y_test

    def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        # FIX HTTP Error 403: Forbidden
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # Get MNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        trainset = torchvision.datasets.EMNIST(
            root=dir_path + "rawdata", train=True, download=True, transform=transform, split='balanced')
        testset = torchvision.datasets.EMNIST(
            root=dir_path + "rawdata", train=False, download=True, transform=transform, split='balanced')
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)


    def select_dataset(self, dataset_name):

        if dataset_name == 'MNIST':
            return self.load_MNIST()

        elif dataset_name == 'CIFAR100':
            return self.load_CIFAR100()

        elif dataset_name == 'CIFAR10':
            return self.load_CIFAR10()

        elif dataset_name == 'ImageNet':
            return self.load_imagenet(dataset_name)

        elif dataset_name == 'ImageNet_v2':
            return self.load_imagenet(dataset_name)

        elif dataset_name == "State Farm":
            return self.load_statefarm()

        elif dataset_name == 'GTSRB':
            return self.load_gtsrb()

        elif dataset_name == 'EMNIST':
            return self.load_emnist()

        # elif dataset_name == 'MotionSense':
        #     return self.load_MotionSense()
        #
        # elif dataset_name == 'UCIHAR':
        #     return self.load_UCIHAR()