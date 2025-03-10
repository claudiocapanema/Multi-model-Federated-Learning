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
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random

batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

###########################################################

# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class HARCNN(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=64*26, num_classes=6, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# https://github.com/FengHZ/KD3A/blob/master/model/digit5.py
class Digit5CNN(nn.Module):
    def __init__(self):
        super(Digit5CNN, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn1", nn.BatchNorm2d(64))
        self.encoder.add_module("relu1", nn.ReLU())
        self.encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn2", nn.BatchNorm2d(64))
        self.encoder.add_module("relu2", nn.ReLU())
        self.encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn3", nn.BatchNorm2d(128))
        self.encoder.add_module("relu3", nn.ReLU())

        self.linear = nn.Sequential()
        self.linear.add_module("fc1", nn.Linear(8192, 3072))
        self.linear.add_module("bn4", nn.BatchNorm1d(3072))
        self.linear.add_module("relu4", nn.ReLU())
        self.linear.add_module("dropout", nn.Dropout())
        self.linear.add_module("fc2", nn.Linear(3072, 2048))
        self.linear.add_module("bn5", nn.BatchNorm1d(2048))
        self.linear.add_module("relu5", nn.ReLU())

        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return out
        

# https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
class AmazonMLP(nn.Module):
    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 1000), 
            # nn.BatchNorm1d(1000), 
            nn.ReLU(), 
            nn.Linear(1000, 500), 
            # nn.BatchNorm1d(500), 
            nn.ReLU(),
            nn.Linear(500, 100), 
            # nn.BatchNorm1d(100), 
            nn.ReLU()
        )
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out
        

# # https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
# class FedAvgCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features,
#                                32,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(32,
#                                64,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.fc1 = nn.Linear(dim, 512)
#         self.fc = nn.Linear(512, num_classes)

#         self.act = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.act(self.conv2(x))
#         x = self.maxpool(x)
#         x = torch.flatten(x, 1)
#         x = self.act(self.fc1(x))
#         x = self.fc(x)
#         return x

class FedAvgCNN(nn.Module):
    def __init__(self, dataset, in_features=1, num_classes=10, dim=1024):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        super().__init__()
        self.dataset = dataset
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_features,
        #               32,
        #               kernel_size=5,
        #               padding=0,
        #               stride=1,
        #               bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2))
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32,
        #               64,
        #               kernel_size=5,
        #               padding=0,
        #               stride=1,
        #               bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2))
        # )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(dim, 512),
        #     nn.ReLU(inplace=True)
        # )
        # self.fc = nn.Linear(512, num_classes)

        self.conv1 = nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True)
        self.conv2 = nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True)
        self.fc1 = nn.Linear(dim, 512)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # random.seed(0)
        # np.random.seed(0)
        # torch.manual_seed(0)
        # if self.dataset == "CIFAR10":
        #     x = x.permute(0, 3, 1, 2)
        # elif self.dataset == "EMNIST":
        #     # [1, 32, 28, 28]
        #     # x = x.permute(0, 1, 2)
        #     # x = torch.reshape(x, (32, 1, 28, 28))
        #     pass

        out = nn.MaxPool2d(kernel_size=(2, 2))(nn.ReLU(inplace=True)(self.conv1(x)))
        out = self.dropout1(out)
        out = nn.MaxPool2d(kernel_size=(2, 2))(nn.ReLU(inplace=True)(self.conv2(out)))
        out = self.dropout2(out)
        out = torch.flatten(out, 1)
        out = nn.ReLU(inplace=True)(self.fc1(out))
        out = self.dropout3(out)
        out = self.fc(out)

        # out = self.conv1(x)
        # out = self.conv2(out)
        # out = torch.flatten(out, 1)
        # out = self.fc1(out)
        # out = self.fc(out)
        return out

    def forward_kd(self, x):
        # random.seed(0)
        # np.random.seed(0)
        # torch.manual_seed(0)
        # if self.dataset == "CIFAR10":
        #     x = x.permute(0, 3, 1, 2)
        # elif self.dataset == "EMNIST":
        #     # [1, 32, 28, 28]
        #     # x = x.permute(0, 1, 2)
        #     # x = torch.reshape(x, (32, 1, 28, 28))
        #     pass

        out = nn.MaxPool2d(kernel_size=(2, 2))(nn.ReLU(inplace=True)(self.conv1(x)))
        out = nn.MaxPool2d(kernel_size=(2, 2))(nn.ReLU(inplace=True)(self.conv2(out)))
        out = torch.flatten(out, 1)
        rep = nn.ReLU(inplace=True)(self.fc1(out))
        out = self.fc(rep)

        # out = self.conv1(x)
        # out = self.conv2(out)
        # out = torch.flatten(out, 1)
        # out = self.fc1(out)
        # out = self.fc(out)
        return out, rep

class FedAvgCNNStudent(nn.Module):
    def __init__(self, dataset, in_features=1, num_classes=10, dim=1024):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        super().__init__()
        self.dataset = dataset

        # self.conv2 = nn.Conv2d(in_features,
        #               32,
        #               kernel_size=5,
        #               padding=0,
        #               stride=1,
        #               bias=True)
        # self.conv3 = nn.Conv2d(32,
        #                        32,
        #                        kernel_size=5,
        #                        padding=0,
        #                        stride=1,
        #                        bias=True)
        # dim = {"EMNIST": 512, "CIFAR10": 800, "GTSRB": 800}[dataset]
        # self.fc1 = nn.Linear(dim, 512)
        # self.fc = nn.Linear(512, num_classes)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True))
        self.out = nn.Linear(512, num_classes)
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):

        # out = nn.MaxPool2d(kernel_size=(2, 2))(nn.ReLU(inplace=True)(self.conv2(x)))
        # out = nn.MaxPool2d(kernel_size=(2, 2))(nn.ReLU(inplace=True)(self.conv3(out)))
        # out = torch.flatten(out, 1)
        # out = nn.ReLU(inplace=True)(self.fc1(out))
        # out = self.fc(out)
        rep = self.conv1(x)
        out = self.out(rep)

        return out

    def forward_kd(self, x):
        # out = nn.MaxPool2d(kernel_size=(2, 2))(nn.ReLU(inplace=True)(self.conv2(x)))
        # out = nn.MaxPool2d(kernel_size=(2, 2))(nn.ReLU(inplace=True)(self.conv3(out)))
        # out = torch.flatten(out, 1)
        # rep = nn.ReLU(inplace=True)(self.fc1(out))
        # out = self.fc(rep)
        rep = self.conv1(x)
        out = self.out(rep)

        return out, rep

class FedAvgCNNKD(nn.Module):
    def __init__(self, dataset, in_features=1, num_classes=10, dim=1024):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        super().__init__()
        self.dataset = dataset

        teacher_dim = {"GTSRB": 256}[dataset]
        student_dim = {"GTSRB": 7200}[dataset]
        self.teacher = CNN_3_proto(dataset, in_features=in_features, num_classes=num_classes, dim=teacher_dim)
        self.student = FedAvgCNNStudent(dataset, in_features=in_features, num_classes=num_classes, dim=student_dim)

    def forward(self, x):

        out, rep = self.teacher.forward_kd(x)

        return out

    def forward_kd(self, x):

        out_teacher, rep_teacher = self.teacher.forward_kd(x)
        out_student, rep_student = self.student.forward_kd(x)

        return out_student, rep_student, out_teacher, rep_teacher

class CNN_3_proto(torch.nn.Module):
    def __init__(self, dataset, in_features, dim, num_classes=10):

        try:
            super(CNN_3_proto, self).__init__()

                # queda para asl
                # nn.Conv2d(input_shape, 32, kernel_size=3, padding=1),
                # nn.ReLU(),
                # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
                #
                # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
                # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
                #
                # nn.Flatten(),
                # nn.Linear(mid_dim,512),
                # nn.ReLU(),
                # nn.Linear(512, num_classes))

                # nn.Linear(28*28, 392),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                # nn.Linear(392, 196),
                # nn.ReLU(),
                # nn.Linear(196, 98),
                # nn.ReLU(),
                # nn.Dropout(0.3),
                # nn.Linear(98, num_classes)

            self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=3, padding=1),
                                             torch.nn.ReLU(),
                                             # Input = 32 x 32 x 32, Output = 32 x 16 x 16
                                             torch.nn.MaxPool2d(kernel_size=2),

                                             # Input = 32 x 16 x 16, Output = 64 x 16 x 16
                                             torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                             torch.nn.ReLU(),
                                             # Input = 64 x 16 x 16, Output = 64 x 8 x 8
                                             torch.nn.MaxPool2d(kernel_size=2),

                                             # Input = 64 x 8 x 8, Output = 64 x 8 x 8
                                             torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                             torch.nn.ReLU(),
                                             # Input = 64 x 8 x 8, Output = 64 x 4 x 4
                                             torch.nn.MaxPool2d(kernel_size=2),
                                             torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),

                                             torch.nn.ReLU(),
                                             # Input = 64 x 8 x 8, Output = 64 x 4 x 4
                                             torch.nn.MaxPool2d(kernel_size=2),
                                             torch.nn.Flatten(),
                                             torch.nn.Linear(dim, 512))

            self.fc = torch.nn.Linear(512, num_classes)

        except Exception as e:
            print("CNN_3_proto")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.fc(proto)
            return out, proto
        except Exception as e:
            print("forward CNN_3_proto")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward_kd(self, x):
        try:
            proto = self.conv1(x)
            out = self.fc(proto)
            return out, proto
        except Exception as e:
            print("forward CNN_3_proto_kd")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

class TinyImageNetCNN(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
    #     self.pool = nn.MaxPool2d(2)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    #     self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
    #     self.fc1 = nn.Linear(128 * 4 * 4, 256)
    #     self.fc = nn.Linear(256, 200)
    #
    # def forward(self, x):
    #     x = self.pool(self.conv1(x))
    #     x = self.pool(self.conv2(x))
    #     x = self.pool(self.conv3(x))
    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = nn.ReLU()(x)
    #     x = self.fc(x)
    #     return x

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(400, 12)
        # self.fc = nn.Linear(200, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        # x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        return x

#     self.net = torchvision.models.squeezenet1_0(pretrained=True)
#     self.fc = nn.Linear(1000, 200)
#
#
# def forward(self, x):
#     x = self.net(x)
#     x = self.fc(x)
#     return x


# ====================================================================================================================

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

# ====================================================================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
        self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1*28*28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ====================================================================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNet(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]
                            
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

class CNN_2(torch.nn.Module):
    def __init__(self, input_shape, mid_dim=64, num_classes=10):
        super().__init__()
        self.model = torch.nn.Sequential(
            # Input = 3 x 32 x 32, Output = 32 x 32 x 32
            torch.nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 32 x 32 x 32, Output = 32 x 16 x 16
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 32 x 16 x 16, Output = 64 x 16 x 16
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 64 x 16 x 16, Output = 64 x 8 x 8
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),
            torch.nn.Linear(mid_dim * 4 * 4, 512),
            torch.nn.ReLU()
        )

        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        a = self.model(x)


        return self.fc(a)

# ====================================================================================================================

class GRU(torch.nn.Module):
    def __init__(self, input_shape, num_layers=1, hidden_size=2, sequence_length=28, num_classes=10):
        super().__init__()
        try:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            self.input_size = input_shape
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = num_classes
            self.time_length = sequence_length

            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
            self.dp = nn.Dropout(0.5)
            self.fc = nn.Linear(self.time_length * self.hidden_size, self.output_size, bias=True)
        except Exception as e:
            print("GRU init")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            x, h = self.gru(x)
            x = nn.Flatten()(x)
            x = self.dp(x)
            out = self.fc(x)
            return out
        except Exception as e:
            print("GRU forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================

class LSTM(torch.nn.Module):
    def __init__(self, input_shape, device, num_layers=1, hidden_size=2, sequence_length=28, num_classes=10):
        super().__init__()
        try:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            self.input_size = input_shape
            self.device = device
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = num_classes
            self.time_length = sequence_length

            self.embedding_category = nn.Embedding(num_embeddings=7, embedding_dim=6)
            self.embedding_hour = nn.Embedding(num_embeddings=48, embedding_dim=15)
            # self.embedding_distance = nn.Embedding(num_embeddings=51, embedding_dim=15)
            # self.embedding_duration = nn.Embedding(num_embeddings=49, embedding_dim=15)

            self.lstm = nn.LSTM(23, self.hidden_size, self.num_layers, batch_first=True)
            self.dp = nn.Dropout(0.5)
            self.fc = nn.Linear(self.time_length * self.hidden_size, self.output_size, bias=True)
        except Exception as e:
            print("LSTM init")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            # print("entrada: ", x.shape)
            x = x.int()
            if self.device == 'cuda':
                x = x.int().cuda()

            category, hour, distance, duration = torch.split(x, [1, 1, 1, 1], dim=-1)
            # category = x[0]
            # hour = x[1]
            # distance = x[2]
            # duration = x[3]
            #print("dim en: ", category.shape, hour.shape, distance.shape, duration.shape)
            # print("valores: ", hour)
            category_embbeded = self.embedding_category(category)
            hour_embedded = self.embedding_hour(hour)
            # distance_embedded = self.embedding_distance(distance)
            # duration_embedded = self.embedding_duration(duration)

            # Concatenando os embeddings com os dados reais
            #print("dim: ", category_embbeded.shape, hour_embedded.shape, distance_embedded.shape, duration_embedded.shape)
            #print("dev: ", category_embbeded.device, hour_embedded.device, distance_embedded.device, duration_embedded.device)
            combined_embedded = torch.cat((category_embbeded, hour_embedded), dim=-1)
            # print("comb: ", combined_embedded.shape)
            # if combined_embedded.shape[0] > 1:
            combined_embedded = combined_embedded.squeeze()
            if combined_embedded.dim() == 2:
                combined_embedded = combined_embedded.unsqueeze(0)
            # print("dimensoes: ", combined_embedded.shape, distance.shape, duration.shape)
            combined_embedded = torch.cat((combined_embedded, distance, duration), dim=-1)
            # print("co dim: ", combined_embedded.shape)
            x, h = self.lstm(combined_embedded)
            #print("sai gru: ", x.shape)
            x = nn.Flatten()(x)
            x = self.dp(x)
            #print("e2: ", x.shape)
            if x.shape[1] == 1:
                x = x.rot90(1, dims=(0, 1))
            #print("e3: ", x.shape)
            out = self.fc(x)
            #print("sai lstm: ", out.shape)
            return out
        except Exception as e:
            print("LSTM forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================

class LSTM_NET(nn.Module):
    """Class to design a LSTM model."""

    def __init__(self, input_dim=6, hidden_dim=6, time_length=200, num_classes=12):
        """Initialisation of the class (constructor)."""
        # Input:
        # input_dim, integer
        # hidden_dim; integer
        # time_length; integer

        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Sequential(nn.Flatten(),
                                 nn.Dropout(0.2),
                                 nn.Linear(time_length * hidden_dim, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, num_classes))

    def forward(self, input_data):
        """The layers are stacked to transport the data through the neural network for the forward part."""
        # Input:
        # input_data; torch.Tensor
        # Output:
        # x; torch.Tensor

        x, h = self.lstm(input_data)
        x = self.fc(x)

        return x

# ====================================================================================================================

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        dims = hidden_dim*2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        text, text_lengths = x
        
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:,-1,:])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
            
        return out

# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out

# ====================================================================================================================

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.8, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels*len(kernel_size), num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text).permute(0,2,1)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out

# ====================================================================================================================

class CNN_3(torch.nn.Module):
    def __init__(self, dataset, in_features, dim=64, num_classes=10):
        super().__init__()
        self.model = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 32 x 32 x 32, Output = 32 x 16 x 16
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 32 x 16 x 16, Output = 64 x 16 x 16
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 64 x 16 x 16, Output = 64 x 8 x 8
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),
            torch.nn.Linear(dim * 4 * 4, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        )


    def forward(self, x):
        return self.model(x)