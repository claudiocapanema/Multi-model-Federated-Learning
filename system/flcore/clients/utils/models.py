from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomAffine, ColorJitter,  Normalize, ToTensor, RandomRotation
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import sys


import logging

import random

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Class for the utils. In this case, we are using the MobileNetV2 utils from Keras
# class CNN(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
#
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class CNN(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            self.mid_dim = mid_dim
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
                          32,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32,
                          64,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.fc1 = nn.Sequential(
                nn.Linear(mid_dim, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, num_classes)
        except Exception as e:
            logger.info("CNN")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc(out)
            return out
        except Exception as e:
            logger.info("""CNN forward {}""".format(self.mid_dim))
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class CNN_3(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_3, self).__init__()
            self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Input = 32 x 32 x 32, Output = 32 x 16 x 16
                torch.nn.MaxPool2d(kernel_size=2))

                # Input = 32 x 16 x 16, Output = 64 x 16 x 16
            self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Input = 64 x 16 x 16, Output = 64 x 8 x 8
                torch.nn.MaxPool2d(kernel_size=2))

                # Input = 64 x 8 x 8, Output = 64 x 8 x 8
            self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Input = 64 x 8 x 8, Output = 64 x 4 x 4
                torch.nn.MaxPool2d(kernel_size=2))
            self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # Input = 64 x 8 x 8, Output = 64 x 4 x 4
                torch.nn.MaxPool2d(kernel_size=2))

            self.fc1 = torch.nn.Sequential(torch.nn.Linear(mid_dim * 4 * 4, 512),
                torch.nn.ReLU())
            self.fc2 = torch.nn.Linear(512, num_classes)

        except Exception as e:

            logger.info("CNN_3 init")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc2(out)
            return out
        except Exception as e:
            logger.info("CNN_3 forward")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class CNN_3_proto(torch.nn.Module):
    def __init__(self, input_shape, mid_dim=64, num_classes=10):

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

            self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, padding=1),
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
            torch.nn.Linear(mid_dim * 4 * 4, 512))

            self.fc = torch.nn.Linear(512, num_classes)

        except Exception as e:
            logger.info("CNN_3_proto")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.fc(proto)
            return out, proto
        except Exception as e:
            logger.info("CNN_3_proto")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class CNN_student(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_student, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
                          32,
                          kernel_size=3,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Flatten(),
                nn.Linear(mid_dim * 4, 512))
            # self.out = nn.Linear(512, num_classes)
            # self.conv1 = nn.Sequential(
            #     nn.Conv2d(input_shape,
            #               32,
            #               kernel_size=3,
            #               padding=0,
            #               stride=1,
            #               bias=True),
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=(2, 2)),
            #     torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            #     torch.nn.ReLU(),
            #     # Input = 64 x 16 x 16, Output = 64 x 8 x 8
            #     torch.nn.MaxPool2d(kernel_size=2),
            #     nn.Flatten(),
            #     nn.Linear(mid_dim * 4, 512),
            #     nn.ReLU(inplace=True))
            self.out = nn.Linear(512, num_classes)
        except Exception as e:
            logger.info("CNN student")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.out(proto)
            return out, proto
        except Exception as e:
            logger.info("CNN student forward")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class CNNDistillation(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10, dataset='CIFAR10'):
        try:
            self.dataset = dataset
            super(CNNDistillation, self).__init__()
            self.new_client = False
            if self.dataset in ['EMNIST', 'MNIST']:
                # mid_dim = 1568
                mid_dim = 1352 # CNN 1 pad 1
                # mid_dim = 400
            else:
                # mid_dim = 400
                mid_dim = 1800 # cnn student 1 cnn
                # mid_dim = 576 # cnn student 2 cnn
            self.student = CNN_student(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes)
            if self.dataset in ['CIFAR10', 'GTSRB']:
                mid_dim = 16
            else:
                mid_dim = 4
            self.teacher = CNN_3_proto(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes)
        except Exception as e:
            logger.info("CNNDistillation")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            out_student, proto_student = self.student(x)
            out_teacher, proto_teacher = self.teacher(x)
            return out_student, proto_student, out_teacher, proto_teacher
        except Exception as e:
            logger.info("CNNDistillation forward")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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
            logger.info("GRU init")
            logger.info('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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
            logger.info("GRU forward")
            logger.info('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)