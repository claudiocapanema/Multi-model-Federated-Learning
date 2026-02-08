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
    def __init__(self, input_shape=1, out_channel=32, mid_dim=256, num_classes=10):
        try:
            self.mid_dim = mid_dim
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
                          out_channel,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel,
                          out_channel * 2,
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
            print("CNN")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc(out)
            return out
        except Exception as e:
            print("""CNN forward {}""".format(self.mid_dim))
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class NextCategoryLSTMWithTime(nn.Module):
    def __init__(
        self,
        num_categories,
        cat_emb_dim=3,
        hour_emb_dim=3,
        day_emb_dim=2,
        hidden_dim=8,
        num_layers=1,
        dropout=0.5
    ):
        super().__init__()

        self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)
        self.hour_emb = nn.Embedding(24, hour_emb_dim)
        self.day_emb = nn.Embedding(7, day_emb_dim)

        input_dim = cat_emb_dim + hour_emb_dim + day_emb_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, num_categories)

    def forward(self, cat_seq, hour_seq, day_seq):
        cat_e = self.cat_emb(cat_seq)
        hour_e = self.hour_emb(hour_seq)
        day_e = self.day_emb(day_seq)

        x = torch.cat([cat_e, hour_e, day_e], dim=-1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

import torch.nn as nn
import torch.nn.functional as F

class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=10):
        try:
            super(TinyImageNetCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 32x32

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 16x16

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 8x8
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )

        except Exception as e:
            print("CNN tiny init error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):

        try:
            x = self.features(x)
            x = self.classifier(x)
            return x
        except Exception as e:
            print("CNN tiny forward error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


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

            print("CNN_3 init")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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
            print("CNN_3 forward")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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
            print("CNN_3_proto")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.fc(proto)
            return out, proto
        except Exception as e:
            print("CNN_3_proto")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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
            print("CNN student")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            proto = self.conv1(x)
            out = self.out(proto)
            return out, proto
        except Exception as e:
            print("CNN student forward")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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
            print("CNNDistillation")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def forward(self, x):
        try:
            out_student, proto_student = self.student(x)
            out_teacher, proto_teacher = self.teacher(x)
            return out_student, proto_student, out_teacher, proto_teacher
        except Exception as e:
            print("CNNDistillation forward")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class LSTM(torch.nn.Module):
    def __init__(self, input_shape, device, num_layers=1, hidden_size=1, sequence_length=28, num_classes=10):
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

            # self.embedding_category = nn.Embedding(num_embeddings=7, embedding_dim=3)
            # self.embedding_hour = nn.Embedding(num_embeddings=48, embedding_dim=3)
            # self.embedding_distance = nn.Embedding(num_embeddings=51, embedding_dim=2)
            # self.embedding_duration = nn.Embedding(num_embeddings=49, embedding_dim=2)

            self.lstm = nn.LSTM(2, self.hidden_size, self.num_layers, batch_first=False)
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

            category, sub_category, sub_sub_category, hour, distance, duration = torch.split(x, [1, 1, 1, 1, 1, 1], dim=-1)

            category = category[:, -self.time_length:, :]
            hour = hour[:, -self.time_length:, :]
            distance = distance[:, -self.time_length:, :]
            duration = duration[:, -self.time_length:, :]
            category = category / 7
            sub_category = sub_category / 124
            hour = hour / 48
            # category = x[0]
            # hour = x[1]
            # distance = x[2]
            # duration = x[3]
            #print("dim en: ", category.shape, hour.shape, distance.shape, duration.shape)
            # print("valores: ", hour)
            # category_embbeded = self.embedding_category(category)
            # hour_embedded = self.embedding_hour(hour)
            distance = torch.clamp(distance, max=60)
            distance = distance / 60
            duration = torch.clamp(duration, max=60)
            duration = duration / 60
            # distance_embedded = self.embedding_distance(distance)
            # duration_embedded = self.embedding_duration(duration)

            # Concatenando os embeddings com os dados reais
            #print("dim: ", category_embbeded.shape, hour_embedded.shape, distance_embedded.shape, duration_embedded.shape)
            #print("dev: ", category_embbeded.device, hour_embedded.device, distance_embedded.device, duration_embedded.device)
            combined_embedded = torch.cat((category, hour), dim=-1)
            # print("comb: ", combined_embedded.shape)
            # if combined_embedded.shape[0] > 1:
            # combined_embedded = combined_embedded.squeeze()
            # if combined_embedded.dim() == 2:
            #     combined_embedded = combined_embedded.unsqueeze(0)
            # print("dimensoes: ", combined_embedded.shape, distance.shape, duration.shape)
            # combined_embedded = torch.cat((combined_embedded, distance, duration), dim=-1)
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

class LSTMNextWord(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        try:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)
        except Exception as e:
            print("LSTMNextWord forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            emb = self.embedding(x.int())
            out, _ = self.lstm(emb)
            last_out = out[:, -1, :]
            logits = self.fc(last_out)
            return logits
        except Exception as e:
            print("LSTMNextWord forward")
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