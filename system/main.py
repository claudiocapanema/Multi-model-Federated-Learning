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

#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
from torchsummary import summary

from flcore.servers.server_multifedavg_separated import MultiFedAvg_Separated
from flcore.servers.server_multifedavg import MultiFedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.server_multifedper import MultiFedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverdistill import FedDistill
from flcore.servers.serverala import FedALA
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
from flcore.servers.servercp import FedCP
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverntd import FedNTD
from flcore.servers.servergh import FedGH
from flcore.servers.serveravgDBE import FedAvgDBE
from flcore.servers.serveravg_with_fedpredict import MultiFedAvgWithFedPredict
from flcore.servers.server_fedfairmmfl import FedFairMMFL
from flcore.servers.server_multifedspeed import MultiFedSpeed
from flcore.servers.server_multifedspeed_dynamic import MultiFedSpeedDynamic
from flcore.servers.server_multifedspeed_relative import MultiFedSpeedRelative
from flcore.servers.server_multifedfifo import MultiFedCP
from flcore.servers.server_multifedpriority import MultiFedPriority
from flcore.servers.server_multifedpriority_ec import MultiFedPriority_ec
from flcore.servers.server_multifedpriority_deterministic import MultiFedPriority_deterministic
from flcore.servers.server_multifedfifo_ec import MultiFedCP_ec
from flcore.servers.server_multifedfifo_deterministic import MultiFedCP_deterministic
from flcore.servers.server_multifedrebalance import MultiFedRebalance
from flcore.servers.server_multifedspeed_deterministic_selection import MultiFedSpeed_Deterministic_Selection
from flcore.servers.server_multifedavg_with_fedpredict_train import MultiFedAvgWithFedPredictTrain
from flcore.servers.server_multifedavg_global_model_eval import MultiFedAvgGlobalModelEval
from flcore.servers.serveravg_rr import MultiFedAvgRR
from flcore.servers.server_multifedyogi import MultiFedYogi
from flcore.servers.server_multifedyogi_with_fedpredict import MultiFedYogiWithFedPredict
from flcore.servers.server_multifedefficiency import MultiFedEfficiency
from flcore.servers.server_multifedavg_global_model_eval_with_fedpredict import MultiFedAvgGlobalModelEvalWithFedPredict
from flcore.servers.server_multifedkd import MultiFedKD
from flcore.servers.server_multifedyogi_global_model_eval import MultiFedYogiGlobalModelEval
from flcore.servers.server_multifedyogi_global_model_eval_with_fedpredict import MultiFedYogiGlobalModelEvalWithFedPredict
from flcore.servers.server_multifedkd_with_fedpredict import MultiFedKDWithFedPredict

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635   #98635 for AG_News and 399198 for Sogou_News
max_len=200
emb_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    models_str = args.model
    dts = args.dataset
    num_classes = []
    for dt in dts:
        num_classes.append({'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'Tiny-ImageNet': 200,
                            'ImageNet100': 15, 'ImageNet': 15, "ImageNet_v2": 15, "Gowalla": 7}[dt])

    args.num_classes = num_classes

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        models = []

        for m in range(len(models_str)):
            # Generate args.model
            model_str = models_str[m]
            dt = args.dataset[m]
            num_classes_m = num_classes[m]
            print(model_str, models_str)
            if "mlr" in model_str: # convex
                if "MNIST" == dt:
                    model = Mclr_Logistic(1*28*28, num_classes=num_classes_m).to(args.device)
                elif "CIFAR10" == dt:
                    model = Mclr_Logistic(3*32*32, num_classes=num_classes_m).to(args.device)
                else:
                    model = Mclr_Logistic(60, num_classes=num_classes_m).to(args.device)

            elif "cnn_a" in model_str: # non-convex
                if "EMNIST" == dt or "MNIST" == dt:
                    model = FedAvgCNN(dataset=dt, in_features=1, num_classes=num_classes_m, dim=1024).to(args.device)
                    if "MultiFedKD" in args.algorithm:
                        model = FedAvgCNNKD(dataset=dt, in_features=1, num_classes=num_classes_m, dim=1024).to(
                            args.device)
                elif "CIFAR10" == dt:
                    model = FedAvgCNN(dataset=dt, in_features=3, num_classes=num_classes_m, dim=1600).to(args.device)
                    print("Sumario: \n", summary(model, (3, 32, 32)))
                    if "MultiFedKD" in args.algorithm:
                        model = FedAvgCNNKD(dataset=dt, in_features=3, num_classes=num_classes_m, dim=1600).to(
                            args.device)
                        print("Sumario: \n", summary(model, (3, 32, 32)))
                elif "GTSRB" == dt:
                    model = FedAvgCNN(dataset=dt, in_features=3, num_classes=num_classes_m, dim=1600).to(args.device)
                    if "MultiFedKD" in args.algorithm:
                        model = FedAvgCNNKD(dataset=dt, in_features=3, num_classes=num_classes_m, dim=1600).to(
                            args.device)
                elif "Omniglot" == dt:
                    model = FedAvgCNN(dataset=dt, in_features=1, num_classes=num_classes_m, dim=33856).to(args.device)
                    # model = CIFARNet(num_classes=num_classes_m).to(args.device)
                elif "Digit5" == dt:
                    model = Digit5CNN().to(args.device)
                elif dt in ["ImageNet100", "ImageNet", "ImageNet_v2"]:
                    # model = TinyImageNetCNN().to(args.device)
                    model = FedAvgCNN(dataset=dt, in_features=3, num_classes=num_classes_m, dim=1600).to(args.device)
                else:
                    model = FedAvgCNN(dataset=dt, in_features=3, num_classes=num_classes_m, dim=10816).to(args.device)

            elif "cnn_b" in model_str:
                if "GTSRB" == dt:
                    model = CNN_2(3, mid_dim=64, num_classes=num_classes_m).to(args.device)

            elif "cnn_c" in model_str: # non-convex
                if "EMNIST" == dt or "MNIST" == dt:
                    model = CNN_3(dataset=dt, in_features=1, num_classes=num_classes_m, dim=4).to(args.device)
                    if args.algorithm == "MultiFedKD":
                        model = FedAvgCNNKD(dataset=dt, in_features=1, num_classes=num_classes_m, dim=1024).to(
                            args.device)
                elif "CIFAR10" == dt:
                    model = CNN_3(dataset=dt, in_features=3, num_classes=num_classes_m, dim=16).to(args.device)
                    print("Sumario: \n", summary(model, (3, 32, 32)))
                    if args.algorithm == "MultiFedKD":
                        model = FedAvgCNNKD(dataset=dt, in_features=3, num_classes=num_classes_m, dim=16).to(
                            args.device)
                        print("Sumario: \n", summary(model, (3, 32, 32)))
                elif "GTSRB" == dt:
                    model = CNN_3(dataset=dt, in_features=3, num_classes=num_classes_m, dim=16).to(args.device)
                    if args.algorithm == "MultiFedKD":
                        model = FedAvgCNNKD(dataset=dt, in_features=3, num_classes=num_classes_m, dim=16).to(
                            args.device)

            elif model_str == "gru":
                if dt in ["WISDM-W", "WISDM-P"]:
                    model = GRU(6, num_layers=1, hidden_size=2, sequence_length=100, num_classes=num_classes_m).to(args.device)
            elif model_str == "lstm":
                if dt in ["Gowalla"]:
                    model = LSTM(4, num_layers=1, hidden_size=1, sequence_length=6, num_classes=num_classes_m).to(
                        args.device)

            elif model_str == "lstm":
                if dt in ["WISDM-W", "WISDM-P"]:
                    model = LSTM_NET(6, hidden_dim=6, num_classes=num_classes_m).to(args.device)



            elif "dnn" in model_str: # non-convex
                if dt in ["MNIST", "EMNIST"]:
                    model = DNN(1*28*28, 100, num_classes=num_classes_m).to(args.device)
                elif "CIFAR10" == dt:
                    model = DNN(3*32*32, 100, num_classes=num_classes_m).to(args.device)
                else:
                    model = DNN(60, 20, num_classes=num_classes_m).to(args.device)

            elif "resnet" in model_str:
                model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes_m).to(args.device)

                # model = torchvision.models.resnet18(pretrained=True).to(args.device)
                # feature_dim = list(model.fc.parameters())[0].shape[1]
                # model.fc = nn.Linear(feature_dim, num_classes_m).to(args.device)

                # model = resnet18(num_classes=num_classes_m, has_bn=True, bn_block_num=4).to(args.device)

            elif "resnet10" in model_str:
                model = resnet10(num_classes=num_classes_m).to(args.device)

            elif "resnet34" in model_str:
                model = torchvision.models.resnet34(pretrained=False, num_classes=num_classes_m).to(args.device)

            elif "alexnet" in model_str:
                model = alexnet(pretrained=False, num_classes=num_classes_m).to(args.device)

                # model = alexnet(pretrained=True).to(args.device)
                # feature_dim = list(model.fc.parameters())[0].shape[1]
                # model.fc = nn.Linear(feature_dim, num_classes_m).to(args.device)

            elif "googlenet" in model_str:
                model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                          num_classes=num_classes_m).to(args.device)

                # model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
                # feature_dim = list(model.fc.parameters())[0].shape[1]
                # model.fc = nn.Linear(feature_dim, num_classes_m).to(args.device)

            elif "mobilenet_v2" in model_str:
                model = mobilenet_v2(pretrained=False, num_classes=num_classes_m).to(args.device)

                # model = mobilenet_v2(pretrained=True).to(args.device)
                # feature_dim = list(model.fc.parameters())[0].shape[1]
                # model.fc = nn.Linear(feature_dim, num_classes_m).to(args.device)

            elif "lstm" in model_str:
                model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=num_classes_m).to(args.device)

            elif "bilstm" in model_str:
                model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                                       output_size=num_classes_m, num_layers=1,
                                                       embedding_dropout=0, lstm_dropout=0, attention_dropout=0,
                                                       embedding_length=emb_dim).to(args.device)

            elif "fastText" in model_str:
                model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=num_classes_m).to(args.device)

            elif "TextCNN" in model_str:
                model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                                     num_classes=num_classes_m).to(args.device)

            elif "Transformer" in model_str:
                model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, nlayers=2,
                                              num_classes=num_classes_m, max_len=max_len).to(args.device)

            elif "AmazonMLP" in model_str:
                model = AmazonMLP().to(args.device)

            elif "harcnn" in model_str:
                if args.dataset == 'HAR':
                    model = HARCNN(9, dim_hidden=1664, num_classes=num_classes_m, conv_kernel_size=(1, 9),
                                        pool_kernel_size=(1, 2)).to(args.device)
                elif args.dataset == 'PAMAP2':
                    model = HARCNN(9, dim_hidden=3712, num_classes=num_classes_m, conv_kernel_size=(1, 9),
                                        pool_kernel_size=(1, 2)).to(args.device)

            else:
                print(models_str)
                raise NotImplementedError

            print(model)

            # select algorithm
            if args.algorithm == "MultiFedAvg":
                server = MultiFedAvg
            elif args.algorithm == "MultiFedAvgGlobalModelEval":
                server = MultiFedAvgGlobalModelEval

            elif args.algorithm == "Local":
                server = Local

            elif args.algorithm == "FedMTL":
                server = FedMTL

            elif args.algorithm == "PerAvg":
                server = PerAvg

            elif args.algorithm == "pFedMe":
                server = pFedMe

            elif args.algorithm == "FedProx":
                server = FedProx

            elif args.algorithm == "FedFomo":
                server = FedFomo

            elif args.algorithm == "FedAMP":
                server = FedAMP

            elif args.algorithm == "APFL":
                server = APFL

            elif args.algorithm == "MultiFedPer":
                server = MultiFedPer

            elif args.algorithm == "Ditto":
                server = Ditto

            elif args.algorithm == "FedRep":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedRep

            elif args.algorithm == "FedPHP":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedPHP

            elif args.algorithm == "FedBN":
                server = FedBN

            elif args.algorithm == "FedROD":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedROD

            elif args.algorithm == "FedProto":
                server = FedProto

            elif args.algorithm == "FedDyn":
                server = FedDyn(args, i)

            elif args.algorithm == "MOON":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = MOON

            elif args.algorithm == "FedBABU":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedBABU

            elif args.algorithm == "APPLE":
                server = APPLE

            elif args.algorithm == "FedGen":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedGen

            elif args.algorithm == "SCAFFOLD":
                server = SCAFFOLD

            elif args.algorithm == "FedDistill":
                server = FedDistill

            elif args.algorithm == "FedALA":
                server = FedALA

            elif args.algorithm == "FedPAC":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedPAC

            elif args.algorithm == "LG-FedAvg":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = LG_FedAvg

            elif args.algorithm == "FedGC":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedGC

            elif args.algorithm == "FML":
                server = FML

            elif args.algorithm == "FedKD":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedKD

            elif args.algorithm == "FedPCL":
                model.fc = nn.Identity()
                server = FedPCL

            elif args.algorithm == "FedCP":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedCP

            elif args.algorithm == "GPFL":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = GPFL

            elif args.algorithm == "FedNTD":
                server = FedNTD

            elif args.algorithm == "FedGH":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedGH

            elif args.algorithm == "FedAvgDBE":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = FedAvgDBE

            elif args.algorithm == "MultiFedEfficiency":
                server = MultiFedEfficiency

            elif args.algorithm == "MultiFedKD":
                server = MultiFedKD

            elif args.algorithm == "MultiFedKDWithFedPredict":
                server = MultiFedKDWithFedPredict

            elif args.algorithm == "MultiFedAvgGlobalModelEvalWithFedPredict":
                server = MultiFedAvgGlobalModelEvalWithFedPredict

            elif args.algorithm == "MultiFedYogiGlobalModelEval":
                server = MultiFedYogiGlobalModelEval

            elif args.algorithm == "MultiFedYogiGlobalModelEvalWithFedPredict":
                server = MultiFedYogiGlobalModelEvalWithFedPredict

            elif args.algorithm == "MultiFedPriority":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = MultiFedPriority

            elif args.algorithm == "MultiFedPriority_ec":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = MultiFedPriority_ec

            elif args.algorithm == "MultiFedPriority_deterministic":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = MultiFedPriority_deterministic

            elif args.algorithm == "MultiFedYogi":
                server = MultiFedYogi

            elif args.algorithm == "MultiFedYogiWithFedPredict":
                server = MultiFedYogiWithFedPredict

            elif args.algorithm == "MultiFedFIFO":
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = MultiFedCP_ec

            elif args.algorithm == "MultiFedFIFO_ec":
                server = MultiFedCP_ec

            elif args.algorithm == "MultiFedFIFO_deterministic":
                server = MultiFedCP_deterministic

            elif args.algorithm == "MultiFedBalance":
                server = MultiFedRebalance

            elif args.algorithm == "MultiFedAvg_Separated":
                server = MultiFedAvg_Separated

            elif args.algorithm == "MultiFedAvg_with_FedPredict_Train":
                server = MultiFedAvgWithFedPredictTrain

            elif args.algorithm == "MultiFedAvgWithFedPredict":
                server = MultiFedAvgWithFedPredict

            elif args.algorithm == "FedFairMMFL":
                server = FedFairMMFL
            elif args.algorithm == "MultiFedAvgRR":
                server = MultiFedAvgRR
            elif "MultiFedSpeedRelative" in args.algorithm:
                server = MultiFedSpeedRelative
            elif "MultiFedSpeedDynamic" in args.algorithm:
                server = MultiFedSpeedDynamic
            elif "MultiFedSpeed_D" in args.algorithm:
                server = MultiFedSpeed_Deterministic_Selection

            elif "MultiFedSpeed" in args.algorithm:
                head = copy.deepcopy(model.fc)
                model.fc = nn.Identity()
                model = BaseHeadSplit(model, head)
                server = MultiFedSpeed

            else:
                print(args.algorithm)
                raise NotImplementedError

            models.append(model)

        args.model = models
        args.models_names = models_str
        server = server(args, i)
        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times, args=args)

    print("All done!")

    # reporter.report()


if __name__ == "__main__":

    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    # parser.add_argument('-dts', "--datasets", action="append", type=str, default=['MNIST', 'CIFAR10'])
    parser.add_argument('-mds', "--models",  default=['dnn', 'cnn'])
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", action="append")
    parser.add_argument('-nb', "--num_classes", default=[10, 10])
    parser.add_argument('-m', "--model",  action="append")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01, # 0.01 e 0.05 aumenta a loss quando reduz a quantidade de clintes usando SGD
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-frw', "--fairness_weight", type=float, default=2)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-re', "--reduction", type=int, default=3,
                        help="Number of clients to reduce - Used by MultiFedSpeed")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-fnc', "--fraction_new_clients", type=float, default=0)
    parser.add_argument('-rnc', "--round_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=100000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # MultiFedEfficiency
    parser.add_argument('-tw', "--tw", type=int, default=16,
                        help="Rounds window")
    parser.add_argument('-df', "--df", type=float, default=0)
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", action="append")
    parser.add_argument('-cd', "--concept_drift", default="")
    parser.add_argument('-cd_ae-', "--alpha_end", action="append")
    parser.add_argument('-cd_r', "--rounds_concept_drift", action="append")
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedAvgDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    # print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    # print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:

    if args.algorithm == "MultiFedEfficiency":
        log_name = args.algorithm + "_tw_" + str(args.tw) + "_df_" + str(args.df)

    else:
        log_name = args.algorithm

    if bool(args.concept_drift):
        result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/log_{}.txt""".format(bool(args.concept_drift),
                                                                                                                                                                                                    args.fraction_new_clients,
                                                                                                                                                                                                    args.round_new_clients,
                                                                                                                                                                                                    args.num_clients,
                                                                                                                                                                                                    args.alpha,
                                                                                                                                                                                                    args.alpha_end[
                                                                                                                                                                                                    0],
                                                                                                                                                                                                    args.alpha_end[
                                                                                                                                                                                                    1],
                                                                                                                                                                                                    args.dataset,
                                                                                                                                                                                                    args.rounds_concept_drift[
                                                                                                                                                                                                    0],
                                                                                                                                                                                                    args.rounds_concept_drift[
                                                                                                                                                                                                    1],
                                                                                                                                                                                                    args.model,
                                                                                                                                                                                                    args.join_ratio,
                                                                                                                                                                                                    args.global_rounds,
                                                                                                                                                                                                    args.local_epochs,
                                                                                                                                                                                                    log_name)
    elif len(args.alpha) == 1:
        # run singe model
        result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/log_{}.txt""".format(
            bool(args.concept_drift),
            args.fraction_new_clients,
            args.round_new_clients,
            args.num_clients,
            [args.alpha[0]],
            args.alpha[0],
            args.alpha[0],
            [args.dataset[0]],
            0,
            0,
            [args.model[0]],
            args.join_ratio,
            args.global_rounds,
            args.local_epochs,
            log_name)
    else:
        result_path = """../results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/log_{}.txt""".format(
            bool(args.concept_drift),
            args.fraction_new_clients,
            args.round_new_clients,
            args.num_clients,
            args.alpha,
            args.alpha,
            args.dataset,
            0,
            0,
            args.model,
            args.join_ratio,
            args.global_rounds,
            args.local_epochs,
            log_name)
    print("log: ", result_path)
    import sys

    # Abra um arquivo para gravação
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        # Redirecione a saída padrão para o arquivo
        original = sys.stdout
        sys.stdout = f
        run(args)
        sys.stdout = original

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
