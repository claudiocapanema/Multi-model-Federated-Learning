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
import random
import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging
from flcore.servers.server_multifedavg import MultiFedAvg
from flcore.servers.server_multifedavg_meh import MultiFedAvgMDH
from flcore.servers.server_multifedavgrr import MultiFedAvgRR
from flcore.servers.server_fedfairmmfl import FedFairMMFL
from flcore.servers.server_multifedavg_with_multifedpredict import MultiFedAvgWithMultiFedPredict
from flcore.servers.server_dma_fl_synchronous import DMAFLSynchronous
from flcore.servers.server_adaptive_fedavg import AdaptiveFedAvg
from flcore.servers.server_multifedavg_with_fedpredict_dynamic import MultiFedAvgWithFedPredictDynamic
from flcore.servers.server_multifedavg_with_fedpredict import MultiFedAvgWithFedPredict
from flcore.servers.server_multifedavg_with_multifedpredict_v0 import MultiFedAvgWithMultiFedPredictv0

from flcore.clients.utils.models import CNN, CNN_3, CNNDistillation, GRU, LSTM, TinyImageNetCNN, LSTMNextWord, NextPlaceModel

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# hyper-params for Text tasks
# vocab_size = 98635   #98635 for AG_News and 399198 for Sogou_News
max_len=200
emb_dim=32

def load_model(model_name, dataset, strategy, device):
    try:
        num_classes = {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'Tiny-ImageNet': 200,
         'ImageNet100': 15, 'ImageNet': 15, "ImageNet10": 10, "ImageNet_v2": 15, "Gowalla": 7, "wikitext": 3743, "Foursquare": 107}[dataset]
        out_channel = 32
        if model_name == 'CNN':
            if dataset in ['MNIST']:
                input_shape = 1
                mid_dim = 256*4
                logger.info("""leu mnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['EMNIST']:
                input_shape = 1
                mid_dim = 256*4
                logger.info("""leu emnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['GTSRB']:
                input_shape = 3
                mid_dim = 36*4
                logger.info("""leu gtsrb com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ["ImageNet"]:
                input_shape=3
                mid_dim=1600
            elif dataset in ["ImageNet10"]:
                input_shape=3
                # mid_dim=21632
                # out_channel = 64
                input_shape = 3
                mid_dim = 1600
                # return TinyImageNetCNN()
            elif dataset == "CIFAR10":
                input_shape = 3
                mid_dim = 400*4
                logger.info("""leu cifar com {} {} {}""".format(input_shape, mid_dim, num_classes))
            return CNN(input_shape=input_shape, out_channel=out_channel, num_classes=num_classes, mid_dim=mid_dim)
        elif model_name == 'CNN_3':
            if dataset in ['MNIST']:
                input_shape = 1
                mid_dim = 4
                logger.info("""leu mnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['EMNIST']:
                input_shape = 1
                mid_dim = 4
                logger.info("""leu emnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['GTSRB']:
                input_shape = 3
                mid_dim = 16
                logger.info("""leu gtsrb com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "ImageNet":
                input_shape = 3
                mid_dim = 16
                logger.info("""leu imagenet com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "ImageNet10":
                input_shape = 3
                mid_dim = 16
                logger.info("""leu imagenet10 com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "CIFAR10":
                input_shape = 3
                mid_dim = 16
                logger.info("""leu cifar com {} {} {}""".format(input_shape, mid_dim, num_classes))

            if "FedKD" in strategy:
                return CNNDistillation(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes, dataset=dataset)
            else:
                return CNN_3(input_shape=input_shape, num_classes=num_classes, mid_dim=mid_dim)

        elif model_name == "gru":
            if dataset in ["WISDM-W", "WISDM-P"]:
                return GRU(6, num_layers=1, hidden_size=2, sequence_length=200, num_classes=num_classes)

        elif model_name == "lstm":
            if dataset == "Foursquare":
                return NextPlaceModel(num_classes, 64, 128)

        elif model_name == "lstm":
            if dataset in ["Gowalla"]:
                return LSTM(6, device=device, num_layers=1, hidden_size=1, sequence_length=4, num_classes=num_classes)
            elif dataset in ["wikitext"]:
                return LSTMNextWord(vocab_size=num_classes, embed_dim=10, hidden_dim=10)

        raise ValueError("""Model not found for model {} and dataset {}""".format(model_name, dataset))

    except Exception as e:
        logger.error("""load_model error""")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def run(args):

    time_list = []
    for fold_id in range(1, args.k_fold + 1):
        print(f"=============== Fold ID {fold_id} of {args.k_fold} ============")
        print("Creating server and clients ...")
        start = time.time()
        models = []
        ME = len(args.model)

        for m in range(ME):
            # Generate args.model
            model_name = args.model[m]
            dataset = args.dataset[m]
            model = load_model(model_name, dataset, args.strategy, args.device)

            print(model)
            models.append(model)

        # select algorithm
        if args.strategy == "MultiFedAvg":
            server = MultiFedAvg
        elif args.strategy == "MultiFedAvg-MDH":
            server = MultiFedAvgMDH
        elif args.strategy == "DMA-FL":
            server = DMAFLSynchronous
        elif args.strategy == "AdaptiveFedAvg":
            server = AdaptiveFedAvg
        elif args.strategy == "MultiFedAvgRR":
            server = MultiFedAvgRR
        elif args.strategy == "FedFairMMFL":
            server = FedFairMMFL
        elif args.strategy == "MultiFedAvg+FP":
            version = None
            server = MultiFedAvgWithFedPredict
        elif args.strategy == "MultiFedAvg+FPD":
            version = None
            server = MultiFedAvgWithFedPredictDynamic
        elif args.strategy == "MultiFedAvg+MFP":
            server = MultiFedAvgWithMultiFedPredictv0
        elif args.strategy == "MultiFedAvg+MFP_v2":
            version = "full"
            server = MultiFedAvgWithMultiFedPredict
        elif args.strategy == "MultiFedAvg+MFP_v2_dh":
            version = "dh"
            server = MultiFedAvgWithMultiFedPredict
        elif args.strategy == "MultiFedAvg+MFP_v2_iti":
            version = "iti"
            server = MultiFedAvgWithMultiFedPredict

        # elif args.strategy == "MultiFedEfficiency":
        #     server = MultiFedEfficiency
        #
        # elif args.strategy == "MultiFedAvg+FP":
        #     server = MultiFedAvgWithFedPredict
        #
        # elif args.strategy == "FedFairMMFL":
        #     server = FedFairMMFL
        # elif args.strategy == "MultiFedAvgRR":
        #     server = MultiFedAvgRR

        else:
            print(args.strategy)
            raise NotImplementedError

        if args.strategy not in ["MultiFedAvg+MFP_v2", "MultiFedAvg+MFP_v2_dh", "MultiFedAvg+MFP_v2_iti", "MultiFedAvg+FP", "MultiFedAvg+FPD"]:
            server = server(args, models, fold_id)
        else:
            server = server(args, models, version, fold_id)
        server.train()

        time_list.append(time.time()-start)

        print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")


        # Global average
        # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times, args=args)

        print("All done!")

        # reporter.report()


if __name__ == "__main__":

    total_start = time.time()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8
    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    parser = argparse.ArgumentParser(description="Generated Docker Compose")
    parser.add_argument(
        "--total_clients", type=int, default=20, help="Total clients to spawn (default: 2)"
    )
    parser.add_argument(
        "--k_fold", type=int, default=1, help="K fold"
    )
    parser.add_argument(
        "--number_of_rounds", type=int, default=5, help="Number of FL rounds (default: 5)"
    )
    parser.add_argument(
        "--data_percentage",
        type=float,
        default=0.8,
        help="Portion of client data to use (default: 0.6)",
    )
    parser.add_argument(
        "--random", action="store_true", help="Randomize client configurations"
    )

    parser.add_argument(
        "--strategy", type=str, default='FedAvg', help="Strategy to use (default: FedAvg)"
    )
    parser.add_argument(
        "--alpha", action="append", help="Dirichlet alpha"
    )
    parser.add_argument(
        "--experiment_id", type=str, default="", help=""
    )
    parser.add_argument(
        "--round_new_clients", type=float, default=0.1, help=""
    )
    parser.add_argument(
        "--fraction_new_clients", type=float, default=0.1, help=""
    )
    parser.add_argument(
        "--local_epochs", type=float, default=1, help=""
    )
    parser.add_argument(
        "--dataset", action="append"
    )
    parser.add_argument(
        "--model", action="append"
    )
    parser.add_argument(
        "--cd", type=str, default="false"
    )
    parser.add_argument(
        "--fraction_fit", type=float, default=0.3
    )
    parser.add_argument(
        "--client_id", type=int, default=1
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01
    )
    parser.add_argument(
        "--server_address", type=str, default="server:8080"
    )

    parser.add_argument(
        "--device", type=str, default="cuda"
    )
    parser.add_argument(
        "--tw", type=int, default=15, help="TW window of rounds used in MultiFedEfficiency"
    )
    parser.add_argument(
        "--reduction", type=int, default=3,
        help="Reduction in the number of training clients used in MultiFedEfficiency"
    )
    parser.add_argument(
        "--df", type=float, default=0, help="Free budget redistribution factor used in MultiFedEfficiency"
    )

    args = parser.parse_args()

    if args.strategy == "MultiFedEfficiency":
        log_name = args.strategy + "_tw_" + str(args.tw) + "_df_" + str(args.df)

    else:
        log_name = args.strategy

    result_path = """results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/log_{}.txt""".format(args.experiment_id,
                                                                                                                       args.total_clients,
                                                                                                            [float(i) for i in args.alpha],
                                                                                                            args.dataset,
                                                                                                            args.model,
                                                                                                            args.fraction_fit,
                                                                                                            args.number_of_rounds,
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
