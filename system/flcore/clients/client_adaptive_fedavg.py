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

import random
import copy
import torch
import numpy as np
import time
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import sys
from flcore.clients.client_multifedavg import MultiFedAvgClient
from fedpredict import fedpredict_client_torch
from .utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
from numpy.linalg import norm
import pickle

def cosine_similarity(p_1, p_2):

    # compute cosine similarity
    try:
        p_1_size = np.array(p_1).shape
        p_2_size = np.array(p_2).shape
        if p_1_size != p_2_size:
            raise Exception(f"Input sizes have different shapes: {p_1_size} and {p_2_size}. Please check your input data.")

        return np.dot(p_1, p_2) / (norm(p_1) * norm(p_2))
    except Exception as e:
        print("cosine_similairty error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class AdaptiveFedAvgClient(MultiFedAvgClient):
    def __init__(self, args, id, model, fold_id):
        try:
            super().__init__(args, id,  model, fold_id)
            self.M = [0] * self.ME
            self.V = [0] * self.ME
            self.R = [0] * self.ME
            self.rounds_seen = [0] * self.ME  # used for bias correction power
            # store last bias-corrected values for denominator (for ratio)
            self.M_hat_prev = [0] * self.ME
            self.V_hat_prev = [0] * self.ME
            self.local_lr = [self.lr_dict[name] for name in self.args.dataset]
        except Exception as e:
            print("__init__ error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def get_optimizer(self, dataset_name, me):
        try:
            return {
                    'EMNIST': torch.optim.SGD(self.model[me].parameters(), lr=self.local_lr[me], momentum=0.9),
                    'MNIST': torch.optim.SGD(self.model[me].parameters(), lr=self.local_lr[me], momentum=0.9),
                    'CIFAR10': torch.optim.SGD(self.model[me].parameters(), lr=self.local_lr[me], momentum=0.9),
                    'GTSRB': torch.optim.SGD(self.model[me].parameters(), lr=self.local_lr[me], momentum=0.9),
                    'WISDM-W': torch.optim.RMSprop(self.model[me].parameters(), lr=self.local_lr[me], momentum=0.9),
                    'WISDM-P': torch.optim.RMSprop(self.model[me].parameters(), lr=self.local_lr[me], momentum=0.9),
                    'ImageNet100': torch.optim.SGD(self.model[me].parameters(), lr=self.local_lr[me], momentum=0.9),
                    'ImageNet': torch.optim.SGD(self.model[me].parameters(), lr=self.local_lr[me]),
                    'ImageNet10': torch.optim.SGD(self.model[me].parameters(), lr=self.local_lr[me]),
                    "ImageNet_v2": torch.optim.Adam(self.model[me].parameters(), lr=self.local_lr[me]),
                    "Gowalla": torch.optim.RMSprop(self.model[me].parameters(), lr=self.local_lr[me]),
                    "wikitext": torch.optim.RMSprop(self.model[me].parameters(), lr=self.local_lr[me])}[dataset_name]
        except Exception as e:
            print("_get_optimizer error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
