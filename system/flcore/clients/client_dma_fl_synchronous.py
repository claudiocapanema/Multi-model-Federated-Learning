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

class DMAFLSynchronousClient(MultiFedAvgClient):
    def __init__(self, args, id, model, fold_id):
        try:
            super().__init__(args, id,  model, fold_id)
            self.last_train_loss = [10000] * self.ME
            self.drift_score = [0] * self.ME
        except Exception as e:
            print("__init__ error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


    def fit(self, me, t, global_model):
        """Train the model with data of this client."""
        try:
            self.lt[me] = t
            p_old = self.p_ME
            parameters, size, metrics = super().fit(me, t, global_model)
            prev = self.last_train_loss[me]
            curr = metrics['train_loss']
            if prev == 10000 or curr is None or prev == 0:
                self.drift_score[me] = 0.0
            else:
                self.drift_score[me] = max(0.0, (curr - prev) / prev)

            self.last_train_loss[me] = metrics['train_loss']

            return parameters, size, metrics
        except Exception as e:
            print("fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
