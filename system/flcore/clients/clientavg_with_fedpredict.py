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

import copy
import torch
import numpy as np
import time
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientavg import clientAVG
from utils.privacy import *
from fedpredict import fedpredict_client_torch


class ClientAvgWithFedPredict(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.last_training_round = 0

    def train(self, m, t):
        self.last_training_round = t
        return super().train(m)


    def test_metrics(self, m, global_model, t, T):
        nt = t - self.last_training_round
        combinel_model = fedpredict_client_torch(local_model=self.model[m], global_model=global_model,
                                  t=t, T=10, nt=nt)
        self.model[m] = combinel_model

        return super().test_metrics(m=m)