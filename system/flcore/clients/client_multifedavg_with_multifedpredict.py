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
import pickle


class ClientMultiFedAvgWithMultiFedPredict(MultiFedAvgClient):
    def __init__(self, args, id, model):
        try:
            super().__init__(args, id,  model)
            self.global_model = copy.deepcopy(self.model)
            print("quntidade de modelos: ", len(model), type(model))
            self.model_shape_mefl = []
            for me in range(self.ME):
                self.model_shape_mefl.append([param.shape for name, param in model[me].named_parameters()])
            self.T = args.number_of_rounds
        except Exception as e:
            print("__init__ error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, me, t, global_model):
        """Train the model with data of this client."""
        try:
            self.lt[me] = t
            return super().fit(me, t, global_model)
        except Exception as e:
            print("fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, me, t, global_model):
        """Evaluate the model on the data this client has."""
        try:
            g = torch.Generator()
            g.manual_seed(t)
            random.seed(t)
            np.random.seed(t)
            torch.manual_seed(t)
            tuple_me = {}
            nt = t - self.lt[me]
            # if nt > 0:
            #     set_weights(self.global_model[me], global_model)
            # global_model = pickle.loads(global_model)
            self.update_local_test_data(t, me)
            combined_model = fedpredict_client_torch(local_model=self.model[me], global_model=global_model,
                                                     t=t, T=self.T, nt=nt, device=self.device, global_model_original_shape=self.model_shape_mefl[me])
            loss, metrics = test(combined_model, self.valloader[me], self.device, self.client_id, t,
                                 self.args.dataset[me], self.n_classes[me], self.concept_drift_window[me])
            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            metrics["Alpha"] = self.alpha[me]
            tuple_me = (loss, len(self.valloader[me].dataset), metrics)
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
