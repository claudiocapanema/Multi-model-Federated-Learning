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
from .utils.models_utils import load_model, get_weights, load_data, set_weights, test, test_fedpredict, train
from numpy.linalg import norm
import pickle
from scipy.stats import ks_2samp

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
            self.reset_round = [0] * self.ME
            self.ps_reset = 1
            self.combined_model = [None] * self.ME
            self.train_losses = {me: [] for me in range(self.ME)}
        except Exception as e:
            print("__init__ error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def detect_drift_ks(self, losses, window=5, alpha=0.05):

        try:
            """
            Detecta drift usando teste de Kolmogorov-Smirnov.
            Apenas aumentos são considerados drift.
        
            Args:
                losses (list[float]): histórico de valores de loss.
                window (int): tamanho da janela de comparação.
                alpha (float): nível de significância.
        
            Returns:
                bool: True se houve drift, False caso contrário.
            """
            if len(losses) < window + 1:
                return False  # histórico insuficiente

            history = np.array(losses[-(window + 1):-1])
            last = np.array([losses[-1]] * len(history))  # simula distribuição do último valor

            stat, p_value = ks_2samp(history, last)

            # KS detecta diferença significativa (p < alpha)
            # mas só consideramos se for aumento
            return (p_value < alpha) and (losses[-1] > history.mean())

        except Exception as e:
            print("detect_drift_ks client error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, me, t, global_model):
        """Train the model with data of this client."""
        try:
            self.lt[me] = t
            p_old = self.p_ME
            parameters, size, metrics = super().fit(me, t, global_model)
            self.train_losses[me].append(metrics['train_loss'])

            similarity = min(cosine_similarity(self.p_ME[me], p_old[me]), 1)
            if 1 - similarity < 0:
                print(f"similaridade is {similarity} rodada {t}")

            metrics["non_iid"] = {"fc": self.fc_ME[me], "il": self.il_ME[me], "similarity": similarity, "ps": 1 - similarity}

            return parameters, size, metrics
        except Exception as e:
            print("fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, me, t, global_model, metrics):
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
            p_ME, fc_ME, il_ME = self.update_local_test_data(t, me)
            fc = metrics["fc"]
            il = metrics["il"]
            similarity = metrics["similarity"]
            homogeneity_degree = metrics["homogeneity_degree"]
            ps = metrics["ps"]
            s = cosine_similarity(self.p_ME[me], p_ME[me])
            a = 0.67
            # b = [0.54, 0.56]
            b = [0.76, 0.76, 0.76]
            c = [0.47, 0.47, 0.47]
            d = 0.55
            # if t <= 10:
            #     similarity = 1
            if similarity > 1:
                similarity = 1
            elif similarity < 0:
                similarity = 0

            print(f"valor t {t} nt {nt} tamanho {len(global_model)}")
            combined_model = fedpredict_client_torch(local_model=self.model[me], global_model=global_model,
                                                     t=t, T=self.T, nt=nt, s=round(float(similarity), 2),
                                                     fc={'global': fc, 'reference': a},
                                                     il={'global': il, 'reference': b[me]},
                                                     dh={'global': homogeneity_degree, 'reference': c[me]},
                                                     ps={'global': ps, 'reference': d},
                                                     device=self.device,
                                                     global_model_original_shape=self.model_shape_mefl[me])
            if (fc >= a and il < b[me] and homogeneity_degree > c[me]) or (
                    ps < d and nt > 0 and t > 10 and homogeneity_degree > c[me]):
                s = 1
                set_weights(self.global_model[me], global_model)
                combined_model = self.global_model[me]

            print(f"rodada {t} recebido fc{fc} il{il} homogeneity degree {homogeneity_degree} ps {ps} nt {nt}")

            loss, metrics = test_fedpredict(combined_model, self.valloader[me], self.device, self.client_id, t,
                                            self.args.dataset[me], self.n_classes[me], s, p_ME[me],
                                            self.concept_drift_window[me])

            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            metrics["Alpha"] = self.alpha[me]
            tuple_me = (loss, len(self.valloader[me].dataset), metrics)
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
