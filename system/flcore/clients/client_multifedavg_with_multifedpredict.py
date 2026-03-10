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
from scipy.stats import chi2_contingency
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest
import copy

def cosine_similarity(p_1, p_2):

    # compute cosine similarity
    try:
        p_1_size = np.array(p_1).shape
        p_2_size = np.array(p_2).shape
        if p_1_size != p_2_size:
            raise Exception(
                f"Input sizes have different shapes: {p_1_size} and {p_2_size}. {p_1} e {p_2}. Please check your input data.")

        p_1 = np.array(p_1).flatten()
        p_2 = np.array(p_2).flatten()

        return np.dot(p_1, p_2) / (norm(p_1) * norm(p_2))
    except Exception as e:
        print("cosine_similairty error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def extract_labels(loader, label_key="label"):
    labels = []
    for batch in loader:
        if isinstance(batch, dict):
            if label_key not in batch:
                raise KeyError(f"Chave '{label_key}' não encontrada no batch. "
                               f"Chaves disponíveis: {list(batch.keys())}")
            y = batch[label_key]
        else:
            raise ValueError(f"Formato inesperado do batch: {type(batch)}")

        if not isinstance(y, torch.Tensor):
            raise TypeError(f"Esperado Tensor como rótulo, mas veio {type(y)}")

        labels.append(y)
    return torch.cat(labels).cpu().numpy()

import torch
import numpy as np
from collections import Counter
from scipy.stats import chisquare, ks_2samp


def extract_from_loader(loader, key="image"):

    X = []
    y = []

    for batch in loader:

        x = batch[key].detach().cpu().numpy()
        label = batch["label"].detach().cpu().numpy()

        X.append(x.reshape(x.shape[0], -1))
        y.append(label)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    return X, y


def detect_label_shift(y1, y2):

    c1 = Counter(y1)
    c2 = Counter(y2)

    classes = sorted(set(y1) | set(y2))

    f1 = np.array([c1.get(c, 0) for c in classes])
    f2 = np.array([c2.get(c, 0) for c in classes])

    stat, p = chisquare(f1, f2)

    return stat, p


def detect_concept_drift(X1, y1, X2, y2):

    classes = np.unique(np.concatenate([y1, y2]))

    drift_detected = False
    results = {}

    for c in classes:

        Xa = X1[y1 == c]
        Xb = X2[y2 == c]

        if len(Xa) == 0 or len(Xb) == 0:
            continue

        stat, p = ks_2samp(Xa.flatten(), Xb.flatten())

        results[c] = (stat, p)

        if p < 0.05:
            drift_detected = True

    return drift_detected, results


def compare_loaders(loader_a, loader_b, key="image"):

    X1, y1 = extract_from_loader(loader_a, key)
    X2, y2 = extract_from_loader(loader_b, key)

    print("\nSamples:", len(y1), len(y2))

    # -------------------------------------------------
    # LABEL SHIFT
    # -------------------------------------------------

    stat, p = detect_label_shift(y1, y2)

    print("\nLabel shift test")
    print("chi2:", stat)
    print("p-value:", p)

    label_shift = p < 0.05

    # -------------------------------------------------
    # CONCEPT DRIFT
    # -------------------------------------------------

    concept_drift, results = detect_concept_drift(X1, y1, X2, y2)

    print("\nConcept drift per class")

    for c, (stat, p) in results.items():

        print(f"class {c}: KS={stat:.4f} p={p:.4f}")

    # -------------------------------------------------
    # RESULTADO FINAL
    # -------------------------------------------------

    if label_shift:
        result = "LABEL_SHIFT"
    elif concept_drift:
        result = "CONCEPT_DRIFT"
    else:
        result = "NO_SHIFT"

    print("\nRESULT:", result)

    return result


class ClientMultiFedAvgWithMultiFedPredict(MultiFedAvgClient):
    def __init__(self, args, id, model, fold_id):
        try:
            super().__init__(args, id,  model, fold_id)
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
            self.train_accuracies = {me: [] for me in range(self.ME)}
            self.data_shift_round = [-1] * self.ME
            self.dataset_input_map = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image",
                                     "Gowalla": "sequence",
                                     "WISDM-W": "sequence", "ImageNet": "image", "ImageNet10": "image",
                                     "wikitext": "sequence", "Foursquare": "sequence"}
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
            p_old = copy.deepcopy(self.p_ME)
            trainloader_A = copy.deepcopy(self.trainloader[me])
            parameters, size, metrics = super().fit(me, t, global_model)
            trainloader_B = self.trainloader[me]
            self.train_losses[me].append(metrics['train_loss'])
            self.train_accuracies[me].append(metrics['train_accuracy'])
            # data_shift_type = compare_loaders(trainloader_A, trainloader_B, key=self.dataset_input_map[self.dataset[me]])
            data_shift_type = "LABEL_SHIFT" if "LABEL_SHIFT" in self.experiment_id else "CONCEPT_DRIFT"
            similarity = min(cosine_similarity(self.p_ME[me], p_old[me]), 1)
            if 1 - similarity < 0:
                print(f"similaridade is {similarity} rodada {t}")

            if t in [20, 30, 40, 50, 60, 70, 80, 90]:
                print(f"cliente #id {self.client_id} rodada {t} modelo {me} accuracies {self.train_accuracies[me]} data shift type {data_shift_type}")

            metrics["non_iid"] = {"fc": self.fc_ME[me], "il": self.il_ME[me], "similarity": similarity, "ps": 1 - similarity, "data_shift_type": data_shift_type}

            return parameters, size, metrics
        except Exception as e:
            print("fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, me, t, global_model, metrics):
        """Evaluate the model on the data this client has."""
        try:
            g = torch.Generator()
            g.manual_seed(t+self.fold_id)
            random.seed(t+self.fold_id)
            np.random.seed(t+self.fold_id)
            torch.manual_seed(t+self.fold_id)
            nt = t - self.lt[me]
            p_ME, fc_ME, il_ME = self.update_local_test_data(t, me)
            fc = metrics["fc"]
            il = metrics["il"]
            similarity_server = metrics["similarity"]
            data_heterogeneity_degree = metrics["heterogeneity_degree"]
            ps = metrics["ps"]
            data_shift_type = metrics["data_shift_type"]
            similarity_local = cosine_similarity(self.p_ME[me], p_ME[me]) # the lower its value the lower the personalization
            a = [0.95, 0.94, 0.81]  # fc > a gw=1
            a = [0, 0, 0]  # fc > a gw=1
            b = [0.59, 0.59, 0.65]  # il < b gw=1
            # b = [1, 1, 1]  # il < b gw=1
            c = [0.31, 0.28, 0.33]  # dh < c gw=1 # 0.43
            c = [0.31, 0.28, 0.33]  # dh < c gw=1 # 0.43
            # c = [1, 1, 1]  # dh < c gw=1 # 0.43
            d = 0.1  # ps > d gw=1

            if self.data_shift_round[me] == -1 and ps > d:
                self.data_shift_round[me] = t

            # if self.lt[me] < self.data_shift_round[me] and data_heterogeneity_degree > c[me] and t - self.data_shift_round[me] < 4:
            #     # chance of using only the global model gw=1
            #     data_shift_adaptation = True
            #     similarity = similarity_server
            # else:
            #     data_shift_adaptation = False
            #     similarity = similarity_local
            #     ps = 0

            # # if t <= 10:
            # #     similarity = 1
            # if similarity_server > 1:
            #     similarity_server = 1
            # elif similarity_server < 0:
            #     similarity_server = 0

            if fc > a[me] and il < b[me] and data_heterogeneity_degree < c[me] and ps > d and nt > 0:
                print(f"usou incorretamente. cliente {self.client_id} rodada {t} modelo {me} valores: {fc}, {il}, {data_heterogeneity_degree} {ps} {nt}")


            # if similarity_server < similarity_local - 0.05 and ps > d and data_heterogeneity_degree < c[me]:
            #     # Delayed labeling scenrio. Local similarity is not reliable
            #     similarity = similarity_server
            # else:
            #     similarity = similarity_local
            print(f"model {me} valor t {t} nt {nt} tamanho {len(global_model)} heterogeneity degree cliente {data_heterogeneity_degree}")
            if self.lt[me] < self.data_shift_round[me] and data_heterogeneity_degree < c[me]:
                similarity = 1
                t_hat = 1
                print("entrou parou")
                local_model_outdated = True
                # exit()
            else:
                t_hat = t
                similarity = 1
                local_model_outdated = False
            combined_model, gw, lw = fedpredict_client_torch(local_model=self.model[me], global_model=global_model,
                                                     t=t, T=self.T, nt=nt,
                                                     s=round(float(similarity), 2),
                                                     lt=self.lt[me],
                                                     data_shift_round=self.data_shift_round[me],
                                                     # fc={'global': fc, 'reference': a[me]},
                                                     # il={'global': il, 'reference': b[me]},
                                                     dh={'global': data_heterogeneity_degree, 'reference': c[me]},
                                                     ps={'global': ps, 'reference': d},
                                                     data_shift_type=data_shift_type,
                                                     device=self.device,
                                                     global_model_original_shape=self.model_shape_mefl[me],
                                                    return_gw_lw=True)

            # if (me == 0 and t < 35 and t >=30 and (t-30) < nt) or (me == 1 and t < 55 and t >= 50 and (t - 50) < nt) or (me == 2 and t < 75 and t >= 70 and (t - 70) < nt):
            #     s = 1  # keeps the standard degree of personalization and does not apply weighted predictions (used for data shift and delayed labeling)
            #     set_weights(self.global_model[me], global_model)
            #     combined_model = self.global_model[me]

            if (gw == 1 and t > 10 and data_heterogeneity_degree < c[me] and ps > d):
                similarity = 1 # keeps the standard degree of personalization and does not apply weighted predictions (used for data shift and delayed labeling)
                set_weights(self.global_model[me], global_model)
                combined_model = self.global_model[me]


            print(f"rodada {t} recebido fc{fc} il{il} homogeneity degree {data_heterogeneity_degree} ps {ps} nt {nt}")

            loss, metrics = test_fedpredict(combined_model, self.valloader[me], self.device, self.client_id, t,
                                            self.args.dataset[me], self.n_classes[me], similarity, p_ME[me],
                                            self.concept_drift_window_test[me])

            # loss, metrics = test(combined_model, self.valloader[me], self.device, self.client_id, t,
            #                      self.args.dataset[me], self.n_classes[me], self.concept_drift_window_test[me])

            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            metrics["Alpha"] = self.alpha_test[me]
            metrics["gw"] = float(gw)
            metrics["lw"] = float(lw)
            tuple_me = (loss, len(self.valloader[me].dataset), metrics)
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    # def evaluate(self, me, t, global_model, metrics):
    #     """Evaluate the model on the data this client has."""
    #     try:
    #         g = torch.Generator()
    #         g.manual_seed(t+self.fold_id)
    #         random.seed(t+self.fold_id)
    #         np.random.seed(t+self.fold_id)
    #         torch.manual_seed(t+self.fold_id)
    #         tuple_me = {}
    #         nt = t - self.lt[me]
    #         self.update_local_test_data(t, me)
    #         combined_model, gw, lw = fedpredict_client_torch(local_model=self.model[me], global_model=global_model,
    #                                                  t=t, T=self.T, nt=nt, device=self.device,
    #                                                  global_model_original_shape=self.model_shape_mefl[me], return_gw_lw=True)
    #         loss, metrics = test(combined_model, self.valloader[me], self.device, self.client_id, t,
    #                              self.args.dataset[me], self.n_classes[me], self.concept_drift_window_test[me])
    #         metrics["Model size"] = self.models_size[me]
    #         metrics["Dataset size"] = len(self.valloader[me].dataset)
    #         metrics["me"] = me
    #         metrics["Alpha"] = self.alpha_test[me]
    #         # metrics["gw"] = float(gw)
    #         # metrics["lw"] = float(lw)
    #         metrics["gw"] = gw
    #         metrics["lw"] = lw
    #         tuple_me = (loss, len(self.valloader[me].dataset), metrics)
    #         return loss, len(self.valloader[me].dataset), tuple_me
    #     except Exception as e:
    #         print("evaluate error")
    #         print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))