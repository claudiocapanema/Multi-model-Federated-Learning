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
from scipy.stats import ks_2samp

def label_distribution_from_loader(loader, n_classes):
    """Return the local empirical P(Y) for one temporal data window."""
    counts = np.zeros(int(n_classes), dtype=np.float64)
    total = 0
    if loader is None:
        return counts
    for batch in loader:
        if not isinstance(batch, dict) or "label" not in batch:
            continue
        y = batch["label"]
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        y = np.asarray(y).reshape(-1).astype(int)
        valid = y[(y >= 0) & (y < int(n_classes))]
        if valid.size:
            counts += np.bincount(valid, minlength=int(n_classes))[:int(n_classes)]
            total += int(valid.size)
    if total == 0:
        return counts
    return counts / float(total)

def _make_sample_loader(loader, fraction=0.20, random_seed=42):
    """Create a deterministic random subset loader containing ``fraction``
    of the samples from ``loader``.

    Only the subset is retained for temporal data-shift detection. The
    original training loader is never modified.
    """
    if loader is None or not hasattr(loader, "dataset"):
        return None

    dataset = loader.dataset
    n = len(dataset)
    if n == 0:
        return None

    sample_size = max(1, int(round(float(n) * float(fraction))))
    sample_size = min(sample_size, n)

    rng = np.random.RandomState(int(random_seed))
    indices = rng.choice(n, size=sample_size, replace=False).tolist()

    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=getattr(loader, "batch_size", None) or 1,
        shuffle=False,
        num_workers=getattr(loader, "num_workers", 0),
        collate_fn=getattr(loader, "collate_fn", None),
        drop_last=False,
        pin_memory=getattr(loader, "pin_memory", False),
    )

class ClientMultiFedAvgWithMultiFedPredict(MultiFedAvgClient):
    def __init__(self, args, id, model, fold_id):
        try:
            super().__init__(
                args,
                id,
                model,
                fold_id
            )

            self.global_model = copy.deepcopy(
                self.model
            )

            print(
                "quntidade de modelos: ",
                len(model),
                type(model)
            )

            self.model_shape_mefl = []

            for me in range(self.ME):
                self.model_shape_mefl.append(
                    [
                        param.shape
                        for name, param
                        in model[me].named_parameters()
                    ]
                )

            self.T = args.number_of_rounds

            self.reset_round = [0] * self.ME

            self.ps_reset = 1

            self.combined_model = [None] * self.ME

            self.train_test_fraction  = 0.4

            self.train_losses = {
                me: [] for me in range(self.ME)
            }

            self.train_accuracies = {
                me: [] for me in range(self.ME)
            }

            self.data_shift_round = [
                                        -1
                                    ] * self.ME

            self.dataset_input_map = {
                "CIFAR10": "img",
                "MNIST": "image",
                "EMNIST": "image",
                "GTSRB": "image",
                "Gowalla": "sequence",
                "WISDM-W": "sequence",
                "ImageNet": "image",
                "ImageNet10": "image",
                "wikitext": "sequence",
                "Foursquare": "sequence"
            }

            # ============================================================
            # Generic data-shift state
            # ============================================================
            # Only a 20% sample of the previous training window is retained
            # for performance-based shift detection.  The full training
            # loader is never copied.
            self.data_shift_reference_trainloader = [None] * self.ME
            self.data_shift_reference_window = [0] * self.ME
            self.data_shift_reference_label_distribution = [None] * self.ME


        except Exception as e:
            print("__init__ error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

    def fit(
            self,
            me,
            t,
            global_model
    ):
        """Train the model after the performance-based shift check."""
        try:
            g = torch.Generator()
            g.manual_seed(t + self.fold_id)
            random.seed(t + self.fold_id)
            np.random.seed(t + self.fold_id)
            torch.manual_seed(t + self.fold_id)

            # ------------------------------------------------------------
            # Keep the original MultiFedAvg ordering: load the global
            # model, then update the local training window.
            # ------------------------------------------------------------
            set_weights(self.model[me], global_model)

            # Save the previous local class-distribution vector before the
            # temporal training window is updated. PS measures the change
            # between consecutive windows.
            p_old = None
            if self.p_ME[me] is not None:
                p_old = np.asarray(copy.deepcopy(self.p_ME[me]), dtype=float).flatten()

            if t > 1:
                self.update_local_train_data(t, me)

            p_current = None
            if self.p_ME[me] is not None:
                p_current = np.asarray(copy.deepcopy(self.p_ME[me]), dtype=float).flatten()

            if (
                    t > 1
                    and p_old is not None
                    and p_current is not None
                    and p_old.shape == p_current.shape
            ):
                similarity = float(np.clip(cosine_similarity(p_current, p_old), 0.0, 1.0))
                ps = float(np.clip(1.0 - similarity, 0.0, 1.0))
            else:
                similarity = 1.0
                ps = 0.0

            current_loader = self.trainloader[me]

            # ------------------------------------------------------------
            # CURRENT LOCAL DH
            # ------------------------------------------------------------
            # update_local_train_data() has already prepared the current
            # training window, so fc_ME/il_ME represent the CURRENT local
            # data heterogeneity used for this detection.
            current_dh = float(
                np.clip(
                    ((1.0 - float(self.fc_ME[me])) + float(self.il_ME[me])) / 2.0,
                    0.0,
                    1.0
                )
            )

            # ------------------------------------------------------------
            # LABEL-SHIFT EVIDENCE
            # ------------------------------------------------------------
            # LS is computed locally from the current training window and
            # the previous training-window class distribution saved during
            # evaluate().  Only the scalar LS score is returned to the
            # server; class distributions are never transmitted.
            #
            # LS = 0.5 * sum_c |p_current(c) - p_previous(c)|
            #
            # The first training window has no previous window, therefore
            # LS=0.0 for the first round.
            ls = 0.0
            if (
                    t > 1
                    and self.data_shift_reference_label_distribution[me] is not None
                    and current_loader is not None
            ):
                p_old_window = self.data_shift_reference_label_distribution[me]
                p_current_window = label_distribution_from_loader(
                    current_loader, self.n_classes[me]
                )
                ls = float(np.clip(
                    0.5 * np.sum(
                        np.abs(p_current_window - p_old_window)
                    ),
                    0.0,
                    1.0
                ))



            # Kept only for backward-compatible logging. The server
            # detector no longer uses this combined score.
            data_shift_score = float(np.clip(ls, 0.0, 1.0))

            # ------------------------------------------------------------
            # NOW start the original local-training flow.
            # ------------------------------------------------------------
            self.lt[me] = t
            self.optimizer[me] = self._get_optimizer(
                dataset_name=self.args.dataset[me],
                me=me
            )

            print(
                f"[TRAIN DEBUG] client={self.client_id} model={me} "
                f"dataset={self.args.dataset[me]} n_classes={self.n_classes[me]}"
            )

            results = train(
                self.model[me],
                self.trainloader[me],
                self.valloader[me],
                self.optimizer[me],
                self.local_epochs,
                self.lr,
                self.device,
                self.client_id,
                t,
                self.args.dataset[me],
                self.n_classes[me],
                self.concept_drift_window_train[me]
            )

            results["me"] = me
            results["client_id"] = self.client_id
            results["Model size"] = self.models_size[me]
            results["alpha"] = self.alpha_train[me]
            self.loss_ME[me] = results["train_loss"]

            self.train_losses[me].append(results["train_loss"])
            self.train_accuracies[me].append(results["train_accuracy"])

            metrics = results
            metrics["non_iid"] = {
                "fc": self.fc_ME[me],
                "il": self.il_ME[me],
                "similarity": similarity,
                "ps": ps,
                "ls": ls,
                "data_shift_score": data_shift_score
            }

            print(
                f"[CLIENT SHIFT EVIDENCE] round={t} client={self.client_id} model={me} "
                f"LS={ls:.6f} train_accuracy={results['train_accuracy']:.6f}"
            )

            return get_weights(self.model[me]), len(self.trainloader[me].dataset), metrics

        except Exception as e:
            print("fit error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))
            return None

    def evaluate(
            self,
            me,
            t,
            global_model,
            metrics
    ):
        """Evaluate the model on the data this client has."""
        try:
            g = torch.Generator()

            g.manual_seed(
                t + self.fold_id
            )

            random.seed(
                t + self.fold_id
            )

            np.random.seed(
                t + self.fold_id
            )

            torch.manual_seed(
                t + self.fold_id
            )

            nt = (
                    t - self.lt[me]
            )

            # ---------------------------------------------------------
            # Validation/test data is updated only for evaluation.
            # It is NOT used to calculate LS or CD.
            # ---------------------------------------------------------
            p_ME, fc_ME, il_ME = (
                self.update_local_test_data(
                    t,
                    me
                )
            )

            fc = metrics["fc"]
            il = metrics["il"]

            similarity_server = (
                metrics["similarity"]
            )

            data_heterogeneity_degree = (
                metrics["heterogeneity_degree"]
            )

            ls = float(
                metrics.get(
                    "ls",
                    0.0
                )
            )


            # Kept only for backward compatibility.
            ps = float(
                metrics.get(
                    "ps",
                    0.0
                )
            )

            # General detector state produced by the server.
            # No shift type is inferred or required by the client.
            data_shift_detected = bool(
                metrics.get(
                    "data_shift",
                    False
                )
            )


            similarity_local = cosine_similarity(
                self.p_ME[me],
                p_ME[me]
            )

            a = [
                0.0,
                0.0,
                0.0
            ]

            b = [
                0.59,
                0.59,
                0.65
            ]

            tau_dh = [
                0.31,
                0.32,
                0.39
            ]

            # Do not independently create a new shift event in
            # evaluate().  Event detection is performed once by the
            # server from aggregated client evidence.  Here we only keep
            # the local data_shift_round for FedPredict compatibility.
            shift_detected = data_shift_detected

            # ---------------------------------------------------------
            # Data-shift round is triggered by either detector.
            #
            # PS is not used.
            # ---------------------------------------------------------
            if (
                    self.data_shift_round[me] == -1
                    and shift_detected
            ):
                self.data_shift_round[me] = t

            # ---------------------------------------------------------
            # Determine whether the local model is outdated.
            #
            # DH remains a heterogeneity signal.
            # Shift detection is based on LS/CD.
            # ---------------------------------------------------------
            if (
                    self.lt[me]
                    < self.data_shift_round[me]
                    and data_heterogeneity_degree
                    < tau_dh[me]
            ):
                similarity = 1
                t_hat = 1
                local_model_outdated = True

                print(
                    "local model considered outdated"
                )

            else:
                t_hat = t
                similarity = 1
                local_model_outdated = False

            # ---------------------------------------------------------
            # FedPredict remains backward-compatible with its current
            # API.
            #
            # data_shift_round is now triggered by LS OR CD.
            # ps is retained only because the current
            # fedpredict_client_torch signature uses it.
            # ---------------------------------------------------------
            combined_model, gw, lw = (
                fedpredict_client_torch(
                    local_model=self.model[me],
                    global_model=global_model,
                    t=t,
                    T=self.T,
                    nt=nt,
                    s=round(
                        float(similarity),
                        2
                    ),
                    lt=self.lt[me],
                    data_shift_round=(
                        self.data_shift_round[me]
                    ),
                    dh={
                        "global":
                            data_heterogeneity_degree,
                        "reference":
                            tau_dh[me]
                    },
                    data_shift_type=(
                        "DATA_SHIFT"
                        if data_shift_detected
                        else "NO_SHIFT"
                    ),
                    device=self.device,
                    global_model_original_shape=(
                        self.model_shape_mefl[me]
                    ),
                    return_gw_lw=True
                )
            )

            # ---------------------------------------------------------
            # Existing global-model fallback.
            #
            # It now applies to either detected LS or CD.
            # ---------------------------------------------------------

            print(
                f"rodada {t} recebido "
                f"fc={fc} "
                f"il={il} "
                f"dh={data_heterogeneity_degree} "
                f"ls={ls} "
                f"ps={ps} "
                f"nt={nt} "
                f"data_shift={'DATA_SHIFT' if data_shift_detected else 'NO_SHIFT'}"
            )

            # =========================================================
            # Save the exact combined model used by this evaluate().
            # The next fit() uses this model to test the previous and
            # current training windows before local training starts.
            # =========================================================
            self.combined_model[me] = copy.deepcopy(combined_model).cpu()

            # =========================================================
            # Combined-model training accuracy
            # =========================================================
            # Only clients that actually trained model ``me`` in this
            # round contribute this value to the server-side history.
            # The combined model is deliberately obtained exactly as in
            # the original implementation above and is evaluated here,
            # inside evaluate().
            combined_train_accuracy = None

            # The combined-model training test is only needed for
            # experiments where concept drift / combined shift is
            # explicitly being simulated.  For all other experiment
            # types (e.g. label shift or no shift), do not evaluate the
            # combined model on the local training data.
            experiment_id = str(
                getattr(self.args, "experiment_id", "")
            ).lower()

            combined_model_test_enabled = (
                "concept_drift" in experiment_id
                or "combined_shift" in experiment_id
            )

            if (
                    combined_model_test_enabled
                    and self.lt[me] == t
                    and self.trainloader[me] is not None
                    and len(self.trainloader[me].dataset) > 0
            ):
                _, combined_train_metrics = test(
                    combined_model,
                    self.trainloader[me],
                    self.device,
                    self.client_id,
                    t,
                    self.args.dataset[me],
                    self.n_classes[me],
                )
                combined_train_accuracy = float(
                    np.clip(
                        combined_train_metrics["Accuracy"],
                        0.0,
                        1.0
                    )
                )

            # Keep only 20% of the training dataset for the next generic
            # performance-based data-shift test.  The full previous
            # training dataset is never retained.
            if self.trainloader[me] is not None:
                self.data_shift_reference_trainloader[me] = _make_sample_loader(
                    self.trainloader[me],
                    fraction=self.train_test_fraction,
                    random_seed=(42 + self.client_id + 1000 * me)
                )
                self.data_shift_reference_window[me] = int(t)
                self.data_shift_reference_label_distribution[me] = (
                    label_distribution_from_loader(
                        self.trainloader[me], self.n_classes[me]
                    )
                )

            # =========================================================
            # IMPORTANT:
            #
            # This is evaluation only.
            # No LS/CD is calculated from this validation loader.
            # =========================================================
            loss, test_metrics = test(
                combined_model,
                self.valloader[me],
                self.device,
                self.client_id,
                t,
                self.args.dataset[me],
                self.n_classes[me],
            )

            test_metrics["Model size"] = (
                self.models_size[me]
            )

            test_metrics["Dataset size"] = (
                len(
                    self.valloader[me].dataset
                )
            )

            test_metrics["me"] = me
            test_metrics["Alpha"] = (
                self.alpha_test[me]
            )

            test_metrics["gw"] = float(gw)
            test_metrics["lw"] = float(lw)
            test_metrics["combined_train_accuracy"] = combined_train_accuracy

            tuple_me = (
                loss,
                len(
                    self.valloader[me].dataset
                ),
                test_metrics
            )

            return (
                loss,
                len(
                    self.valloader[me].dataset
                ),
                tuple_me
            )

        except Exception as e:
            print("evaluate error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )