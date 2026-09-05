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
    """
    Detect label shift from the change in P(Y) between two local windows.

    The returned statistic is the chi-square statistic from a 2 x C
    contingency table. The p-value is used as statistical evidence; the
    magnitude used by MultiFedPredict remains the Total Variation distance
    computed from the class proportions.
    """
    y1 = np.asarray(y1).reshape(-1)
    y2 = np.asarray(y2).reshape(-1)

    classes = sorted(set(y1.tolist()) | set(y2.tolist()))
    if len(classes) < 2 or len(y1) == 0 or len(y2) == 0:
        return 0.0, 1.0

    table = np.asarray([
        [np.sum(y1 == c) for c in classes],
        [np.sum(y2 == c) for c in classes],
    ], dtype=np.int64)

    if np.any(table.sum(axis=0) == 0):
        table = table[:, table.sum(axis=0) > 0]

    if table.shape[1] < 2:
        return 0.0, 1.0

    try:
        stat, p, _, _ = chi2_contingency(table, correction=False)
    except ValueError:
        return 0.0, 1.0

    return float(stat), float(p)


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


def _rbf_kernel_matrix(x, y, sigma):
    """Compute an RBF kernel matrix using squared Euclidean distances."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        return np.empty((0, 0), dtype=np.float64)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    dist2 = np.maximum(x2 + y2 - 2.0 * np.dot(x, y.T), 0.0)
    return np.exp(-dist2 / (2.0 * max(float(sigma) ** 2, 1e-12)))


def _median_heuristic_sigma(x, y):
    """Robust RBF bandwidth selected from pooled pairwise distances."""
    from scipy.spatial.distance import pdist
    pooled = np.concatenate([x, y], axis=0)
    if len(pooled) < 2:
        return 1.0
    distances = pdist(pooled, metric="euclidean")
    distances = distances[np.isfinite(distances) & (distances > 1e-12)]
    if len(distances) == 0:
        return 1.0
    sigma = float(np.median(distances))
    return max(sigma, 1e-6)


def _mmd2_unbiased(x, y, sigma):
    """Unbiased squared MMD with an RBF kernel."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2 or x.shape[1] != y.shape[1]:
        return 0.0

    kxx = _rbf_kernel_matrix(x, x, sigma)
    kyy = _rbf_kernel_matrix(y, y, sigma)
    kxy = _rbf_kernel_matrix(x, y, sigma)

    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)

    term_xx = np.sum(kxx) / (nx * (nx - 1))
    term_yy = np.sum(kyy) / (ny * (ny - 1))
    term_xy = 2.0 * np.mean(kxy)
    return float(max(term_xx + term_yy - term_xy, 0.0))


def _mmd_effect_and_pvalue(xa, xb, random_seed=42, n_permutations=100):
    """Generic distribution-shift evidence using RBF-MMD.

    The effect is normalized to [0,1] and the p-value is obtained from a
    permutation null distribution. No labels or shift type are required.
    """
    xa = np.asarray(xa, dtype=np.float32)
    xb = np.asarray(xb, dtype=np.float32)
    if xa.ndim != 2 or xb.ndim != 2 or len(xa) < 10 or len(xb) < 10:
        return 0.0, 1.0
    if xa.shape[1] != xb.shape[1]:
        return 0.0, 1.0

    sigma = _median_heuristic_sigma(xa, xb)
    observed = _mmd2_unbiased(xa, xb, sigma)

    # Convert MMD^2 to a bounded effect-size-like score.  Under an RBF
    # kernel, MMD^2 is naturally bounded by a small multiple of one; this
    # transformation preserves ordering while avoiding dataset-specific
    # raw-distance scales.
    effect = float(np.clip(np.sqrt(max(observed, 0.0)), 0.0, 1.0))

    rng = np.random.RandomState(random_seed)
    pooled = np.concatenate([xa, xb], axis=0)
    n_a = len(xa)
    observed_count = 0
    n_total = len(pooled)

    for _ in range(int(n_permutations)):
        idx = rng.permutation(n_total)
        perm_a = pooled[idx[:n_a]]
        perm_b = pooled[idx[n_a:]]
        stat = _mmd2_unbiased(perm_a, perm_b, sigma)
        if stat >= observed - 1e-12:
            observed_count += 1

    p_value = float((observed_count + 1.0) / (int(n_permutations) + 1.0))
    return effect, p_value


def extract_features_from_loader(
        loader,
        key="image",
        max_samples=512,
        random_seed=42
):
    """Extract an unlabeled sample of X from a temporal local data window."""
    if loader is None:
        return np.empty((0, 0), dtype=np.float32)

    features = []
    total = 0
    for batch in loader:
        if not isinstance(batch, dict) or key not in batch:
            continue
        x = batch[key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.detach().cpu().numpy()
        if x.ndim == 0:
            continue
        x = x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
        features.append(x)
        total += len(x)

    if not features:
        return np.empty((0, 0), dtype=np.float32)

    x = np.concatenate(features, axis=0)
    if len(x) > int(max_samples):
        rng = np.random.RandomState(random_seed)
        idx = rng.choice(len(x), size=int(max_samples), replace=False)
        x = x[idx]
    return x


def _compact_features(x, n_components=32, random_seed=42):
    """Project high-dimensional inputs to a deterministic compact space."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        return x
    d = x.shape[1]
    if d <= int(n_components):
        return x
    rng = np.random.RandomState(random_seed)
    projection = rng.normal(
        0.0, 1.0, size=(d, int(n_components))
    ).astype(np.float32)
    projection /= np.sqrt(float(n_components))
    return x @ projection


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


def _performance_from_loader(model, loader, device, dataset_name, n_classes):
    """Evaluate one model on a loader and return balanced accuracy."""
    if loader is None or len(loader.dataset) == 0:
        return 0.0, np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    key = {
        "CIFAR10": "img",
        "MNIST": "image",
        "EMNIST": "image",
        "GTSRB": "image",
        "Gowalla": "sequence",
        "WISDM-W": "sequence",
        "ImageNet": "image",
        "ImageNet10": "image",
        "wikitext": "sequence",
        "Foursquare": "sequence",
    }[dataset_name]

    model.eval()
    model.to(device)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            x = batch[key].to(device)
            labels = batch["label"].to(device)
            outputs = model(x)
            predictions = torch.argmax(outputs, dim=1)
            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(y_true).astype(np.int64)
    y_pred = np.concatenate(y_pred).astype(np.int64)
    score = float(metrics.balanced_accuracy_score(y_true, y_pred))
    correct = (y_true == y_pred).astype(np.int8)
    return score, correct, y_true, y_pred


def _bootstrap_performance_drop_pvalue(
        old_true,
        old_pred,
        current_true,
        current_pred,
        random_seed=42,
        n_bootstrap=200
):
    """Estimate significance of a balanced-accuracy performance drop."""
    old_true = np.asarray(old_true).reshape(-1)
    old_pred = np.asarray(old_pred).reshape(-1)
    current_true = np.asarray(current_true).reshape(-1)
    current_pred = np.asarray(current_pred).reshape(-1)

    if min(len(old_true), len(current_true)) < 10:
        return 1.0

    rng = np.random.RandomState(int(random_seed))
    drops = np.empty(int(n_bootstrap), dtype=np.float64)

    for i in range(int(n_bootstrap)):
        old_idx = rng.randint(0, len(old_true), size=len(old_true))
        current_idx = rng.randint(0, len(current_true), size=len(current_true))

        old_score = metrics.balanced_accuracy_score(
            old_true[old_idx], old_pred[old_idx]
        )
        current_score = metrics.balanced_accuracy_score(
            current_true[current_idx], current_pred[current_idx]
        )
        drops[i] = old_score - current_score

    # One-sided bootstrap evidence for a positive degradation.
    return float(
        (np.sum(drops <= 0.0) + 1.0) / (len(drops) + 1.0)
    )

def detect_generic_data_shift(
        model,
        old_loader,
        current_loader,
        device,
        dataset_name,
        n_classes,
        min_performance_drop=0.20,
        alpha=0.05,
        n_bootstrap=200,
        random_seed=42
):
    """
    Detect generic data shift from relative model-performance degradation.

    The same combined model is evaluated on a sample of the previous
    training window and a sample of the current training window.

    A generic shift is reported only when:

        1. The relative performance reduction is at least
           ``min_performance_drop``; and

        2. The performance degradation is statistically significant
           according to the bootstrap p-value.

    The relative performance reduction is defined as:

        relative_drop =
            max(old_score - current_score, 0) / old_score

    Therefore, ``min_performance_drop=0.20`` means that the current
    balanced accuracy must be at least 20% lower than the previous
    balanced accuracy.

    Example:

        old_score = 0.90
        current_score = 0.72

        absolute_drop = 0.18
        relative_drop = 0.18 / 0.90 = 0.20

    In this case, the practical degradation threshold is satisfied.

    The bootstrap is used only to determine whether the observed
    performance degradation is statistically significant. It does
    not determine the magnitude of the practical degradation.
    """

    try:
        # ------------------------------------------------------------
        # Evaluate the same combined model on the previous window
        # and on the current window.
        # ------------------------------------------------------------
        old_score, _, old_true, old_pred = _performance_from_loader(
            model,
            old_loader,
            device,
            dataset_name,
            n_classes
        )

        current_score, _, current_true, current_pred = (
            _performance_from_loader(
                model,
                current_loader,
                device,
                dataset_name,
                n_classes
            )
        )

        # ------------------------------------------------------------
        # Insufficient data for a reliable comparison.
        # ------------------------------------------------------------
        if len(old_true) < 10 or len(current_true) < 10:
            return (
                0.0,
                1.0,
                old_score,
                current_score
            )

        # ------------------------------------------------------------
        # Absolute performance degradation.
        #
        # This is retained for diagnostics and interpretation, but
        # it is NOT the threshold used to determine the shift.
        # ------------------------------------------------------------
        absolute_drop = float(
            max(old_score - current_score, 0.0)
        )

        # ------------------------------------------------------------
        # Relative performance degradation.
        #
        # Example:
        #
        # old_score = 0.80
        # current_score = 0.64
        #
        # relative_drop = (0.80 - 0.64) / 0.80
        #                = 0.20
        #
        # Therefore, this represents a 20% reduction.
        # ------------------------------------------------------------
        if old_score > 1e-12:
            relative_drop = float(
                absolute_drop / old_score
            )
        else:
            relative_drop = 0.0

        # Numerical safety.
        relative_drop = float(
            np.clip(relative_drop, 0.0, 1.0)
        )

        # ------------------------------------------------------------
        # Bootstrap statistical significance test.
        #
        # The bootstrap estimates whether the observed degradation
        # can reasonably be explained by sampling variability.
        #
        # IMPORTANT:
        # The bootstrap p-value is independent of the practical
        # relative-drop threshold above.
        # ------------------------------------------------------------
        p_value = _bootstrap_performance_drop_pvalue(
            old_true,
            old_pred,
            current_true,
            current_pred,
            random_seed=random_seed,
            n_bootstrap=n_bootstrap
        )

        # ------------------------------------------------------------
        # Generic data shift is detected only when BOTH conditions
        # are satisfied:
        #
        #   1. Relative degradation >= threshold
        #   2. Bootstrap p-value < alpha
        #
        # Thus, with min_performance_drop=0.20:
        #
        #       current_score <= 0.80 * old_score
        #
        # AND:
        #
        #       p_value < 0.05
        # ------------------------------------------------------------
        detected = (
            relative_drop >= float(min_performance_drop)
            and p_value < float(alpha)
        )

        # ------------------------------------------------------------
        # Return the RELATIVE degradation as the GDS score.
        #
        # Previously this returned the absolute difference
        # old_score - current_score. Now it returns:
        #
        #       (old_score - current_score) / old_score
        #
        # whenever the shift is detected.
        # ------------------------------------------------------------
        return (
            relative_drop if detected else 0.0,
            p_value,
            old_score,
            current_score
        )

    except Exception as e:
        print("detect_generic_data_shift error")
        print(
            "Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno,
                type(e).__name__,
                e
            )
        )

        return (
            0.0,
            1.0,
            0.0,
            0.0
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

            self.train_test_fraction  = 0.5

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

            self.gds_score = [0.0] * self.ME
            self.gds_pvalue = [1.0] * self.ME
            self.gds_old_performance = [0.0] * self.ME
            self.gds_current_performance = [0.0] * self.ME

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

            if t > 1:
                self.update_local_train_data(t, me)

            current_loader = self.trainloader[me]

            # ------------------------------------------------------------
            # GENERIC DATA SHIFT -- BEFORE LOCAL TRAINING
            # ------------------------------------------------------------
            gds = 0.0
            gds_pvalue = 1.0
            old_performance = 0.0
            current_performance = 0.0

            previous_loader = self.data_shift_reference_trainloader[me]
            combined_model = self.combined_model[me]

            if (
                    t > 15
                    and previous_loader is not None
                    and current_loader is not None
                    and combined_model is not None
            ):
                # Test only 20% of the current training data. The previous
                # 20% was already sampled and stored at the last evaluate().
                current_eval_loader = _make_sample_loader(
                    current_loader,
                    fraction=self.train_test_fraction,
                    random_seed=(42 + self.client_id + 1000 * me + t)
                )

                gds, gds_pvalue, old_performance, current_performance = (
                    detect_generic_data_shift(
                        model=combined_model,
                        old_loader=previous_loader,
                        current_loader=current_eval_loader,
                        device=self.device,
                        dataset_name=self.args.dataset[me],
                        n_classes=self.n_classes[me],
                        min_performance_drop=0.20,
                        alpha=0.05,
                        n_bootstrap=200,
                        random_seed=(42 + self.client_id + 1000 * me + t)
                    )
                )

                gds = float(np.clip(gds, 0.0, 1.0))
                gds_pvalue = float(np.clip(gds_pvalue, 0.0, 1.0))
                old_performance = float(np.clip(old_performance, 0.0, 1.0))
                current_performance = float(np.clip(current_performance, 0.0, 1.0))

            self.gds_score[me] = gds
            self.gds_pvalue[me] = gds_pvalue
            self.gds_old_performance[me] = old_performance
            self.gds_current_performance[me] = current_performance

            # ------------------------------------------------------------
            # Existing local label-shift signal.
            # A compact class distribution is stored at evaluate(), so the
            # full previous training dataset is not required here.
            # ------------------------------------------------------------
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
                    0.5 * np.sum(np.abs(p_current_window - p_old_window)),
                    0.0, 1.0
                ))
            else:
                ls = 0.0

            # ------------------------------------------------------------
            # Existing PS/similarity behavior is preserved.
            # ------------------------------------------------------------
            p_old = np.asarray(copy.deepcopy(self.p_ME[me]), dtype=float).flatten()
            p_current = np.asarray(copy.deepcopy(self.p_ME[me]), dtype=float).flatten()
            similarity = min(cosine_similarity(p_current, p_old), 1.0)
            ps = 1.0 - similarity

            data_shift_score = float(np.clip(max(ls, gds), 0.0, 1.0))

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
                "gds": gds,
                "gds_pvalue": gds_pvalue,
                "gds_old_performance": old_performance,
                "gds_current_performance": current_performance,
                "data_shift_score": data_shift_score
            }

            print(
                f"[CLIENT GDS] round={t} client={self.client_id} model={me} "
                f"old_bal_acc={old_performance:.6f} "
                f"current_bal_acc={current_performance:.6f} "
                f"drop={gds:.6f} p={gds_pvalue:.6g} "
                f"sample_fraction=0.20"
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

            gds = float(
                metrics.get(
                    "gds",
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

            # Generic-data-shift p-value is produced during fit() and may be
            # useful for diagnostics only.  It MUST be read from the
            # server metrics here; evaluate() has no local gds_pvalue
            # variable.  This fixes the previous NameError.
            gds_pvalue = float(
                metrics.get(
                    "gds_pvalue",
                    1.0
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

            # ---------------------------------------------------------
            # These thresholds are kept consistent with the server.
            # They are used only for local adaptation state.
            # ---------------------------------------------------------
            tau_ls = 0.10

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
                f"gds={gds} "
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

            # Keep only 20% of the training dataset for the next generic
            # performance-based data-shift test.  The full previous
            # training dataset is never retained.
            if self.trainloader[me] is not None:
                self.data_shift_reference_trainloader[me] = _make_sample_loader(
                    self.trainloader[me],
                    fraction=self.train_test_fraction,
                    random_seed=(42 + self.client_id + 1000 * me + t)
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