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


def _extract_class_conditional_features(
        loader,
        key,
        n_classes,
        label_offset=0,
        max_samples_per_class=256,
        random_seed=42
):
    """
    Extract a bounded and deterministic random sample of training
    examples for each class.

    The concept-drift detector compares:

        P(X | Y)

    between two training windows.

    Important:
    - Only local training data is used.
    - No validation/test data is accessed.
    - Samples are selected independently within each class.
    - Random selection avoids artifacts caused by the ordering of
      examples inside the DataLoader.
    - label_offset is kept for backward compatibility with experiments
      that explicitly transform labels.
    """

    try:
        class_features = {
            c: []
            for c in range(n_classes)
        }

        for batch in loader:

            if not isinstance(batch, dict):
                raise ValueError(
                    f"Unexpected batch type: {type(batch)}"
                )

            if key not in batch:
                raise KeyError(
                    f"Input key '{key}' not found in batch. "
                    f"Available keys: {list(batch.keys())}"
                )

            if "label" not in batch:
                raise KeyError(
                    "Key 'label' not found in batch."
                )

            x = batch[key]
            y = batch["label"]

            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)

            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)

            x = (
                x.detach()
                .cpu()
                .numpy()
            )

            y = (
                y.detach()
                .cpu()
                .numpy()
            )

            # Flatten each sample independently.
            x = x.reshape(
                x.shape[0],
                -1
            ).astype(
                np.float32,
                copy=False
            )

            y = y.astype(
                np.int64,
                copy=False
            )

            # Keep compatibility with the existing
            # concept-drift simulation.
            if label_offset != 0:
                y = (
                    y + int(label_offset)
                ) % int(n_classes)

            for class_id in range(n_classes):

                indices = np.where(
                    y == class_id
                )[0]

                if len(indices) == 0:
                    continue

                current_count = sum(
                    len(part)
                    for part in class_features[class_id]
                )

                remaining = (
                    max_samples_per_class
                    - current_count
                )

                if remaining <= 0:
                    continue

                # Deterministic random sampling inside each batch.
                rng = np.random.RandomState(
                    random_seed
                    + 1009 * class_id
                )

                if len(indices) > remaining:
                    indices = rng.choice(
                        indices,
                        size=remaining,
                        replace=False
                    )

                class_features[class_id].append(
                    x[indices]
                )

            # Stop when all classes have enough samples.
            if all(
                    sum(
                        len(part)
                        for part in class_features[c]
                    )
                    >= max_samples_per_class
                    for c in range(n_classes)
            ):
                break

        # ------------------------------------------------------------
        # Concatenate samples for each class.
        # ------------------------------------------------------------

        for class_id in range(n_classes):

            if len(
                    class_features[class_id]
            ) == 0:

                class_features[class_id] = np.empty(
                    (0, 0),
                    dtype=np.float32
                )

                continue

            class_features[class_id] = np.concatenate(
                class_features[class_id],
                axis=0
            )

            # Final deterministic cap.
            if (
                    len(class_features[class_id])
                    > max_samples_per_class
            ):

                rng = np.random.RandomState(
                    random_seed
                    + 1009 * class_id
                    + 7919
                )

                indices = rng.choice(
                    len(class_features[class_id]),
                    size=max_samples_per_class,
                    replace=False
                )

                class_features[class_id] = (
                    class_features[class_id][indices]
                )

        return class_features

    except Exception as e:

        print(
            "_extract_class_conditional_features error"
        )

        print(
            "Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno,
                type(e).__name__,
                e
            )
        )

        raise


def detect_concept_drift(
        loader_a,
        loader_b,
        n_classes,
        key="image",
        label_offset_a=0,
        label_offset_b=0,
        max_samples_per_class=256,
        n_projections=8,
        min_samples_per_class=20,
        random_seed=42,
        projection_alpha=0.01,
        min_significant_projections=2
):
    """
    Detect Concept Drift by comparing P(X | Y) between two
    local training windows.

    The detector is explicitly conditioned on the class label.
    Therefore, changes in P(Y) alone should not be classified as
    Concept Drift.

    A class is considered to exhibit Concept Drift only when a
    sufficient number of independent random projections provide
    statistically significant evidence of a distributional change.

    This is intentionally more conservative than simply taking
    the maximum KS statistic across projections. The previous
    implementation could classify sampling variability as Concept
    Drift because a single projection with a relatively large KS
    statistic was sufficient.

    Returns
    -------
    float
        Concept-drift score in [0, 1].
    """

    try:

        # ============================================================
        # Extract class-conditional features
        # ============================================================

        features_a = _extract_class_conditional_features(
            loader=loader_a,
            key=key,
            n_classes=n_classes,
            label_offset=label_offset_a,
            max_samples_per_class=max_samples_per_class
        )

        features_b = _extract_class_conditional_features(
            loader=loader_b,
            key=key,
            n_classes=n_classes,
            label_offset=label_offset_b,
            max_samples_per_class=max_samples_per_class
        )

        class_scores = []
        class_weights = []

        # ============================================================
        # Compare P(X | Y) independently for every class
        # ============================================================

        for class_id in range(n_classes):

            xa = features_a[class_id]
            xb = features_b[class_id]

            # --------------------------------------------------------
            # A class must be represented in BOTH windows.
            #
            # If a class disappears because of a change in P(Y),
            # this is label-shift evidence, not concept-drift
            # evidence.
            # --------------------------------------------------------

            if (
                    xa.ndim != 2
                    or xb.ndim != 2
                    or len(xa) < min_samples_per_class
                    or len(xb) < min_samples_per_class
            ):
                continue

            if xa.shape[1] != xb.shape[1]:
                raise ValueError(
                    "Feature dimensions differ between training "
                    f"windows for class {class_id}: "
                    f"{xa.shape[1]} vs {xb.shape[1]}"
                )

            feature_dim = xa.shape[1]

            if feature_dim <= 0:
                continue

            # ========================================================
            # Deterministic random projections
            # ========================================================

            rng = np.random.RandomState(
                random_seed + class_id
            )

            projections = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(
                    n_projections,
                    feature_dim
                )
            ).astype(np.float32)

            projection_norms = np.linalg.norm(
                projections,
                axis=1,
                keepdims=True
            )

            projections = (
                projections
                / np.maximum(
                    projection_norms,
                    1e-12
                )
            )

            projected_a = np.matmul(
                xa,
                projections.T
            )

            projected_b = np.matmul(
                xb,
                projections.T
            )

            # ========================================================
            # KS tests over projections
            # ========================================================

            projection_statistics = []
            significant_statistics = []

            for projection_id in range(
                    n_projections
            ):

                values_a = projected_a[
                    :,
                    projection_id
                ]

                values_b = projected_b[
                    :,
                    projection_id
                ]

                # ----------------------------------------------------
                # Common normalization.
                #
                # KS is invariant to a common monotonic scaling, but
                # this improves numerical stability.
                # ----------------------------------------------------

                combined = np.concatenate(
                    [
                        values_a,
                        values_b
                    ]
                )

                scale = np.std(
                    combined
                )

                if (
                        not np.isfinite(scale)
                        or scale < 1e-12
                ):
                    scale = 1.0

                values_a = (
                    values_a / scale
                )

                values_b = (
                    values_b / scale
                )

                statistic, p_value = ks_2samp(
                    values_a,
                    values_b
                )

                if not np.isfinite(statistic):
                    continue

                statistic = float(
                    np.clip(
                        statistic,
                        0.0,
                        1.0
                    )
                )

                projection_statistics.append(
                    statistic
                )

                # ----------------------------------------------------
                # IMPORTANT:
                #
                # A single significant projection is not enough.
                #
                # This prevents random sampling variability from
                # producing Concept Drift.
                # ----------------------------------------------------

                if (
                        np.isfinite(p_value)
                        and p_value < projection_alpha
                ):
                    significant_statistics.append(
                        statistic
                    )

            # ========================================================
            # No valid projections
            # ========================================================

            if len(projection_statistics) == 0:
                continue

            # ========================================================
            # Require consistent statistical evidence
            # ========================================================

            if (
                    len(significant_statistics)
                    < min_significant_projections
            ):
                class_score = 0.0

            else:
                # ----------------------------------------------------
                # Use the mean of the significant projections rather
                # than the maximum.
                #
                # This is deliberately conservative.
                # ----------------------------------------------------

                class_score = float(
                    np.mean(
                        significant_statistics
                    )
                )

            class_scores.append(
                float(
                    np.clip(
                        class_score,
                        0.0,
                        1.0
                    )
                )
            )

            # Weight classes by the amount of paired evidence.
            class_weights.append(
                float(
                    min(
                        len(xa),
                        len(xb)
                    )
                )
            )

        # ============================================================
        # No class had enough paired observations
        # ============================================================

        if len(class_scores) == 0:
            return 0.0

        class_scores = np.asarray(
            class_scores,
            dtype=float
        )

        class_weights = np.asarray(
            class_weights,
            dtype=float
        )

        # ============================================================
        # Weighted mean class-conditional drift
        # ============================================================

        weight_sum = np.sum(
            class_weights
        )

        if weight_sum <= 0:
            weighted_mean = float(
                np.mean(
                    class_scores
                )
            )

        else:
            weighted_mean = float(
                np.sum(
                    class_scores
                    * class_weights
                )
                / weight_sum
            )

        # ============================================================
        # Maximum class drift
        #
        # A strong drift in only one class should still be visible.
        # ============================================================

        maximum = float(
            np.max(
                class_scores
            )
        )

        # ============================================================
        # Final CD score
        # ============================================================

        cd_score = (
            0.5 * maximum
            + 0.5 * weighted_mean
        )

        return float(
            np.clip(
                cd_score,
                0.0,
                1.0
            )
        )

    except Exception as e:

        print(
            "detect_concept_drift error"
        )

        print(
            "Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno,
                type(e).__name__,
                e
            )
        )

        return 0.0




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
            # NEW:
            # Independent reference used exclusively by the Concept
            # Drift detector.
            #
            # IMPORTANT:
            # recent_trainloader is intentionally NOT modified here.
            # Its current behavior remains unchanged.
            # ============================================================

            self.cd_reference_trainloader = [
                                                None
                                            ] * self.ME

            self.cd_reference_window = [
                                           0
                                       ] * self.ME

            self.cd_score = [
                                0.0
                            ] * self.ME

            # ------------------------------------------------------------
            # Initialize the CD reference using the initial local
            # training data.
            #
            # This is a separate copy from recent_trainloader.
            # ------------------------------------------------------------

            for me in range(self.ME):
                if self.trainloader[me] is not None:
                    self.cd_reference_trainloader[me] = (
                        copy.deepcopy(
                            self.trainloader[me]
                        )
                    )

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
        """Train the model with data of this client."""

        try:

            self.lt[me] = t

            # ============================================================
            # Previous local label distribution
            # ============================================================

            p_old = copy.deepcopy(
                self.p_ME[me]
            )

            # ============================================================
            # Previous CD reference
            #
            # This reference is independent from recent_trainloader.
            # ============================================================

            cd_reference_loader = (
                self.cd_reference_trainloader[me]
            )

            # ============================================================
            # Train using the parent implementation
            # ============================================================

            parameters, size, metrics = (
                super().fit(
                    me,
                    t,
                    global_model
                )
            )

            self.train_losses[me].append(
                metrics["train_loss"]
            )

            self.train_accuracies[me].append(
                metrics["train_accuracy"]
            )

            # ============================================================
            # Current local label distribution
            # ============================================================

            p_current = copy.deepcopy(
                self.p_ME[me]
            )

            # ============================================================
            # Existing similarity metric
            #
            # Kept for backward compatibility.
            # It is NOT used for LS/CD detection.
            # ============================================================

            similarity = min(
                cosine_similarity(
                    p_current,
                    p_old
                ),
                1.0
            )

            if 1 - similarity < 0:
                print(
                    f"similaridade is "
                    f"{similarity} "
                    f"rodada {t}"
                )

            # ============================================================
            # LABEL SHIFT
            #
            # LS is the Total Variation Distance between the current
            # and previous local label distributions.
            # ============================================================

            p_current = np.asarray(
                p_current,
                dtype=float
            ).flatten()

            p_old = np.asarray(
                p_old,
                dtype=float
            ).flatten()

            if (
                    p_current.shape
                    != p_old.shape
            ):
                raise ValueError(
                    "Label distribution shapes differ: "
                    f"{p_current.shape} vs "
                    f"{p_old.shape}"
                )

            ls = (
                    0.5
                    * np.sum(
                np.abs(
                    p_current
                    - p_old
                )
            )
            )

            ls = float(
                np.clip(
                    ls,
                    0.0,
                    1.0
                )
            )

            # ============================================================
            # CONCEPT DRIFT
            #
            # CD is based exclusively on P(X | Y).
            #
            # The detector does NOT use:
            #   - experiment_id
            #   - ground truth
            #   - label-shift configuration
            #   - PS
            #   - similarity
            #
            # Therefore a pure change in Dirichlet alpha should not
            # generate CD as long as P(X | Y) remains unchanged.
            # ============================================================

            cd = 0.0

            if (
                    cd_reference_loader is not None
                    and self.trainloader[me] is not None
                    and t > 1
            ):
                input_key = (
                    self.dataset_input_map[
                        self.args.dataset[me]
                    ]
                )

                cd = detect_concept_drift(
                    loader_a=(
                        cd_reference_loader
                    ),
                    loader_b=(
                        self.trainloader[me]
                    ),
                    n_classes=(
                        self.n_classes[me]
                    ),
                    key=input_key,
                    max_samples_per_class=256,
                    n_projections=8,
                    min_samples_per_class=20,
                    random_seed=(
                            42
                            + self.client_id
                            + 1000 * me
                    ),
                    projection_alpha=0.01,
                    min_significant_projections=2
                )

            cd = float(
                np.clip(
                    cd,
                    0.0,
                    1.0
                )
            )

            self.cd_score[me] = cd

            # ============================================================
            # PS
            #
            # Kept only for backward compatibility with MultiFedPredict.
            # It is NOT used for data-shift classification.
            # ============================================================

            ps = (
                    1.0
                    - similarity
            )

            # ============================================================
            # Only scalar metrics leave the client
            # ============================================================

            metrics["non_iid"] = {

                "fc": self.fc_ME[me],

                "il": self.il_ME[me],

                "similarity": similarity,

                "ps": ps,

                "ls": ls,

                "cd": cd
            }

            # ============================================================
            # Update CD reference AFTER detection
            #
            # This is important:
            #
            # round t compares:
            #
            #   reference from t-1
            #       VS
            #   current data from t
            #
            # Only AFTER that comparison do we make the current window
            # the reference for the next round.
            # ============================================================

            self.cd_reference_trainloader[me] = (
                copy.deepcopy(
                    self.trainloader[me]
                )
            )

            self.cd_reference_window[me] = (
                self.concept_drift_window_train[me]
            )

            if t in [
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90
            ]:
                print(
                    f"cliente #{self.client_id} "
                    f"rodada {t} "
                    f"modelo {me} "
                    f"accuracies "
                    f"{self.train_accuracies[me]} "
                    f"LS={ls:.6f} "
                    f"CD={cd:.6f}"
                )

            return (
                parameters,
                size,
                metrics
            )

        except Exception as e:

            print(
                "fit error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

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

            cd = float(
                metrics.get(
                    "cd",
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

            # This is now the detector decision produced by the server.
            data_shift_type = (
                metrics.get(
                    "data_shift_type",
                    "NO_SHIFT"
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
            tau_cd = 0.15

            shift_detected = (
                    ls >= tau_ls
                    or cd >= tau_cd
            )

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
            # Diagnostic condition.
            # DH remains a heterogeneity descriptor and is not
            # required to detect LS or CD.
            # ---------------------------------------------------------
            if (
                    fc > a[me]
                    and il < b[me]
                    and data_heterogeneity_degree
                    < tau_dh[me]
                    and shift_detected
                    and nt > 0
            ):
                print(
                    f"detected shift with low DH. "
                    f"cliente {self.client_id} "
                    f"rodada {t} "
                    f"modelo {me} "
                    f"fc={fc} "
                    f"il={il} "
                    f"dh={data_heterogeneity_degree} "
                    f"ls={ls} "
                    f"cd={cd} "
                    f"nt={nt}"
                )

            print(
                f"model {me} "
                f"round {t} "
                f"nt {nt} "
                f"heterogeneity "
                f"{data_heterogeneity_degree} "
                f"ls {ls} "
                f"cd {cd} "
                f"type {data_shift_type}"
            )

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
                    ps={
                        "global": ps,
                        "reference": 0.1
                    },
                    data_shift_type=(
                        data_shift_type
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
            if (
                    gw == 1
                    and t > 10
                    and data_heterogeneity_degree
                    < tau_dh[me]
                    and shift_detected
            ):
                similarity = 1

                set_weights(
                    self.global_model[me],
                    global_model
                )

                combined_model = (
                    self.global_model[me]
                )

            print(
                f"rodada {t} recebido "
                f"fc={fc} "
                f"il={il} "
                f"dh={data_heterogeneity_degree} "
                f"ls={ls} "
                f"cd={cd} "
                f"ps={ps} "
                f"nt={nt} "
                f"type={data_shift_type}"
            )

            # =========================================================
            # IMPORTANT:
            #
            # This is evaluation only.
            # No LS/CD is calculated from this loader.
            # =========================================================
            loss, test_metrics = test(
                combined_model,
                self.valloader[me],
                self.device,
                self.client_id,
                t,
                self.args.dataset[me],
                self.n_classes[me],
                self.concept_drift_window_test[me]
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