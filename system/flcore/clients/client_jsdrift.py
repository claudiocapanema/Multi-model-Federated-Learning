import copy
import math
import random
import sys

import numpy as np
import torch

from flcore.clients.client_multifedavg import MultiFedAvgClient
from flcore.clients.utils.models_utils import get_weights, set_weights, train


class ClientJSDrift(MultiFedAvgClient):
    """JS-Drift adapted to the existing MEFL client abstraction.

    The original JS-Drift procedure compares a client's current label
    distribution with the previous one, converts the Jensen-Shannon
    divergence into a stability coefficient delta = exp(-gamma * JSD),
    and sends that coefficient together with the normal local update.

    MEFL adaptation: the state is maintained independently for every
    model ``me`` because each MEFL model has its own local data stream.
    A client therefore has one previous label distribution per model.
    """

    def __init__(self, args, id, model, fold_id):
        super().__init__(args, id, model, fold_id)

        self.jsdrift_gamma = float(
            getattr(args, "jsdrift_gamma", 0.5)
        )
        self.jsdrift_epsilon = float(
            getattr(args, "jsdrift_epsilon", 1e-10)
        )

        # Evaluation-only threshold. It is NOT part of the paper's
        # weighting mechanism. The paper defines continuous weighting,
        # not a binary detector threshold.
        self.jsdrift_detection_threshold = float(
            getattr(args, "jsdrift_threshold", 0.1)
        )

        if self.jsdrift_gamma < 0:
            raise ValueError("jsdrift_gamma must be >= 0")
        if self.jsdrift_epsilon <= 0:
            raise ValueError("jsdrift_epsilon must be > 0")
        if self.jsdrift_detection_threshold < 0:
            raise ValueError("jsdrift_threshold must be >= 0")

        self.previous_label_distribution = {
            me: None for me in range(self.ME)
        }
        self.previous_distribution_round = {
            me: None for me in range(self.ME)
        }

    def _label_distribution(self, me):
        """Return the normalized local label distribution for model ``me``."""
        loader = self.trainloader[me]
        if loader is None:
            raise RuntimeError(
                f"JS-Drift: trainloader is None for client={self.client_id}, me={me}"
            )

        dataset = loader.dataset
        labels = None

        # Standard path: Hugging Face Dataset / Dataset-like object.
        if hasattr(dataset, "column_names") and "label" in dataset.column_names:
            try:
                labels = dataset["label"]
            except Exception:
                labels = None

        # Common PyTorch Dataset path.
        if labels is None and hasattr(dataset, "labels"):
            try:
                labels = dataset.labels
            except Exception:
                labels = None

        # Fallback for custom datasets, including the gradual label-shift
        # mixture used by this project. We only read labels; no raw sample
        # is sent to the server.
        if labels is None:
            labels_list = []
            for sample in dataset:
                if isinstance(sample, dict):
                    if "label" not in sample:
                        raise KeyError(
                            "JS-Drift could not find 'label' in a dataset sample"
                        )
                    label = sample["label"]
                elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    label = sample[-1]
                else:
                    raise TypeError(
                        "JS-Drift could not infer the label from a dataset sample"
                    )

                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels_list.append(int(label))

            labels = labels_list

        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        labels = labels[(labels >= 0) & (labels < self.n_classes[me])]

        if labels.size == 0:
            raise RuntimeError(
                f"JS-Drift found no valid labels for client={self.client_id}, me={me}"
            )

        counts = np.bincount(
            labels,
            minlength=self.n_classes[me]
        ).astype(np.float64)

        # Reference article: add epsilon before normalization so absent
        # classes do not cause numerical problems in the divergence.
        distribution = counts + self.jsdrift_epsilon
        distribution /= distribution.sum()
        return distribution

    @staticmethod
    def _js_divergence(p, q):
        """Compute the Jensen-Shannon divergence using natural logarithms."""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        m = 0.5 * (p + q)

        p_safe = np.clip(p, 1e-300, None)
        q_safe = np.clip(q, 1e-300, None)
        m_safe = np.clip(m, 1e-300, None)

        return float(
            0.5 * np.sum(p_safe * np.log(p_safe / m_safe))
            + 0.5 * np.sum(q_safe * np.log(q_safe / m_safe))
        )

    def compute_jsdrift(self, me, t):
        """Compute current JSD and stability coefficient for one MEFL model."""
        current = self._label_distribution(me)
        previous = self.previous_label_distribution[me]

        if previous is None:
            jsd = 0.0
        else:
            jsd = self._js_divergence(current, previous)

        delta = math.exp(
            -self.jsdrift_gamma * jsd
        )

        self.previous_label_distribution[me] = current
        self.previous_distribution_round[me] = t

        return jsd, delta, current

    def fit(self, me, t, global_model):
        """Train locally and return the JS-Drift coefficient with the update."""
        try:
            g = torch.Generator()
            g.manual_seed(t + self.fold_id)
            random.seed(t + self.fold_id)
            np.random.seed(t + self.fold_id)
            torch.manual_seed(t + self.fold_id)

            set_weights(self.model[me], global_model)

            # Keep the project's existing data-shift mechanism unchanged.
            if t > 1:
                self.update_local_train_data(t, me)

            self.lt[me] = t

            jsd, delta, _ = self.compute_jsdrift(me, t)
            drift_detected = int(
                jsd >= self.jsdrift_detection_threshold
            )

            self.optimizer[me] = self._get_optimizer(
                dataset_name=self.args.dataset[me],
                me=me
            )

            print(
                f"[JS-Drift] client={self.client_id} model={me} "
                f"round={t} JSD={jsd:.8f} delta={delta:.8f} "
                f"gamma={self.jsdrift_gamma:.4f} "
                f"threshold={self.jsdrift_detection_threshold:.4f}"
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
            results["JSD"] = jsd
            results["JS-Drift coefficient"] = delta
            results["Drift detected"] = drift_detected
            results["Data shift"] = (
                "DATA_SHIFT" if drift_detected else "NO_SHIFT"
            )

            self.loss_ME[me] = results["train_loss"]

            return (
                get_weights(self.model[me]),
                len(self.trainloader[me].dataset),
                results
            )

        except Exception as e:
            print("JS-Drift fit error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )
            raise