import copy
import math
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.client_multifedavg import MultiFedAvgClient
from flcore.clients.utils.models_utils import (
    get_weights,
    set_weights,
    test,
    train,
)


class ClientCDAFedAvg(MultiFedAvgClient):
    """
    CDA-FedAvg adapted to the existing MEFL/MultiFedAvg client.

    The detector follows Casado et al.:
      - confidence of the current classifier is used as the unlabeled signal;
      - confidences are stored in a bounded sliding window;
      - the window is split at candidate change points;
      - the two parts are modeled by Beta distributions;
      - a CUSUM/log-likelihood-ratio score is computed;
      - a drift is declared when the maximum score exceeds -log(lambda).

    The implementation keeps the existing MultiFedAvg/MEFL data and
    training machinery. Detection itself never uses y.

    Important MEFL adaptation:
      the original CDA-FedAvg is asynchronous and operates on a continuous
      stream. Here the detector is evaluated over the currently available
      trainloader whenever a client participates. This preserves the
      label-free detector while fitting the synchronous MultiFedAvg loop.

    Returned fit result:
        (model_weights, n_examples, metrics)
    """

    def __init__(self, args, id, model, fold_id):
        super().__init__(args, id, model, fold_id)

        # Paper defaults.
        self.cda_lambda = float(getattr(args, "cda_lambda", 0.05))
        self.cda_delta = int(getattr(args, "cda_delta", 100))
        self.cda_window_size = int(getattr(args, "cda_window_size", 1000))

        # The paper executes Algorithm 5 with probability exp(-2*q_i).
        # Set to False only if you want deterministic evaluation at every
        # sample; True is the paper-faithful default.
        self.cda_stochastic_detection = bool(
            getattr(args, "cda_stochastic_detection", True)
        )

        # Adaptation/rehearsal parameters from the paper.
        self.cda_memory_size = int(getattr(args, "cda_memory_size", 1400))
        self.cda_adaptation_rounds = int(
            getattr(args, "cda_adaptation_rounds", 5)
        )

        # MEFL compatibility: normally local training continues every selected
        # round, while the detector/adaptation is added on top.
        # Set True to reproduce the "train only when drift is detected" idea
        # more literally.
        self.cda_skip_training_without_drift = bool(
            getattr(args, "cda_skip_training_without_drift", False)
        )

        self.cda_confidence_history = {
            me: deque(maxlen=self.cda_window_size)
            for me in range(self.ME)
        }

        self.cda_long_term_memory = {
            me: []
            for me in range(self.ME)
        }

        self.cda_seen_concepts = {
            me: 0
            for me in range(self.ME)
        }

        self.cda_last_change_point = {
            me: None
            for me in range(self.ME)
        }

        self.cda_detection_count = {
            me: 0
            for me in range(self.ME)
        }

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_batch(batch, dataset_name=None):
        if isinstance(batch, dict):
            if "label" not in batch:
                raise KeyError(
                    f"Could not find 'label' in batch for dataset "
                    f"{dataset_name!r}. Available keys: {list(batch.keys())}"
                )

            dataset_key = {
                "CIFAR10": "img",
                "MNIST": "image",
                "EMNIST": "image",
                "GTSRB": "image",
                "ImageNet": "image",
                "ImageNet10": "image",
                "Gowalla": "sequence",
                "WISDM-W": "sequence",
                "WISDM-P": "sequence",
                "Foursquare": "sequence",
                "wikitext": "text",
            }.get(dataset_name)

            candidates = []
            if dataset_key is not None:
                candidates.append(dataset_key)
            candidates.extend(
                [
                    "img",
                    "image",
                    "sequence",
                    "text",
                    "data",
                    "x",
                    "input",
                    "features",
                ]
            )

            for key in candidates:
                if key in batch:
                    return batch[key], batch["label"]

            raise KeyError(
                f"Could not find input tensor for dataset {dataset_name!r}. "
                f"Available keys: {list(batch.keys())}"
            )

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(
            f"Unsupported dataloader batch format for dataset "
            f"{dataset_name!r}: {type(batch).__name__}"
        )

    @staticmethod
    def _as_logits(output):
        if isinstance(output, dict):
            for key in ("logits", "output", "out", "prediction"):
                if key in output:
                    return output[key]
            raise TypeError("Model returned a dict without a logits-like key.")

        if isinstance(output, (tuple, list)):
            return output[0]

        return output

    # ------------------------------------------------------------------
    # CDA-FedAvg detector
    # ------------------------------------------------------------------
    @staticmethod
    def _beta_parameters(values):
        """
        Method-of-moments estimator used by the paper.

        For mean m and variance v:
            alpha = m * (m(1-m)/v - 1)
            beta  = (1-m) * (m(1-m)/v - 1)
        """
        x = np.asarray(values, dtype=np.float64)

        mean = float(np.mean(x))
        var = float(np.var(x, ddof=0))

        eps = 1e-8
        mean = float(np.clip(mean, eps, 1.0 - eps))

        # A Beta distribution has v < m(1-m). Numerical clipping is needed
        # because confidence values can be almost constant.
        max_var = max(mean * (1.0 - mean) - eps, eps)
        var = float(np.clip(var, eps, max_var))

        common = mean * (1.0 - mean) / var - 1.0
        common = max(common, eps)

        alpha = max(mean * common, eps)
        beta = max((1.0 - mean) * common, eps)

        return alpha, beta

    @staticmethod
    def _beta_logpdf(values, alpha, beta):
        x = np.asarray(values, dtype=np.float64)
        x = np.clip(x, 1e-7, 1.0 - 1e-7)

        log_beta = (
            math.lgamma(alpha)
            + math.lgamma(beta)
            - math.lgamma(alpha + beta)
        )

        return (
            (alpha - 1.0) * np.log(x)
            + (beta - 1.0) * np.log1p(-x)
            - log_beta
        )

    def _detect_change_point(self, q_values):
        """
        Algorithm 5 from CDA-FedAvg.

        Returns:
            detected: bool
            k_max: estimated change point or None
            score: maximum log-likelihood-ratio score
        """
        q = np.asarray(q_values, dtype=np.float64)
        n = len(q)

        delta = self.cda_delta
        if n < 2 * delta:
            return False, None, 0.0

        lam = self.cda_lambda
        threshold = -math.log(lam)

        best_score = 0.0
        best_k = None

        # k is the split point. Q_b is the historical part and Q_a is
        # the recent part, matching the paper's description.
        for k in range(delta, n - delta + 1):
            q_b = q[:k]
            q_a = q[k:]

            mean_b = float(np.mean(q_b))
            mean_a = float(np.mean(q_a))

            # The paper searches only for decreases in confidence.
            if mean_a > (1.0 - lam) * mean_b:
                continue

            alpha_b, beta_b = self._beta_parameters(q_b)
            alpha_a, beta_a = self._beta_parameters(q_a)

            # CUSUM/log-likelihood-ratio score over the complete window.
            # Positive values indicate that the recent distribution is more
            # likely under the recent-window Beta model.
            log_recent = self._beta_logpdf(q, alpha_a, beta_a)
            log_old = self._beta_logpdf(q, alpha_b, beta_b)

            score = float(np.sum(log_recent - log_old))

            if score > best_score:
                best_score = score
                best_k = k

        return best_score > threshold, best_k, best_score

    @torch.no_grad()
    def _detect_drift_on_stream(self, me, model):
        """
        Feed the currently available samples through the detector.

        Labels are unpacked only because the existing PFLlib dataloader
        returns (x, y). y is never used by the detector.
        """
        model.eval()

        loader = self.trainloader[me]
        dataset_name = self.args.dataset[me]

        detected = False
        detected_k = None
        detected_score = 0.0

        for batch in loader:
            x, _ = self._unpack_batch(batch, dataset_name)

            x = x.to(self.device)
            logits = self._as_logits(model(x))

            confidence = torch.softmax(logits, dim=1).max(dim=1).values
            confidences = confidence.detach().cpu().numpy()

            for q_i in confidences:
                q_i = float(np.clip(q_i, 1e-7, 1.0 - 1e-7))

                self.cda_confidence_history[me].append(q_i)

                q = self.cda_confidence_history[me]
                if len(q) < 2 * self.cda_delta:
                    continue

                if self.cda_stochastic_detection:
                    # Algorithm 4: execute Algorithm 5 with probability e^(-2q_i).
                    if random.random() > math.exp(-2.0 * q_i):
                        continue

                detected, detected_k, detected_score = (
                    self._detect_change_point(list(q))
                )

                if detected:
                    # Algorithm 4: reset Q after detection.
                    self.cda_confidence_history[me].clear()
                    self.cda_last_change_point[me] = detected_k
                    self.cda_detection_count[me] += 1
                    return True, detected_k, detected_score

        return False, detected_k, detected_score

    # ------------------------------------------------------------------
    # Long-term memory / rehearsal
    # ------------------------------------------------------------------
    def _snapshot_batch(self, x, y):
        return (
            x.detach().cpu().clone(),
            y.detach().cpu().long().clone(),
        )

    @torch.no_grad()
    def _collect_memory(self, me):
        """
        Keep a bounded representative memory for rehearsal.

        The paper's Algorithm 6 keeps enough data from each concept and
        guarantees class coverage. In this MEFL implementation, the memory
        is bounded by cda_memory_size and filled from the currently available
        labeled data. The detector itself remains label-free.
        """
        loader = self.trainloader[me]
        dataset_name = self.args.dataset[me]

        samples = []
        per_class = {}

        for batch in loader:
            x, y = self._unpack_batch(batch, dataset_name)

            x_cpu = x.detach().cpu()
            y_cpu = y.detach().cpu().long()

            for i in range(len(y_cpu)):
                label = int(y_cpu[i])
                per_class.setdefault(label, []).append(
                    (x_cpu[i].clone(), y_cpu[i].clone())
                )

        if not per_class:
            return []

        labels = sorted(per_class.keys())
        quota = max(1, self.cda_memory_size // max(1, len(labels)))

        selected = []
        for label in labels:
            selected.extend(per_class[label][:quota])

        # Fill remaining slots deterministically.
        if len(selected) < self.cda_memory_size:
            remaining = []
            for label in labels:
                remaining.extend(per_class[label][quota:])
            selected.extend(remaining[: self.cda_memory_size - len(selected)])

        return selected[: self.cda_memory_size]

    def _update_long_term_memory(self, me):
        new_memory = self._collect_memory(me)

        if not new_memory:
            return

        self.cda_long_term_memory[me] = new_memory
        self.cda_seen_concepts[me] += 1

    def _train_with_rehearsal(self, me, t):
        """
        Paper-inspired drift adaptation.

        The original CDA-FedAvg trains on long-term memory after a detected
        drift. Here we preserve the existing MultiFedAvg optimizer/training
        path and augment the current local training with the stored memory
        through repeated local updates.
        """
        if not self.cda_long_term_memory[me]:
            return None

        model = self.model[me]
        model.train()

        optimizer = self._get_optimizer(
            dataset_name=self.args.dataset[me],
            me=me,
        )

        # We intentionally do not construct a second DataLoader here because
        # the existing training helper controls dataset-specific batch logic.
        # Instead, the standard current trainloader is used for the main
        # update, while the long-term memory is replayed with a lightweight
        # manual SGD pass.
        batch_size = max(
            1,
            int(getattr(self.args, "batch_size", getattr(self, "batch_size", 32))),
        )

        memory = self.cda_long_term_memory[me]

        for _ in range(max(1, self.cda_adaptation_rounds)):
            random.shuffle(memory)

            for start in range(0, len(memory), batch_size):
                batch = memory[start : start + batch_size]
                if not batch:
                    continue

                x = torch.stack([item[0] for item in batch]).to(self.device)
                y = torch.stack([item[1] for item in batch]).to(
                    self.device
                ).long()

                optimizer.zero_grad()
                logits = self._as_logits(model(x))
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

        return True

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, me, t, global_model):
        try:
            random.seed(t + self.fold_id)
            np.random.seed(t + self.fold_id)
            torch.manual_seed(t + self.fold_id)

            self.update_local_test_data(t, me)
            set_weights(self.model[me], global_model)

            loss, metrics = test(
                self.model[me],
                self.valloader[me],
                self.device,
                self.client_id,
                t,
                self.args.dataset[me],
                self.n_classes[me],
                self.concept_drift_window_test[me],
            )

            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            metrics["Alpha"] = self.alpha_test[me]

            return (
                loss,
                len(self.valloader[me].dataset),
                metrics,
            )

        except Exception as e:
            print("CDA-FedAvg evaluate error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )
            raise

    # ------------------------------------------------------------------
    # Local training + detection/adaptation
    # ------------------------------------------------------------------
    def fit(self, me, t, global_model):
        try:
            self.lt[me] = t
            set_weights(self.model[me], global_model)

            if t > 1:
                self.update_local_train_data(t, me)

            # ----------------------------------------------------------
            # 1. Detect BEFORE local supervised training.
            #    Detection uses confidence only; y is ignored.
            # ----------------------------------------------------------
            drift_detected, change_point, drift_score = (
                self._detect_drift_on_stream(me, self.model[me])
            )

            # ----------------------------------------------------------
            # 2. If drift is detected, update rehearsal memory.
            # ----------------------------------------------------------
            if drift_detected:
                self._update_long_term_memory(me)

            # ----------------------------------------------------------
            # 3. Standard MultiFedAvg local training.
            # ----------------------------------------------------------
            should_train = (
                (not self.cda_skip_training_without_drift)
                or t == 1
                or drift_detected
            )

            if should_train:
                self.optimizer[me] = self._get_optimizer(
                    dataset_name=self.args.dataset[me],
                    me=me,
                )

                results = train(
                    model=self.model[me],
                    trainloader=self.trainloader[me],
                    valloader=self.valloader[me],
                    optimizer=self.optimizer[me],
                    epochs=self.local_epochs,
                    learning_rate=self.lr,
                    device=self.device,
                    client_id=self.client_id,
                    t=t,
                    dataset_name=self.args.dataset[me],
                    n_classes=self.n_classes[me],
                    concept_drift_window=self.concept_drift_window_train[me],
                )
            else:
                results = {
                    "train_loss": float("nan"),
                    "Accuracy": float("nan"),
                    "Balanced accuracy": float("nan"),
                }

            # ----------------------------------------------------------
            # 4. Rehearsal adaptation after a detected drift.
            # ----------------------------------------------------------
            if drift_detected:
                self._train_with_rehearsal(me, t)

            results["me"] = me
            results["client_id"] = self.client_id
            results["Model size"] = self.models_size[me]
            results["alpha"] = self.alpha_train[me]

            # Metrics expected by the existing detection server.
            results["Drift detected"] = int(drift_detected)
            results["Data shift"] = (
                "DATA_SHIFT" if drift_detected else "NO_SHIFT"
            )
            results["CDA-FedAvg change point"] = (
                int(change_point) if change_point is not None else -1
            )
            results["CDA-FedAvg drift score"] = float(drift_score)
            results["CDA-FedAvg lambda"] = self.cda_lambda
            results["CDA-FedAvg delta"] = self.cda_delta
            results["CDA-FedAvg window"] = self.cda_window_size
            results["CDA-FedAvg memory size"] = self.cda_memory_size
            results["CDA-FedAvg adaptation rounds"] = (
                self.cda_adaptation_rounds
            )
            results["CDA-FedAvg detection count"] = (
                self.cda_detection_count[me]
            )

            self.loss_ME[me] = results["train_loss"]

            return (
                get_weights(self.model[me]),
                len(self.trainloader[me].dataset),
                results,
            )

        except Exception as e:
            print("CDA-FedAvg fit error")
            print(
                "Error on line {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                )
            )
            raise