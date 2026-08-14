import copy
import numpy as np
import torch
from scipy.stats import norm
import math


from flcore.clients.client_multifedavg import MultiFedAvgClient
from flcore.clients.utils.models_utils import (
    get_weights,
    set_weights,
    test,
    train
)

from flcore.clients.utils.models_utils import (
    DATASET_INPUT_MAP,
    get_weights,
    set_weights,
    test,
    train
)


class ClientCDAFedAvg(MultiFedAvgClient):

    def __init__(self, args, id, model, fold_id):
        super().__init__(args, id, model, fold_id)

        # ============================================================
        # CDA-FedAvg parameters
        # ============================================================

        # Sensitivity to change: lambda = 0.05
        self.cda_lambda = {
            me: 0.05
            for me in range(self.ME)
        }

        # Minimum size of each sub-window
        # Delta = 100
        self.cda_delta = {
            me: 100
            for me in range(self.ME)
        }

        # Maximum size of the short-term memory Q
        # Nmax = 1000
        self.cda_nmax = {
            me: 1000
            for me in range(self.ME)
        }

        # Minimum amount of data per concept in long-term memory
        # L = 1400
        self.cda_memory_size = {
            me: 1400
            for me in range(self.ME)
        }

        # Number of local training rounds per concept
        # R = 5
        self.cda_rounds_per_concept = {
            me: 5
            for me in range(self.ME)
        }

        # ============================================================
        # Short-term memory Q
        # ============================================================

        # Confidence values.
        self.cda_Q = {
            me: []
            for me in range(self.ME)
        }

        # Corresponding raw samples.
        #
        # cda_Q_data[me][i] corresponds to cda_Q[me][i].
        #
        # This is necessary because k_max is used to identify
        # the samples belonging to the new concept.
        self.cda_Q_data = {
            me: []
            for me in range(self.ME)
        }

        # ============================================================
        # Long-term memory L
        # ============================================================

        # L[me][concept_id] = {
        #     "X": [...],
        #     "y": [...]
        # }
        #
        # This preserves representative samples from every
        # concept seen by each model.
        self.cda_L = {
            me: {}
            for me in range(self.ME)
        }

        # ============================================================
        # Concept state
        # ============================================================

        # Current concept for each MEFL model.
        self.cda_concept_id = {
            me: 0
            for me in range(self.ME)
        }

        # Whether the first concept has already been learned.
        self.cda_initialized = {
            me: False
            for me in range(self.ME)
        }

        # Whether the client is currently collecting samples
        # for the current concept.
        self.cda_collecting_concept = {
            me: True
            for me in range(self.ME)
        }

        # Whether the client is currently adapting/training
        # after obtaining enough samples for the concept.
        self.cda_training = {
            me: False
            for me in range(self.ME)
        }

        # Number of local training rounds already performed
        # for the current concept.
        self.cda_training_round = {
            me: 0
            for me in range(self.ME)
        }

        # ============================================================
        # Stream state
        # ============================================================

        self.cda_stream_position = {
            me: 0
            for me in range(self.ME)
        }

        # ============================================================
        # Drift detection state
        # ============================================================

        self.cda_kmax = {
            me: None
            for me in range(self.ME)
        }

        self.cda_drift_detected = {
            me: False
            for me in range(self.ME)
        }

        self.cda_detection_score = {
            me: 0.0
            for me in range(self.ME)
        }

        self.cda_detection_threshold = {
            me: -np.log(
                self.cda_lambda[me]
            )
            for me in range(self.ME)
        }

        # ============================================================
        # Data-shift state
        # ============================================================

        self.cda_data_shift = {
            me: "NO_SHIFT"
            for me in range(self.ME)
        }

        self.cda_new_concept_data = {
            me: 0
            for me in range(self.ME)
        }

    def fit(self, me, t, global_model):

        print(
            f"[CDA-FedAvg] CLIENT {self.client_id} ENTERED FIT "
            f"round={t}"
        )

        self.lt[me] = t

        # ============================================================
        # Reset event state for this call
        # ============================================================

        self.cda_drift_detected[me] = False
        self.cda_data_shift[me] = "NO_SHIFT"
        self.cda_detection_score[me] = 0.0
        self.cda_kmax[me] = None

        # ============================================================
        # Update local data according to the existing MEFL
        # data-shift mechanism.
        # ============================================================

        if t > 1:
            self.update_local_train_data(
                t,
                me
            )

        # ============================================================
        # Acquire new data
        # ============================================================

        X_new, y_new = self._cda_get_stream_batch(
            me
        )

        if X_new is None:
            return (
                get_weights(
                    self.model[me]
                ),
                0,
                {
                    "me": me,
                    "client_id": self.client_id,
                    "CDA training": 0,
                    "Drift detected": 0,
                    "Data shift": "NO_SHIFT",
                    "Detection score": 0.0,
                    "Detection threshold":
                        self.cda_detection_threshold[me],
                    "Concept":
                        self.cda_concept_id[me],
                    "Model size":
                        self.models_size[me],
                    "alpha":
                        self.alpha_train[me],
                    "train_loss": 0.0,
                }
            )

        # ============================================================
        # 1. INITIAL CONCEPT / NEW CONCEPT COLLECTION
        # ============================================================

        if self.cda_collecting_concept[me]:

            self._cda_add_to_long_term_memory(
                me,
                X_new,
                y_new
            )

            concept_id = self.cda_concept_id[me]

            self.cda_new_concept_data[me] = len(
                self.cda_L[me][concept_id]["y"]
            )

            # --------------------------------------------------------
            # Keep collecting until:
            #
            # count(class) >= L / (2M)
            #
            # for every class.
            # --------------------------------------------------------

            print(
                f"[CDA-FedAvg] CLIENT {self.client_id} RETURN "
                f"round={t} "
                f"model={me} "
            )

            if self._cda_has_balanced_concept(me):

                self.cda_collecting_concept[me] = False

                self.cda_training[me] = True

                self.cda_training_round[me] = 0

            else:

                return (
                    get_weights(
                        self.model[me]
                    ),
                    0,
                    {
                        "me": me,
                        "client_id": self.client_id,
                        "CDA training": 0,
                        "Drift detected": 0,
                        "Data shift": "NO_SHIFT",
                        "Detection score": 0.0,
                        "Detection threshold":
                            self.cda_detection_threshold[me],
                        "Concept":
                            self.cda_concept_id[me],
                        "Model size":
                            self.models_size[me],
                        "alpha":
                            self.alpha_train[me],
                        "train_loss": 0.0,
                    }
                )

        # ============================================================
        # 2. LOCAL REHEARSAL / ADAPTATION
        # ============================================================

        if self.cda_training[me]:

            # The current global model is the starting point for
            # this local update.
            set_weights(
                self.model[me],
                global_model
            )

            memory_loader = (
                self._cda_build_memory_loader(me)
            )

            if memory_loader is None:
                return (
                    get_weights(
                        self.model[me]
                    ),
                    0,
                    {
                        "me": me,
                        "client_id": self.client_id,
                        "CDA training": 0,
                        "Drift detected": 0,
                        "Data shift": "NO_SHIFT",
                        "Detection score": 0.0,
                        "Detection threshold":
                            self.cda_detection_threshold[me],
                        "Concept":
                            self.cda_concept_id[me],
                        "Model size":
                            self.models_size[me],
                        "alpha":
                            self.alpha_train[me],
                        "train_loss": 0.0,
                    }
                )

            # --------------------------------------------------------
            # Standard FedAvg local optimizer.
            #
            # No FedConD proximal term.
            # --------------------------------------------------------

            self.optimizer[me] = self._get_optimizer(
                dataset_name=self.args.dataset[me],
                me=me
            )

            results = train(
                model=self.model[me],
                trainloader=memory_loader,
                valloader=self.valloader[me],
                optimizer=self.optimizer[me],
                epochs=self.local_epochs,
                learning_rate=self.lr,
                device=self.device,
                client_id=self.client_id,
                t=t,
                dataset_name=self.args.dataset[me],
                n_classes=self.n_classes[me],
                concept_drift_window=
                self.concept_drift_window_train[me]
            )

            self.cda_training_round[me] += 1

            # --------------------------------------------------------
            # CDA-FedAvg performs a limited number R of local
            # training rounds for the concept.
            # --------------------------------------------------------

            if (
                    self.cda_training_round[me]
                    >= self.cda_rounds_per_concept[me]
            ):
                self.cda_training[me] = False

            results["me"] = me
            results["client_id"] = self.client_id

            results["Model size"] = (
                self.models_size[me]
            )

            results["alpha"] = (
                self.alpha_train[me]
            )

            results["Data shift"] = (
                "NO_SHIFT"
            )

            results["Drift detected"] = 0

            results["Detection score"] = (
                self.cda_detection_score[me]
            )

            results["Detection threshold"] = (
                self.cda_detection_threshold[me]
            )

            results["Concept"] = (
                self.cda_concept_id[me]
            )

            results["CDA training"] = 1

            self.loss_ME[me] = (
                results["train_loss"]
            )

            return (
                get_weights(
                    self.model[me]
                ),
                len(memory_loader.dataset),
                results
            )

        # ============================================================
        # 3. MONITORING / DRIFT DETECTION
        # ============================================================

        confidence = self._cda_get_confidence(
            me,
            X_new
        )

        # Process every newly acquired instance.
        for i, q in enumerate(confidence):

            q = float(q)

            x_i = X_new[i].detach().cpu().clone()

            y_i = y_new[i].detach().cpu().clone()

            # --------------------------------------------------------
            # Add confidence and corresponding raw sample to Q.
            # --------------------------------------------------------

            self.cda_Q[me].append(q)

            self.cda_Q_data[me].append(
                (
                    x_i,
                    y_i
                )
            )

            # --------------------------------------------------------
            # Sliding window.
            # --------------------------------------------------------

            while (
                    len(self.cda_Q[me])
                    > self.cda_nmax[me]
            ):
                self.cda_Q[me].pop(0)

                self.cda_Q_data[me].pop(0)

            # --------------------------------------------------------
            # Run Algorithm 5 with probability exp(-2q_i).
            # --------------------------------------------------------

            detection_probability = np.exp(
                -2.0 * q
            )

            if (
                    np.random.random()
                    > detection_probability
            ):
                continue

            (
                drift,
                k_max,
                score
            ) = self._cda_detect_drift(me)

            self.cda_detection_score[me] = (
                score
            )

            if not drift:
                continue

            # ========================================================
            # DRIFT DETECTED
            # ========================================================

            self.cda_drift_detected[me] = True
            self.cda_data_shift[me] = (
                "DATA_SHIFT"
            )

            self.cda_kmax[me] = k_max

            # --------------------------------------------------------
            # Samples after k_max belong to the new concept.
            #
            # Q[k_max:] is precisely the portion identified by
            # Algorithm 5 after the detected cut-off point.
            # --------------------------------------------------------

            new_concept_samples = (
                self.cda_Q_data[me][k_max:]
            )

            # --------------------------------------------------------
            # Move to a new concept.
            # --------------------------------------------------------

            self.cda_concept_id[me] += 1

            new_concept_id = (
                self.cda_concept_id[me]
            )

            self.cda_L[me][new_concept_id] = {
                "X": [],
                "y": []
            }

            for x_new, y_new_item in (
                    new_concept_samples
            ):
                self.cda_L[me][new_concept_id]["X"].append(
                    x_new.clone()
                )

                self.cda_L[me][new_concept_id]["y"].append(
                    int(y_new_item.item())
                )

            self.cda_new_concept_data[me] = len(
                self.cda_L[me][new_concept_id]["y"]
            )

            # --------------------------------------------------------
            # Reset Q after detecting drift.
            # The paper explicitly reinitializes Q.
            # --------------------------------------------------------

            self.cda_Q[me] = []
            self.cda_Q_data[me] = []

            # --------------------------------------------------------
            # The new concept may already have enough data.
            # Otherwise, continue collecting until it does.
            # --------------------------------------------------------

            if self._cda_has_balanced_concept(me):

                self.cda_collecting_concept[me] = False

                self.cda_training[me] = True

                self.cda_training_round[me] = 0

            else:

                self.cda_collecting_concept[me] = True

                self.cda_training[me] = False

            # Only one drift event is handled per fit call.
            break

        # ============================================================
        # 4. MONITORING RESULT
        # ============================================================

        results = {
            "me": me,
            "client_id": self.client_id,
            "Model size":
                self.models_size[me],
            "alpha":
                self.alpha_train[me],
            "Data shift":
                self.cda_data_shift[me],
            "Drift detected":
                int(
                    self.cda_drift_detected[me]
                ),
            "Detection score":
                self.cda_detection_score[me],
            "Detection threshold":
                self.cda_detection_threshold[me],
            "Concept":
                self.cda_concept_id[me],
            "CDA training": 0,
            "train_loss": 0.0,
        }

        return (
            get_weights(
                self.model[me]
            ),
            0,
            results
        )

    @torch.no_grad()
    def _cda_get_confidence(self, me, x):
        """
        Compute the classifier confidence for one or more samples.

        CDA-FedAvg defines confidence as the maximum posterior
        probability among all classes.
        """

        model = self.model[me]
        model.eval()

        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.to(self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        output = model(x)

        # Some models may return tuples/lists.
        if isinstance(output, (tuple, list)):
            output = output[0]

        probabilities = torch.softmax(output, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        return confidence.detach().cpu().numpy()

    def _cda_estimate_beta_parameters(self, values):
        """
        Estimate Beta(alpha, beta) parameters using the
        method of moments, as in Algorithm 5.
        """

        values = np.asarray(values, dtype=np.float64)

        if len(values) < 2:
            return None, None

        mean = np.mean(values)
        variance = np.var(values, ddof=1)

        mean = np.clip(mean, 1e-6, 1.0 - 1e-6)
        variance = max(variance, 1e-8)

        # Method of moments for Beta distribution:
        #
        # alpha = mean * ((mean * (1-mean) / variance) - 1)
        # beta  = (1-mean) * ((mean * (1-mean) / variance) - 1)

        common = (
                         mean * (1.0 - mean) / variance
                 ) - 1.0

        alpha = mean * common
        beta = (1.0 - mean) * common

        # Numerical protection.
        alpha = max(alpha, 1e-6)
        beta = max(beta, 1e-6)

        return alpha, beta

    def _cda_beta_logpdf(self, values, alpha, beta):
        """
        Stable log-PDF of a Beta distribution.
        """

        values = np.asarray(values, dtype=np.float64)

        values = np.clip(
            values,
            1e-8,
            1.0 - 1e-8
        )

        log_pdf = (
                (alpha - 1.0) * np.log(values)
                + (beta - 1.0) * np.log(1.0 - values)
                - (
                        math.lgamma(alpha)
                        + math.lgamma(beta)
                        - math.lgamma(alpha + beta)
                )
        )

        return log_pdf

    def _cda_detect_drift(self, me):
        """
        CDA-FedAvg drift detection.

        Implements Algorithm 5 from the paper.

        Returns:
            drift_detected: bool
            k_max: detected change point
            s_f: maximum dissimilarity score
        """

        Q = np.asarray(
            self.cda_Q[me],
            dtype=np.float64
        )

        N = len(Q)

        delta = self.cda_delta[me]
        lam = self.cda_lambda[me]

        # ------------------------------------------------------------
        # The algorithm requires at least 2*Delta observations.
        # ------------------------------------------------------------

        if N < 2 * delta:
            return False, None, 0.0

        s_f = 0.0
        k_max = None

        # ------------------------------------------------------------
        # Algorithm 5:
        #
        # for k = Delta ... N-Delta
        #
        # Q_b = Q[0:k]
        # Q_a = Q[k:N]
        #
        # Q_a is the most recent sub-window.
        # ------------------------------------------------------------

        for k in range(delta, N - delta + 1):

            Q_b = Q[:k]
            Q_a = Q[k:]

            if len(Q_b) < delta or len(Q_a) < delta:
                continue

            mean_b = np.mean(Q_b)
            mean_a = np.mean(Q_a)

            # --------------------------------------------------------
            # Only negative changes are relevant.
            #
            # m_a <= (1-lambda) * m_b
            # --------------------------------------------------------

            if mean_a > (1.0 - lam) * mean_b:
                continue

            alpha_b, beta_b = (
                self._cda_estimate_beta_parameters(Q_b)
            )

            alpha_a, beta_a = (
                self._cda_estimate_beta_parameters(Q_a)
            )

            if (
                    alpha_b is None
                    or beta_b is None
                    or alpha_a is None
                    or beta_a is None
            ):
                continue

            # --------------------------------------------------------
            # Log-likelihood ratio.
            #
            # s_k = sum log(
            #     f(q_i | alpha_a, beta_a)
            #     /
            #     f(q_i | alpha_b, beta_b)
            # )
            #
            # The paper computes this over Q.
            # --------------------------------------------------------

            log_pdf_a = self._cda_beta_logpdf(
                Q,
                alpha_a,
                beta_a
            )

            log_pdf_b = self._cda_beta_logpdf(
                Q,
                alpha_b,
                beta_b
            )

            s_k = np.sum(
                log_pdf_a - log_pdf_b
            )

            if s_k > s_f:
                s_f = float(s_k)
                k_max = k

        # ------------------------------------------------------------
        # Threshold:
        #
        # T_h = -log(lambda)
        # ------------------------------------------------------------

        threshold = -np.log(lam)

        drift = s_f > threshold

        return drift, k_max, float(s_f)

    def _cda_has_balanced_concept(self, me):
        """
        Check whether the current concept contains enough
        representative samples.

        CDA-FedAvg requires, for every class:

            |L_l^j(c)| >= L / (2M)

        where:
            L = minimum amount of data per concept
            M = number of classes
        """

        concept_id = self.cda_concept_id[me]

        if concept_id not in self.cda_L[me]:
            return False

        labels = self.cda_L[me][concept_id]["y"]

        if len(labels) == 0:
            return False

        labels = np.asarray(labels)

        num_classes = self.n_classes[me]

        minimum_per_class = 0

        for c in range(num_classes):

            count = np.sum(
                labels == c
            )

            if count < minimum_per_class:
                return False

        return True

    def _cda_add_to_long_term_memory(self, me, x, y):
        """
        Add samples to the long-term memory of the current concept.
        """

        concept_id = self.cda_concept_id[me]

        if concept_id not in self.cda_L[me]:
            self.cda_L[me][concept_id] = {
                "X": [],
                "y": []
            }

        if torch.is_tensor(x):
            x_cpu = x.detach().cpu()
        else:
            x_cpu = torch.tensor(x)

        if torch.is_tensor(y):
            y_cpu = y.detach().cpu()
        else:
            y_cpu = torch.tensor(
                y,
                dtype=torch.long
            )

        if x_cpu.dim() == 0:
            x_cpu = x_cpu.unsqueeze(0)

        if y_cpu.dim() == 0:
            y_cpu = y_cpu.unsqueeze(0)

        for i in range(len(y_cpu)):
            self.cda_L[me][concept_id]["X"].append(
                x_cpu[i].clone()
            )

            self.cda_L[me][concept_id]["y"].append(
                int(y_cpu[i].item())
            )

    def _cda_build_memory_loader(self, me):
        """
        Build a DataLoader containing representative samples from
        all concepts stored in the CDA-FedAvg long-term memory.

        The dataset format is kept compatible with the existing
        MEFL train() function.
        """

        from torch.utils.data import Dataset, DataLoader

        X_all = []
        y_all = []

        for concept_id in sorted(
                self.cda_L[me].keys()
        ):

            X_concept = (
                self.cda_L[me][concept_id]["X"]
            )

            y_concept = (
                self.cda_L[me][concept_id]["y"]
            )

            if len(X_concept) == 0:
                continue

            X_all.extend(X_concept)
            y_all.extend(y_concept)

        if len(X_all) == 0:
            return None

        X = torch.stack(X_all)

        y = torch.tensor(
            y_all,
            dtype=torch.long
        )

        dataset_name = self.args.dataset[me]

        input_key = DATASET_INPUT_MAP[
            dataset_name
        ]

        class CDARehearsalDataset(Dataset):

            def __init__(
                    self,
                    X,
                    y,
                    input_key
            ):
                self.X = X
                self.y = y
                self.input_key = input_key

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return {
                    input_key: self.X[idx],
                    "label": self.y[idx]
                }

        dataset = CDARehearsalDataset(
            X,
            y,
            input_key
        )

        batch_size = (
            self.trainloader[me].batch_size
        )

        if batch_size is None:
            batch_size = 1

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def _cda_get_stream_batch(self, me):
        """
        Obtain the next batch from the local data stream.

        The dataset follows the same dictionary format used by the
        existing MEFL training pipeline:

            batch[DATASET_INPUT_MAP[dataset_name]]
            batch["label"]
        """

        loader = self.trainloader[me]

        if loader is None:
            return None, None

        dataset = loader.dataset

        if dataset is None:
            return None, None

        n = len(dataset)

        if n == 0:
            return None, None

        batch_size = loader.batch_size

        if batch_size is None:
            batch_size = 1

        start = self.cda_stream_position[me]

        indices = [
            (start + i) % n
            for i in range(batch_size)
        ]

        self.cda_stream_position[me] = (
                                               start + batch_size
                                       ) % n

        dataset_name = self.args.dataset[me]

        input_key = DATASET_INPUT_MAP[
            dataset_name
        ]

        X = []
        y = []

        for idx in indices:

            sample = dataset[idx]

            if isinstance(sample, dict):

                if input_key not in sample:
                    raise KeyError(
                        f"Dataset '{dataset_name}' "
                        f"does not contain input key "
                        f"'{input_key}'. "
                        f"Available keys: "
                        f"{list(sample.keys())}"
                    )

                if "label" not in sample:
                    raise KeyError(
                        f"Dataset '{dataset_name}' "
                        f"does not contain label key "
                        f"'label'. "
                        f"Available keys: "
                        f"{list(sample.keys())}"
                    )

                x_i = sample[input_key]
                y_i = sample["label"]

            elif isinstance(sample, (tuple, list)):

                x_i = sample[0]
                y_i = sample[1]

            else:

                raise ValueError(
                    "Unsupported dataset sample format "
                    f"for dataset '{dataset_name}'."
                )

            if not torch.is_tensor(x_i):
                x_i = torch.tensor(x_i)

            if not torch.is_tensor(y_i):
                y_i = torch.tensor(
                    y_i,
                    dtype=torch.long
                )

            X.append(x_i)
            y.append(y_i)

        X = torch.stack(X)
        y = torch.stack(y)

        return X, y