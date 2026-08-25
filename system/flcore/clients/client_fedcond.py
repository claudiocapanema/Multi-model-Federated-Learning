import copy
import numpy as np
import torch
from scipy.stats import norm

from flcore.clients.client_multifedavg import MultiFedAvgClient
from flcore.clients.utils.models_utils import (
    get_weights,
    set_weights,
    test,
    train
)


class ClientFedConD(MultiFedAvgClient):

    def __init__(
            self,
            args,
            id,
            model,
            fold_id
    ):
        super().__init__(
            args,
            id,
            model,
            fold_id
        )

        # =========================================================
        # FedConD: historical performance
        # =========================================================

        self.performance_history = {
            me: []
            for me in range(self.ME)
        }

        self.drift_detected = {
            me: False
            for me in range(self.ME)
        }

        # =========================================================
        # FedConD: regularization parameter
        #
        # λ is increased only when a drift is detected.
        # There is NO automatic decay when no drift is detected.
        # =========================================================

        self.lambda_fedcond = {
            me: 0.001
            for me in range(self.ME)
        }

        self.lambda_min = 0.001
        self.lambda_max = 0.1
        self.lambda_growth = 1.5

        # =========================================================
        # FedConD drift detector
        #
        # The original paper uses a bounded queue of size 20
        # and significance level 0.05.
        # =========================================================

        self.history_window = 5
        self.significance_level = 0.05

    def detect_drift(
            self,
            me,
            current_acc
    ):
        """
        Detect concept drift using the FedConD statistical test.

        The current performance corresponds to the evaluation of
        the current global model on the updated local trainloader.

        The current observation is tested against the historical
        queue BEFORE it is inserted into that queue.

        Parameters
        ----------
        me : int
            Model index.

        current_acc : float
            Current predictive performance.

        Returns
        -------
        bool
            True when concept drift is detected.
        """

        history = self.performance_history[me]

        current_acc = float(
            current_acc
        )

        # =========================================================
        # Warm-up
        #
        # The statistical test requires historical observations.
        # During warm-up, store the observations but do not declare
        # drift.
        # =========================================================

        if len(history) < self.history_window:
            history.append(
                current_acc
            )

            return False

        # =========================================================
        # Historical queue
        #
        # [s_1, ..., s_a]
        # =========================================================

        hist = history[
            -self.history_window:
        ]

        a = len(hist)

        # =========================================================
        # Historical mean
        #
        # s_bar
        # =========================================================

        s_bar = float(
            np.mean(hist)
        )

        # =========================================================
        # Mean including the current observation
        #
        # s_hat =
        # mean(s_1, ..., s_a, s_{a+1})
        # =========================================================

        s_hat = float(
            np.mean(
                hist + [current_acc]
            )
        )

        # =========================================================
        # Delta
        #
        # Δ_k = 1 / (a + 1)
        # =========================================================

        delta = (
                1.0 /
                (a + 1)
        )

        # =========================================================
        # Denominator of the FedConD statistic
        # =========================================================

        variance_term = (
                s_hat *
                (1.0 - s_hat) *
                delta
        )

        denominator = np.sqrt(
            max(
                variance_term,
                1e-12
            )
        )

        # =========================================================
        # FedConD test statistic
        #
        # Γ_k =
        #
        # |s_bar - s_{a+1}| - 0.5 Δ_k
        # --------------------------------
        # sqrt(s_hat(1-s_hat)Δ_k)
        #
        # The paper describes a two-sided statistical test and
        # evaluates the corresponding p-value.
        # =========================================================

        gamma_stat = (
                             abs(
                                 s_bar -
                                 current_acc
                             )
                             -
                             0.5 * delta
                     ) / denominator

        # =========================================================
        # Two-sided p-value
        # =========================================================

        p_value = 2.0 * (
                1.0 -
                norm.cdf(
                    abs(gamma_stat)
                )
        )

        # =========================================================
        # Drift decision
        # =========================================================

        drift = (
                p_value <
                self.significance_level
        )

        # =========================================================
        # Update historical queue AFTER the test
        # =========================================================

        history.append(
            current_acc
        )

        if len(history) > self.history_window:
            history.pop(0)

        return bool(drift)

    def fit(
            self,
            me,
            t,
            global_model
    ):
        """
        Perform one local FedConD training update.

        The local data are updated first through the parent
        MultiFedAvgClient. This guarantees that delayed labeling
        and data-shift mechanisms have already updated
        self.trainloader[me].

        FedConD then evaluates the received global model on this
        UPDATED trainloader before local training.

        If drift is detected, lambda is increased. If no drift is
        detected, lambda remains unchanged.
        """

        # =========================================================
        # Current local training round
        # =========================================================

        self.lt[me] = t

        # =========================================================
        # Load current global model
        # =========================================================

        set_weights(
            self.model[me],
            global_model
        )

        # =========================================================
        # Update local data BEFORE drift detection
        #
        # The parent class is responsible for delayed labeling and
        # data-shift updates.
        #
        # The resulting self.trainloader[me] is therefore the
        # correct data stream for FedConD detection.
        # =========================================================

        if t > 1:
            self.update_local_train_data(
                t,
                me
            )

        # =========================================================
        # FEDCOND DRIFT DETECTION
        #
        # Evaluate the current global model on the UPDATED
        # trainloader BEFORE local training.
        # =========================================================

        _, metrics_before = test(
            self.model[me],
            self.trainloader[me],
            self.device,
            self.client_id,
            t,
            self.args.dataset[me],
            self.n_classes[me],
            self.concept_drift_window_train[me]
        )

        current_acc = float(
            metrics_before["Accuracy"]
        )

        # =========================================================
        # Statistical drift detection
        # =========================================================

        drift = self.detect_drift(
            me,
            current_acc
        )

        self.drift_detected[me] = bool(
            drift
        )

        results_shift = (
            "DATA_SHIFT"
            if drift
            else "NO_SHIFT"
        )

        # =========================================================
        # FEDCOND LOCAL DRIFT ADAPTATION
        #
        # Increase lambda only when drift is detected.
        #
        # If there is no drift, lambda is kept unchanged.
        # =========================================================

        if drift:
            self.lambda_fedcond[me] = min(
                self.lambda_max,
                self.lambda_fedcond[me]
                * self.lambda_growth
            )

        # =========================================================
        # Global parameters for the FedConD regularization term
        # =========================================================

        global_params_torch = [
            torch.tensor(
                p,
                dtype=torch.float32,
                device=self.device
            )
            for p in global_model
        ]

        # =========================================================
        # Optimizer
        # =========================================================

        self.optimizer[me] = (
            self._get_optimizer(
                dataset_name=self.args.dataset[me],
                me=me
            )
        )

        # =========================================================
        # Local training
        #
        # IMPORTANT:
        #
        # This is the SAME updated trainloader that was evaluated
        # above.
        # =========================================================

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
            concept_drift_window=(
                self.concept_drift_window_train[me]
            ),
            global_params=global_params_torch,
            mu=self.lambda_fedcond[me]
        )

        # =========================================================
        # Return FedConD metadata
        # =========================================================

        results["me"] = me

        results["client_id"] = (
            self.client_id
        )

        results["Model size"] = (
            self.models_size[me]
        )

        results["alpha"] = (
            self.alpha_train[me]
        )

        results["Data shift"] = (
            results_shift
        )

        results["Lambda"] = (
            self.lambda_fedcond[me]
        )

        results["Drift detected"] = int(
            drift
        )

        results["Pre-train accuracy"] = (
            current_acc
        )

        # =========================================================
        # Save local loss
        # =========================================================

        self.loss_ME[me] = (
            results["train_loss"]
        )

        # =========================================================
        # Return local model update
        # =========================================================

        return (
            get_weights(
                self.model[me]
            ),
            len(
                self.trainloader[me].dataset
            ),
            results
        )