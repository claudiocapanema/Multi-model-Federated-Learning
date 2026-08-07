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

    def __init__(self, args, id, model, fold_id):
        super().__init__(args, id, model, fold_id)

        self.performance_history = {
            me: []
            for me in range(self.ME)
        }

        self.drift_detected = {
            me: False
            for me in range(self.ME)
        }

        self.lambda_fedcond = {
            me: 0.001
            for me in range(self.ME)
        }

        self.history_window = 5
        self.performance_drop_threshold = 0.05
        self.significance_level = 0.05

        self.lambda_min = 0.001
        self.lambda_max = 0.1
        self.lambda_growth = 1.5
        self.lambda_decay = 0.95

    def detect_drift(self, me, current_acc):

        history = self.performance_history[me]

        if len(history) < self.history_window:
            history.append(current_acc)
            return False

        hist = history[-self.history_window:]

        a = len(hist)

        s_bar = np.mean(hist)

        s_hat = np.mean(hist + [current_acc])

        delta = 1.0 / (a + 1)

        denom = np.sqrt(
            max(
                s_hat * (1.0 - s_hat) * delta,
                1e-12
            )
        )

        gamma = (
                        abs(s_bar - current_acc)
                        - 0.5 * delta
                ) / denom

        p_value = 2.0 * (
                1.0 - norm.cdf(abs(gamma))
        )

        drift = p_value < self.significance_level

        history.append(current_acc)

        if len(history) > self.history_window:
            history.pop(0)

        return drift

    def fit(self, me, t, global_model):

        self.lt[me] = t

        set_weights(
            self.model[me],
            global_model
        )

        if t > 1:
            self.update_local_train_data(t, me)

        #################################################
        # FEDCOND DRIFT DETECTION
        #################################################

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

        current_acc = metrics_before["Accuracy"]

        drift = self.detect_drift(
            me,
            current_acc
        )

        self.drift_detected[me] = drift

        results_shift = "DATA_SHIFT" if drift else "NO_SHIFT"

        if drift:

            self.lambda_fedcond[me] = min(
                self.lambda_max,
                self.lambda_fedcond[me]
                * self.lambda_growth
            )

        else:

            self.lambda_fedcond[me] = max(
                self.lambda_min,
                self.lambda_fedcond[me]
                * self.lambda_decay
            )

        #################################################
        # FEDCOND TRAINING
        #################################################

        global_params_torch = [
            torch.tensor(p)
            for p in global_model
        ]

        self.optimizer[me] = self._get_optimizer(
            dataset_name=self.args.dataset[me],
            me=me
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
            global_params=global_params_torch,
            mu=self.lambda_fedcond[me]
        )

        results["me"] = me
        results["client_id"] = self.client_id
        results["Model size"] = self.models_size[me]
        results["alpha"] = self.alpha_train[me]
        results["Data shift"] = results_shift
        results["Lambda"] = self.lambda_fedcond[me]
        results["Drift detected"] = int(drift)
        results["Pre-train accuracy"] = current_acc

        self.loss_ME[me] = results["train_loss"]

        return (
            get_weights(self.model[me]),
            len(self.trainloader[me].dataset),
            results
        )