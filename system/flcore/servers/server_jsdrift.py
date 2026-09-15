import copy
import sys

import numpy as np

from flwr.server.strategy.aggregate import aggregate

from flcore.clients.client_jsdrift import ClientJSDrift
from flcore.servers.server_multifedavg import MultiFedAvg


class JSDrift(MultiFedAvg):
    """JS-Drift adapted from single-model FL to Multi-model FL (MEFL).

    For every MEFL model independently, the server replaces the standard
    FedAvg sample-count weight n_k with n_k * delta_k, where

        delta_k = exp(-gamma * JSD(P_k^t, P_k^(t-1))).

    This is the core mechanism specified by the article. The model-specific
    state and aggregation are independent across ``me``.
    """

    def __init__(self, args, models, fold_id):
        super().__init__(args, models, fold_id)

        self.detector = "JS-Drift"
        self.jsdrift_gamma = float(getattr(args, "jsdrift_gamma", 0.5))
        self.jsdrift_detection_threshold = float(
            getattr(args, "jsdrift_threshold", 0.1)
        )

        self.data_shift_type = {
            me: "NO_SHIFT" for me in range(self.ME)
        }
        self.drift_clients = {
            me: 0 for me in range(self.ME)
        }
        self.drift_rate = {
            me: 0.0 for me in range(self.ME)
        }
        self.jsd_mean = {
            me: 0.0 for me in range(self.ME)
        }
        self.delta_mean = {
            me: 1.0 for me in range(self.ME)
        }

        # MultiFedAvg.train() expects the strategy to expose this
        # dictionary immediately after aggregate_fit().  The base
        # MultiFedAvg class does not initialize it in __init__, so
        # JS-Drift must initialize it explicitly.
        self.parameters_aggregated_mefl = {
            me: []
            for me in range(self.ME)
        }
        self.metrics_aggregated_mefl = {
            me: {}
            for me in range(self.ME)
        }

        self.shift_rounds = {
            me: [] for me in range(self.ME)
        }
        self.shift_ground_truth_state = {
            me: [] for me in range(self.ME)
        }
        self.shift_ground_truth_event = {
            me: [] for me in range(self.ME)
        }
        self.shift_detected = {
            me: [] for me in range(self.ME)
        }
        self.false_alarm_rounds = {
            me: [] for me in range(self.ME)
        }
        self.true_detection_round = {
            me: None for me in range(self.ME)
        }
        self.detection_delay = {
            me: -1 for me in range(self.ME)
        }
        self.previous_detector_state = {
            me: "NO_SHIFT" for me in range(self.ME)
        }
        self.detection_event = {
            me: 0 for me in range(self.ME)
        }

        # Add method-specific fields to the standard MEFL test CSV.
        self.test_metrics_names = list(self.test_metrics_names) + [
            "JSD",
            "JS-Drift coefficient",
            "Drift clients",
            "Drift rate",
            "Data shift",
            "Ground truth shift",
        ]

        self.clients = []
        for i in range(self.total_clients):
            self.clients.append(
                ClientJSDrift(
                    self.args,
                    id=i,
                    model=copy.deepcopy(self.global_model),
                    fold_id=self.fold_id
                )
            )

        if self.clients:
            for me in range(self.ME):
                if me in self.clients[0].data_shift_config:
                    self.shift_rounds[me] = self.clients[0].data_shift_config[me][
                        "data_shift_rounds"
                    ]

    def aggregate_fit(self, server_round, results, failures):
        try:
            # Defensive initialization.  This is also important if the
            # strategy object is restored/reused without a fresh __init__.
            if not hasattr(self, "parameters_aggregated_mefl"):
                self.parameters_aggregated_mefl = {
                    me: [] for me in range(self.ME)
                }

            self.metrics_aggregated_mefl = {
                me: {} for me in range(self.ME)
            }

            self.selected_clients_m = [
                [] for _ in range(self.ME)
            ]

            results_mefl = {
                me: [] for me in range(self.ME)
            }
            trained_models = []

            for parameters, num_examples, fit_res in results:
                me = int(fit_res["me"])
                client_id = int(fit_res["client_id"])

                if me not in trained_models:
                    trained_models.append(me)

                self.selected_clients_m[me].append(client_id)
                results_mefl[me].append(
                    (parameters, num_examples, fit_res)
                )

            for me in trained_models:
                model_results = results_mefl[me]

                # --------------------------------------------------
                # Article Eq. (3): a_k = n_k * delta_k / sum_j n_j*delta_j
                # --------------------------------------------------
                raw_weights = np.asarray(
                    [
                        float(num_examples)
                        * float(fit_res.get("JS-Drift coefficient", 1.0))
                        for _, num_examples, fit_res in model_results
                    ],
                    dtype=np.float64
                )

                if not np.isfinite(raw_weights).all() or raw_weights.sum() <= 0:
                    raw_weights = np.asarray(
                        [float(num_examples) for _, num_examples, _ in model_results],
                        dtype=np.float64
                    )

                normalized_weights = raw_weights / raw_weights.sum()

                weights_results = [
                    (parameters, float(weight))
                    for (parameters, _, _), weight in zip(
                        model_results,
                        normalized_weights
                    )
                ]

                if len(weights_results) == 1:
                    aggregated = weights_results[0][0]
                else:
                    aggregated = aggregate(weights_results)

                self.parameters_aggregated_mefl[me] = aggregated

                # Normal fit metrics remain sample-count weighted.
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [
                        (num_examples, fit_res)
                        for _, num_examples, fit_res in model_results
                    ]
                    metrics = self.fit_metrics_aggregation_fn(fit_metrics)
                else:
                    metrics = {}

                jsd_values = [
                    float(fit_res.get("JSD", 0.0))
                    for _, _, fit_res in model_results
                ]
                delta_values = [
                    float(fit_res.get("JS-Drift coefficient", 1.0))
                    for _, _, fit_res in model_results
                ]
                drift_values = [
                    int(fit_res.get("Drift detected", 0))
                    for _, _, fit_res in model_results
                ]

                n_clients = len(model_results)
                n_drift = sum(drift_values)
                drift_rate = n_drift / n_clients if n_clients else 0.0

                self.jsd_mean[me] = float(np.mean(jsd_values)) if jsd_values else 0.0
                self.delta_mean[me] = float(np.mean(delta_values)) if delta_values else 1.0
                self.drift_clients[me] = n_drift
                self.drift_rate[me] = drift_rate

                self.data_shift_type[me] = (
                    "DATA_SHIFT" if drift_rate >= 0.4 else "NO_SHIFT"
                )

                ground_truth_state = int(
                    any(server_round >= r for r in self.shift_rounds[me])
                )
                ground_truth_event = int(
                    server_round in self.shift_rounds[me]
                )

                self.shift_ground_truth_state[me].append(ground_truth_state)
                self.shift_ground_truth_event[me].append(ground_truth_event)
                self.shift_detected[me].append(
                    int(self.data_shift_type[me] == "DATA_SHIFT")
                )

                current_state = self.data_shift_type[me]
                self.detection_event[me] = int(
                    self.previous_detector_state[me] == "NO_SHIFT"
                    and current_state == "DATA_SHIFT"
                )

                if self.detection_event[me]:
                    if ground_truth_event and self.true_detection_round[me] is None:
                        self.true_detection_round[me] = server_round
                        self.detection_delay[me] = 0
                    elif self.true_detection_round[me] is None:
                        self.false_alarm_rounds[me].append(server_round)

                self.previous_detector_state[me] = current_state

                metrics.update({
                    "JSD": self.jsd_mean[me],
                    "JS-Drift coefficient": self.delta_mean[me],
                    "Drift clients": self.drift_clients[me],
                    "Drift rate": self.drift_rate[me],
                    "Data shift": self.data_shift_type[me],
                    "Ground truth shift": ground_truth_state,
                })

                self.metrics_aggregated_mefl[me] = metrics

            # Models not selected in this round keep their last global update.
            self._save_shift_detection_metrics(server_round)
            self._save_shift_detection_curve(server_round)

            print(
                f"[JS-Drift] round={server_round} "
                f"metrics={self.metrics_aggregated_mefl}"
            )

            return self.parameters_aggregated_mefl, self.metrics_aggregated_mefl

        except Exception as e:
            print("JS-Drift aggregate_fit error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )
            raise

    def add_metrics(self, server_round, metrics_aggregated, me):
        # The base class stores aggregated test metrics in the same structure
        # used by _save_results. We add the JS-Drift fields there.
        metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
        metrics_aggregated[me]["# training clients"] = self.n_trained_clients
        metrics_aggregated[me]["training clients and models"] = self.selected_clients_m[me]
        metrics_aggregated[me]["Fold ID"] = self.fold_id
        metrics_aggregated[me]["JSD"] = self.jsd_mean[me]
        metrics_aggregated[me]["JS-Drift coefficient"] = self.delta_mean[me]
        metrics_aggregated[me]["Drift clients"] = self.drift_clients[me]
        metrics_aggregated[me]["Drift rate"] = self.drift_rate[me]
        metrics_aggregated[me]["Data shift"] = self.data_shift_type[me]
        metrics_aggregated[me]["Ground truth shift"] = (
            self.shift_ground_truth_state[me][-1]
            if self.shift_ground_truth_state[me]
            else 0
        )

        for metric, value in metrics_aggregated[me].items():
            if metric not in self.results_test_metrics[me]:
                self.results_test_metrics[me][metric] = []
            self.results_test_metrics[me][metric].append(value)

    def _init_shift_detection_files(self):
        result_path = self.get_result_path("test")
        self._write_header(
            result_path + f"shift_detection_metrics_{self.strategy_name}.csv",
            [
                "Detector", "Dataset", "Fold ID", "Round", "Model",
                "Shift Type", "Shift Configuration", "Precision", "Recall",
                "F1", "Detection Delay", "False Alarms", "First Detection Round",
                "Shift Round",
            ],
            mode="w",
        )
        self._write_header(
            result_path + f"shift_detection_curve_{self.strategy_name}.csv",
            [
                "Detector", "Dataset", "Fold ID", "Round", "Model",
                "Ground Truth", "Detection Event", "Detector State",
                "Drift Clients", "Drift Rate",
            ],
            mode="w",
        )

    def _save_shift_detection_metrics(self, server_round):
        result_path = self.get_result_path("test")
        file_path = result_path + f"shift_detection_metrics_{self.strategy_name}.csv"

        for me in range(self.ME):
            tp = 1 if self.true_detection_round[me] is not None else 0
            fp = len(self.false_alarm_rounds[me])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = float(tp)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0 else 0.0
            )

            shift_round = self.shift_rounds[me][0] if self.shift_rounds[me] else -1
            row = [[
                self.detector,
                self.dataset[me],
                self.fold_id,
                server_round,
                me,
                self._shift_type(),
                self._shift_configuration(),
                precision,
                recall,
                f1,
                self.detection_delay[me],
                len(self.false_alarm_rounds[me]),
                self.true_detection_round[me] if self.true_detection_round[me] is not None else -1,
                shift_round,
            ]]
            self._write_rows(file_path, row)

    def _save_shift_detection_curve(self, server_round):
        result_path = self.get_result_path("test")
        file_path = result_path + f"shift_detection_curve_{self.strategy_name}.csv"

        for me in range(self.ME):
            state = self.shift_ground_truth_state[me][-1] if self.shift_ground_truth_state[me] else 0
            event = self.detection_event[me]
            gt_event = self.shift_ground_truth_event[me][-1] if self.shift_ground_truth_event[me] else 0
            row = [[
                self.detector,
                self.dataset[me],
                self.fold_id,
                server_round,
                me,
                gt_event,
                event,
                self.data_shift_type[me],
                self.drift_clients[me],
                self.drift_rate[me],
            ]]
            self._write_rows(file_path, row)

    def _shift_type(self):
        eid = self.experiment_id.lower()
        if "combined_shift" in eid:
            return "Combined"
        if "label_shift" in eid:
            return "Label"
        if "concept_drift" in eid:
            return "Concept"
        return "Unknown"

    def _shift_configuration(self):
        return (
            self.experiment_id
            .replace("label_shift#", "")
            .replace("concept_drift#", "")
            .replace("combined_shift#", "")
            .replace("_sudden", "")
            .replace("_gradual", "")
        )