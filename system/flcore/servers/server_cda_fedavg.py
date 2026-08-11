from flcore.servers.server_multifedavg import MultiFedAvg
from flcore.clients.client_cda_fedavg import ClientCDAFedAvg
from flwr.server.strategy.aggregate import aggregate

import copy
import os
import sys


class CDAFedAvg(MultiFedAvg):
    """
    CDA-FedAvg server adapted to the existing MEFL/MultiFedAvg server.

    The server keeps the existing per-model weighted aggregation. Client-side
    CDA-FedAvg performs the actual drift detection. The server aggregates the
    local drift signals to obtain MEFL-level detection statistics and writes
    the same shift-detection CSV structure used by the existing experiments.

    Unlike the original CDA-FedAvg paper, the surrounding MEFL framework is
    synchronous. The original method is asynchronous; here we preserve the
    detector/adaptation logic while keeping MultiFedAvg's orchestration.
    """

    def __init__(self, args, models, fold_id):
        super().__init__(args, models, fold_id)

        self.train_metrics_names = [
            "Accuracy",
            "Balanced accuracy",
            "Loss",
            "Round (t)",
            "Fraction fit",
            "# training clients",
            "training clients and models",
            "Model size",
            "Alpha",
            "Drift clients",
            "Drift rate",
            "Data shift",
        ]

        self.test_metrics_names = [
            "Accuracy",
            "Balanced accuracy",
            "Loss",
            "Round (t)",
            "Fraction fit",
            "# training clients",
            "training clients and models",
            "Model size",
            "Fold ID",
            "Alpha",
            "Drift clients",
            "Drift rate",
            "Data shift",
            "Ground truth shift",
            "Detection delay",
            "False alarm",
            "Detection rate",
        ]

        self.results_test_metrics = {
            me: {metric: [] for metric in self.test_metrics_names}
            for me in range(self.ME)
        }

        self.detector = "CDA-FedAvg"
        self.dataset = self.args.dataset

        self.shift_type = (
            "Label"
            if "label_shift" in self.args.experiment_id
            else "Concept"
        )

        self.shift_configuration = (
            self.args.experiment_id
            .replace("label_shift#", "")
            .replace("concept_drift#", "")
            .replace("_sudden", "")
        )

        # In the MEFL evaluation, any participating client reporting a local
        # drift is considered evidence of a global/local data-shift event.
        # Change to >0.0 only if a stricter client-fraction criterion is
        # desired for a particular experiment.
        self.global_drift_fraction = float(
            getattr(args, "cda_global_drift_fraction", 0.0)
        )

        self._init_detection_state()

    # ------------------------------------------------------------------
    # Client creation
    # ------------------------------------------------------------------
    def set_clients(self):
        self._init_detection_state()

        self.clients = []

        for i in range(self.total_clients):
            client = ClientCDAFedAvg(
                self.args,
                id=i,
                model=copy.deepcopy(self.global_model),
                fold_id=self.fold_id,
            )
            self.clients.append(client)

        if len(self.clients) > 0:
            for me in range(self.ME):
                if me in self.clients[0].data_shift_config:
                    self.shift_rounds[me] = (
                        self.clients[0]
                        .data_shift_config[me]["data_shift_rounds"]
                    )

    def _init_detection_state(self):
        self.data_shift_type = {
            me: "NO_SHIFT" for me in range(self.ME)
        }

        self.drift_clients = {
            me: 0 for me in range(self.ME)
        }

        self.drift_rate = {
            me: 0.0 for me in range(self.ME)
        }

        self.shift_rounds = {
            me: [] for me in range(self.ME)
        }

        self.shift_detected = {
            me: [] for me in range(self.ME)
        }

        self.shift_ground_truth = {
            me: [] for me in range(self.ME)
        }

        self.drift_rate_history = {
            me: [] for me in range(self.ME)
        }

        self.shift_ground_truth_state = {
            me: [] for me in range(self.ME)
        }

        self.shift_ground_truth_event = {
            me: [] for me in range(self.ME)
        }

        self.previous_detector_state = {
            me: "NO_SHIFT" for me in range(self.ME)
        }

        self.detection_event = {
            me: 0 for me in range(self.ME)
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

        self.clients = []

    # ------------------------------------------------------------------
    # Fit aggregation
    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round: int, results, failures):
        try:
            self.selected_clients_m = [[] for _ in range(self.ME)]

            trained_models = []
            results_mefl = {me: [] for me in range(self.ME)}

            for parameter, num_examples, result in results:
                me = result["me"]

                if me not in trained_models:
                    trained_models.append(me)

                self.selected_clients_m[me].append(result["client_id"])
                results_mefl[me].append(
                    (parameter, num_examples, result)
                )

            aggregated_ndarrays_mefl = {
                me: None for me in range(self.ME)
            }

            for me in trained_models:
                weights_results = [
                    (parameters, num_examples)
                    for parameters, num_examples, _ in results_mefl[me]
                ]

                if len(weights_results) > 1:
                    aggregated_ndarrays_mefl[me] = aggregate(
                        weights_results
                    )
                elif len(weights_results) == 1:
                    aggregated_ndarrays_mefl[me] = results_mefl[me][0][0]

            for me in trained_models:
                self.parameters_aggregated_mefl[me] = (
                    aggregated_ndarrays_mefl[me]
                )

            metrics_aggregated_mefl = {
                me: [] for me in range(self.ME)
            }

            for me in trained_models:
                model_results = results_mefl[me]

                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [
                        (num_examples, metrics)
                        for _, num_examples, metrics in model_results
                    ]
                    metrics_aggregated_mefl[me] = (
                        self.fit_metrics_aggregation_fn(fit_metrics)
                    )
                else:
                    metrics_aggregated_mefl[me] = {}

                drift_flags = [
                    int(metrics.get("Drift detected", 0))
                    for _, _, metrics in model_results
                ]

                n_drift = int(sum(drift_flags))
                n_clients = len(model_results)

                drift_rate = (
                    n_drift / n_clients if n_clients > 0 else 0.0
                )

                self.drift_clients[me] = n_drift
                self.drift_rate[me] = drift_rate
                self.drift_rate_history[me].append(drift_rate)

                self.data_shift_type[me] = (
                    "DATA_SHIFT"
                    if drift_rate >= self.global_drift_fraction
                    and n_drift > 0
                    else "NO_SHIFT"
                )

                metrics_aggregated_mefl[me]["Drift clients"] = n_drift
                metrics_aggregated_mefl[me]["Drift rate"] = drift_rate
                metrics_aggregated_mefl[me]["Data shift"] = (
                    self.data_shift_type[me]
                )

                # ------------------------------------------------------
                # Detection evaluation
                # ------------------------------------------------------
                ground_truth_state = int(
                    any(
                        server_round >= r
                        for r in self.shift_rounds[me]
                    )
                )

                ground_truth_event = int(
                    server_round in self.shift_rounds[me]
                )

                current_state = self.data_shift_type[me]

                self.detection_event[me] = int(
                    self.previous_detector_state[me] == "NO_SHIFT"
                    and current_state == "DATA_SHIFT"
                )

                self.previous_detector_state[me] = current_state

                self.shift_ground_truth_state[me].append(
                    ground_truth_state
                )

                self.shift_ground_truth_event[me].append(
                    ground_truth_event
                )

                self.shift_detected[me].append(
                    self.detection_event[me]
                )

                if self.detection_event[me]:
                    if ground_truth_event == 0:
                        self.false_alarm_rounds[me].append(
                            server_round
                        )
                    elif self.true_detection_round[me] is None:
                        self.true_detection_round[me] = server_round
                        self.detection_delay[me] = (
                            server_round
                            - self.shift_rounds[me][0]
                        )

                metrics_aggregated_mefl[me]["Ground truth shift"] = (
                    ground_truth_state
                )
                metrics_aggregated_mefl[me]["Detection delay"] = (
                    self.detection_delay[me]
                )
                metrics_aggregated_mefl[me]["False alarm"] = len(
                    self.false_alarm_rounds[me]
                )
                metrics_aggregated_mefl[me]["Detection rate"] = (
                    drift_rate
                )

            self._save_data_metrics()
            self._save_shift_detection_metrics(server_round)
            self._save_shift_detection_curve(server_round)

            self.metrics_aggregated_mefl = metrics_aggregated_mefl

            print(
                f"Round {server_round} CDA-FedAvg metrics: "
                f"{self.metrics_aggregated_mefl}"
            )

            return (
                self.parameters_aggregated_mefl,
                metrics_aggregated_mefl,
            )

        except Exception as e:
            print("CDA-FedAvg aggregate_fit error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )
            raise

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def add_metrics(self, server_round, metrics_aggregated, me):
        try:
            metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
            metrics_aggregated[me]["# training clients"] = (
                self.n_trained_clients
            )
            metrics_aggregated[me]["training clients and models"] = (
                self.selected_clients_m[me]
            )
            metrics_aggregated[me]["Fold ID"] = self.fold_id
            metrics_aggregated[me]["Data shift"] = (
                self.data_shift_type[me]
            )
            metrics_aggregated[me]["Drift clients"] = (
                self.drift_clients[me]
            )
            metrics_aggregated[me]["Drift rate"] = (
                self.drift_rate[me]
            )

            for metric in metrics_aggregated[me]:
                if metric in self.results_test_metrics[me]:
                    self.results_test_metrics[me][metric].append(
                        metrics_aggregated[me][metric]
                    )

        except Exception as e:
            print("CDA-FedAvg add_metrics error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )
            raise

    def _get_results(self, train_test, mode, me):
        try:
            algo = self.dataset[me] + "_" + self.strategy_name

            result_path = self.get_result_path(train_test)

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            file_path = result_path + "{}.csv".format(algo)

            if train_test == "test":
                header = self.test_metrics_names
                list_of_metrics = [
                    self.results_test_metrics[me][metric]
                    for metric in self.results_test_metrics[me]
                ]

                length = (
                    len(list_of_metrics[0])
                    if list_of_metrics
                    else 0
                )

                data = []
                for i in range(length):
                    data.append(
                        [
                            metric_values[i]
                            for metric_values in list_of_metrics
                        ]
                    )

            else:
                if mode == "":
                    header = self.train_metrics_names
                    list_of_metrics = [
                        self.results_train_metrics[me][metric]
                        for metric in self.results_train_metrics[me]
                    ]

                    length = (
                        len(list_of_metrics[0])
                        if list_of_metrics
                        else 0
                    )

                    data = []
                    for i in range(length):
                        data.append(
                            [
                                metric_values[i]
                                if len(metric_values) > i
                                else 0
                                for metric_values in list_of_metrics
                            ]
                        )

            return file_path, header, data

        except Exception as e:
            print("CDA-FedAvg _get_results error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )
            raise

    def _save_shift_detection_metrics(self, server_round):
        try:
            from sklearn.metrics import (
                precision_score,
                recall_score,
                f1_score,
            )

            result_path = self.get_result_path("test")
            file_path = (
                result_path
                + f"shift_detection_metrics_{self.strategy_name}.csv"
            )

            for me in range(self.ME):
                y_true = self.shift_ground_truth_event[me]
                y_pred = self.shift_detected[me]

                if not y_true:
                    precision = recall = f1 = 0.0
                else:
                    precision = precision_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                    recall = recall_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                    f1 = f1_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )

                shift_round = (
                    self.shift_rounds[me][0]
                    if self.shift_rounds[me]
                    else -1
                )

                row = [[
                    self.detector,
                    self.dataset[me],
                    self.fold_id,
                    server_round,
                    me,
                    self.shift_type,
                    self.shift_configuration,
                    precision,
                    recall,
                    f1,
                    self.detection_delay[me],
                    len(self.false_alarm_rounds[me]),
                    (
                        self.true_detection_round[me]
                        if self.true_detection_round[me] is not None
                        else -1
                    ),
                    shift_round,
                ]]

                self._write_rows(file_path, row)

        except Exception as e:
            print("CDA-FedAvg _save_shift_detection_metrics error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )
            raise

    def _init_shift_detection_files(self):
        result_path = self.get_result_path("test")

        metrics_file = (
            result_path
            + f"shift_detection_metrics_{self.strategy_name}.csv"
        )

        curve_file = (
            result_path
            + f"shift_detection_curve_{self.strategy_name}.csv"
        )

        self._write_header(
            metrics_file,
            [
                "Detector",
                "Dataset",
                "Fold ID",
                "Round",
                "Model",
                "Shift Type",
                "Shift Configuration",
                "Precision",
                "Recall",
                "F1",
                "Detection Delay",
                "False Alarms",
                "First Detection Round",
                "Shift Round",
            ],
            mode="w",
        )

        self._write_header(
            curve_file,
            [
                "Detector",
                "Dataset",
                "Fold ID",
                "Round",
                "Model",
                "Ground Truth",
                "Detection Event",
                "Detector State",
                "Drift Clients",
                "Drift Rate",
            ],
            mode="w",
        )

    def _save_shift_detection_curve(self, server_round):
        try:
            result_path = self.get_result_path("test")

            file_path = (
                result_path
                + f"shift_detection_curve_{self.strategy_name}.csv"
            )

            for me in range(self.ME):
                if not self.shift_ground_truth_state[me]:
                    continue

                row = [[
                    self.detector,
                    self.dataset[me],
                    self.fold_id,
                    server_round,
                    me,
                    self.shift_ground_truth_state[me][-1],
                    self.detection_event[me],
                    self.data_shift_type[me],
                    self.drift_clients[me],
                    self.drift_rate[me],
                ]]

                self._write_rows(file_path, row)

        except Exception as e:
            print("CDA-FedAvg _save_shift_detection_curve error")
            print(
                "Error on line {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )
            raise