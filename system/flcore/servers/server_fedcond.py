from flcore.servers.server_multifedavg import MultiFedAvg
from flcore.clients.client_fedcond import ClientFedConD
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
import copy
import os
import sys
import numpy as np


class FedConD(MultiFedAvg):

    def __init__(
            self,
            args,
            models,
            fold_id
    ):
        try:

            # ========================================================
            # Initialize parent MultiFedAvg server
            # ========================================================

            super().__init__(
                args,
                models,
                fold_id
            )

            self.detector = "FedConD"

            self.shift_type = (
                "Label" if "label_shift" in args.experiment_id else "Concept"
            )
            self.shift_configuration = (
                args.experiment_id
                .replace("label_shift#", "")
                .replace("concept_drift#", "")
                .replace("_sudden", "")
            )

            # ========================================================
            # FedConD parameters
            # ========================================================

            self.gamma = getattr(
                self.args,
                "fedcond_gamma",
                self.fraction_fit
            )

            self.gamma = min(
                max(
                    float(self.gamma),
                    0.0
                ),
                1.0
            )

            # ========================================================
            # Model-specific client participation counters
            #
            # client_update_count[me][cid] =
            # number of previous rounds in which client cid
            # trained model me.
            # ========================================================

            self.client_update_count = {
                me: {
                    cid: 0
                    for cid in range(
                        self.total_clients
                    )
                }
                for me in range(self.ME)
            }

            # ========================================================
            # Aggregated parameters and metrics
            # ========================================================

            self.parameters_aggregated_mefl = {
                me: []
                for me in range(self.ME)
            }

            self.metrics_aggregated_mefl = {
                me: {}
                for me in range(self.ME)
            }

            # ========================================================
            # FedConD detection state
            # ========================================================

            self.data_shift_type = {
                me: "NO_SHIFT"
                for me in range(self.ME)
            }

            self.drift_clients = {
                me: 0
                for me in range(self.ME)
            }

            self.drift_rate = {
                me: 0.0
                for me in range(self.ME)
            }

            self.shift_rounds = {
                me: []
                for me in range(self.ME)
            }

            self.shift_detected = {
                me: []
                for me in range(self.ME)
            }

            self.shift_ground_truth = {
                me: []
                for me in range(self.ME)
            }

            self.drift_rate_history = {
                me: []
                for me in range(self.ME)
            }

            self.shift_ground_truth_state = {
                me: []
                for me in range(self.ME)
            }

            self.shift_ground_truth_event = {
                me: []
                for me in range(self.ME)
            }

            # ========================================================
            # Detection-event state
            # ========================================================

            self.previous_detector_state = {
                me: "NO_SHIFT"
                for me in range(self.ME)
            }

            self.detection_event = {
                me: 0
                for me in range(self.ME)
            }

            self.false_alarm_rounds = {
                me: []
                for me in range(self.ME)
            }

            self.true_detection_round = {
                me: None
                for me in range(self.ME)
            }

            self.detection_delay = {
                me: -1
                for me in range(self.ME)
            }

            # ========================================================
            # Create clients
            # ========================================================

            self.clients = []

            for cid in range(
                    self.total_clients
            ):
                client = ClientFedConD(
                    self.args,
                    id=cid,
                    model=copy.deepcopy(
                        self.global_model
                    ),
                    fold_id=self.fold_id
                )

                self.clients.append(
                    client
                )

            # ========================================================
            # Obtain ground-truth shift rounds from clients
            # ========================================================

            if len(self.clients) > 0:

                for me in range(self.ME):

                    if (
                            me in
                            self.clients[0].data_shift_config
                    ):
                        self.shift_rounds[me] = (
                            self.clients[0]
                            .data_shift_config[me]
                            ["data_shift_rounds"]
                        )

            self.test_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                       "# training clients", "training clients and models", "Model size", "Alpha",
                                       "Fold ID", "Data shift", "Drift clients", "Drift rate", "Ground truth shift"]

        except Exception as e:

            print(
                "__init__ FedConD server error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            raise

    def set_clients(self):

        self.data_shift_type = {
            me: "NO_SHIFT"
            for me in range(self.ME)
        }

        self.drift_clients = {
            me: 0
            for me in range(self.ME)
        }

        self.drift_rate = {
            me: 0.0
            for me in range(self.ME)
        }

        self.shift_rounds = {
            me: []
            for me in range(self.ME)
        }

        # Histórico (usado para métricas)
        self.shift_detected = {
            me: []
            for me in range(self.ME)
        }

        self.shift_ground_truth = {
            me: []
            for me in range(self.ME)
        }

        self.drift_rate_history = {
            me: []
            for me in range(self.ME)
        }

        self.shift_ground_truth_state = {
            me: []
            for me in range(self.ME)
        }

        self.shift_ground_truth_event = {
            me: []
            for me in range(self.ME)
        }

        # ===============================
        # Novos atributos
        # ===============================

        # Estado do detector na rodada anterior
        self.previous_detector_state = {
            me: "NO_SHIFT"
            for me in range(self.ME)
        }

        # Evento ocorrido na rodada corrente
        self.detection_event = {
            me: 0
            for me in range(self.ME)
        }

        # Rodadas de falso alarme
        self.false_alarm_rounds = {
            me: []
            for me in range(self.ME)
        }

        # Primeira detecção correta
        self.true_detection_round = {
            me: None
            for me in range(self.ME)
        }

        self.detection_delay = {
            me: -1
            for me in range(self.ME)
        }

        self.clients = []

        for i in range(self.total_clients):
            client = ClientFedConD(
                self.args,
                id=i,
                model=copy.deepcopy(
                    self.global_model
                ),
                fold_id=self.fold_id
            )

            self.clients.append(client)

        if len(self.clients) > 0:

            for me in range(self.ME):

                if me in self.clients[0].data_shift_config:
                    self.shift_rounds[me] = (
                        self.clients[0]
                        .data_shift_config[me]
                        ["data_shift_rounds"]
                    )

    def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
    ):
        """
        Aggregate FedConD local updates independently for each model.

        The current MultiFedAvg implementation is synchronous:
        selected clients perform local training and the server waits
        for the complete set of updates before aggregation.

        FedConD-specific information generated by the clients is
        collected here for drift-detection evaluation.
        """

        try:

            print(
                f"FedConD aggregate_fit - round {server_round}"
            )

            # ========================================================
            # Initialize aggregation structures
            # ========================================================

            self.selected_clients_m = [
                []
                for _ in range(self.ME)
            ]

            results_mefl = {
                me: []
                for me in range(self.ME)
            }

            aggregated_ndarrays_mefl = {
                me: None
                for me in range(self.ME)
            }

            # IMPORTANT:
            # This attribute must exist BEFORE any method that may
            # access it is called.
            self.metrics_aggregated_mefl = {
                me: {}
                for me in range(self.ME)
            }

            trained_models = []

            # ========================================================
            # Organize client results by model
            # ========================================================

            for (
                    parameters,
                    num_examples,
                    fit_res
            ) in results:

                me = int(
                    fit_res["me"]
                )

                client_id = int(
                    fit_res["client_id"]
                )

                if me not in trained_models:
                    trained_models.append(
                        me
                    )

                self.selected_clients_m[me].append(
                    client_id
                )

                results_mefl[me].append(
                    (
                        parameters,
                        num_examples,
                        fit_res
                    )
                )

            # ========================================================
            # Aggregate each trained model
            # ========================================================

            for me in trained_models:

                model_results = (
                    results_mefl[me]
                )

                # ----------------------------------------------------
                # Weighted FedAvg
                # ----------------------------------------------------

                weights_results = [
                    (
                        parameters,
                        num_examples
                    )
                    for (
                        parameters,
                        num_examples,
                        _
                    ) in model_results
                ]

                if len(weights_results) == 1:

                    aggregated_ndarrays_mefl[me] = (
                        weights_results[0][0]
                    )

                elif len(weights_results) > 1:

                    aggregated_ndarrays_mefl[me] = (
                        aggregate(
                            weights_results
                        )
                    )

                # ----------------------------------------------------
                # Update global parameters for this model
                # ----------------------------------------------------

                self.parameters_aggregated_mefl[me] = (
                    aggregated_ndarrays_mefl[me]
                )

                # ====================================================
                # Initialize model metrics
                # ====================================================

                metrics = {}

                # ----------------------------------------------------
                # Aggregate client metrics if configured
                # ----------------------------------------------------

                if self.fit_metrics_aggregation_fn:

                    fit_metrics = [
                        (
                            num_examples,
                            fit_res
                        )
                        for (
                            _,
                            num_examples,
                            fit_res
                        ) in model_results
                    ]

                    aggregated_client_metrics = (
                        self.fit_metrics_aggregation_fn(
                            fit_metrics
                        )
                    )

                    if aggregated_client_metrics:
                        metrics.update(
                            aggregated_client_metrics
                        )

                # ====================================================
                # FedConD drift statistics
                # ====================================================

                n_clients = len(
                    model_results
                )

                n_drift = sum(
                    int(
                        fit_res.get(
                            "Drift detected",
                            0
                        )
                    )
                    for (
                        _,
                        _,
                        fit_res
                    ) in model_results
                )

                drift_rate = (
                    n_drift / n_clients
                    if n_clients > 0
                    else 0.0
                )

                self.drift_clients[me] = (
                    n_drift
                )

                self.drift_rate[me] = (
                    drift_rate
                )

                self.drift_rate_history[me].append(
                    drift_rate
                )

                # ====================================================
                # Experimental global shift state
                # ====================================================

                self.data_shift_type[me] = (
                    "DATA_SHIFT"
                    if drift_rate >= 0.4
                    else "NO_SHIFT"
                )

                metrics[
                    "Drift clients"
                ] = n_drift

                metrics[
                    "Drift rate"
                ] = drift_rate

                metrics[
                    "Data shift"
                ] = self.data_shift_type[me]

                # ====================================================
                # Ground-truth shift state
                # ====================================================

                ground_truth_state = int(
                    any(
                        server_round >= r
                        for r in self.shift_rounds[me]
                    )
                )

                ground_truth_event = int(
                    server_round in
                    self.shift_rounds[me]
                )

                current_state = (
                    self.data_shift_type[me]
                )

                # ====================================================
                # Detection event
                # ====================================================

                self.detection_event[me] = int(
                    self.previous_detector_state[me]
                    == "NO_SHIFT"
                    and
                    current_state
                    == "DATA_SHIFT"
                )

                self.previous_detector_state[me] = (
                    current_state
                )

                # ====================================================
                # Save detection histories
                # ====================================================

                self.shift_ground_truth_state[me].append(
                    ground_truth_state
                )

                self.shift_ground_truth_event[me].append(
                    ground_truth_event
                )

                self.shift_detected[me].append(
                    self.detection_event[me]
                )

                # ====================================================
                # Detection delay / false alarms
                # ====================================================

                if self.detection_event[me]:

                    if len(
                            self.shift_rounds[me]
                    ) > 0:

                        shift_round = min(
                            self.shift_rounds[me]
                        )

                        if (
                                server_round <
                                shift_round
                        ):

                            self.false_alarm_rounds[
                                me
                            ].append(
                                server_round
                            )

                        elif (
                                self.true_detection_round[me]
                                is None
                        ):

                            self.true_detection_round[
                                me
                            ] = server_round

                            self.detection_delay[
                                me
                            ] = (
                                    server_round -
                                    shift_round
                            )

                # ====================================================
                # Detection metrics
                # ====================================================

                metrics[
                    "Ground truth shift"
                ] = ground_truth_state

                metrics[
                    "Detection delay"
                ] = self.detection_delay[me]

                metrics[
                    "False alarm"
                ] = len(
                    self.false_alarm_rounds[me]
                )

                metrics[
                    "Detection rate"
                ] = (
                    1.0
                    if (
                            self.true_detection_round[me]
                            is not None
                    )
                    else 0.0
                )

                # ----------------------------------------------------
                # Save metrics for this model
                # ----------------------------------------------------

                self.metrics_aggregated_mefl[me] = (
                    metrics
                )

            # ========================================================
            # IMPORTANT:
            #
            # Save self.metrics_aggregated_mefl BEFORE calling any
            # method that can indirectly access it.
            # ========================================================

            self._save_data_metrics()

            self._save_shift_detection_metrics(
                server_round
            )

            self._save_shift_detection_curve(
                server_round
            )

            print(
                f"round {server_round} aggregated metrics: "
                f"{self.metrics_aggregated_mefl}"
            )

            return (
                self.parameters_aggregated_mefl,
                self.metrics_aggregated_mefl
            )

        except Exception as e:

            print(
                "aggregate_fit error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            raise

    def add_metrics(
            self,
            server_round,
            metrics_aggregated,
            me
    ):
        try:

            print(
                "adicionar metricas"
            )

            # ========================================================
            # MultiFedAvg metadata
            # ========================================================

            metrics_aggregated[me][
                "Fraction fit"
            ] = self.fraction_fit

            metrics_aggregated[me][
                "# training clients"
            ] = self.n_trained_clients

            metrics_aggregated[me][
                "training clients and models"
            ] = self.selected_clients_m[me]

            metrics_aggregated[me][
                "Fold ID"
            ] = self.fold_id

            # ========================================================
            # FedConD drift metadata
            # ========================================================

            metrics_aggregated[me][
                "Data shift"
            ] = self.data_shift_type[me]

            metrics_aggregated[me][
                "Drift clients"
            ] = self.drift_clients[me]

            metrics_aggregated[me][
                "Drift rate"
            ] = self.drift_rate[me]

            metrics_aggregated[me][
                "Ground truth shift"
            ] = (
                self.shift_ground_truth_state[me][-1]
                if len(
                    self.shift_ground_truth_state[me]
                ) > 0
                else 0
            )

            # ========================================================
            # Store only metrics that have a corresponding column
            # ========================================================

            for metric, value in (
                    metrics_aggregated[me].items()
            ):

                if metric not in (
                        self.results_test_metrics[me]
                ):
                    self.results_test_metrics[me][
                        metric
                    ] = []

                self.results_test_metrics[me][
                    metric
                ].append(
                    value
                )

        except Exception as e:

            print(
                "add_metrics error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            raise

    def select_clients(self, t):
        """
        Select clients for each model using model-specific
        participation counts.

        Each client can train at most one model in a round.

        The participation counter is defined as:

            client_update_count[me][cid]

        and represents the number of rounds in which client `cid`
        has previously trained model `me`.
        """

        try:

            # ========================================================
            # Number of clients participating in the round
            # ========================================================

            num_selected = max(
                1,
                int(
                    np.ceil(
                        self.gamma *
                        self.total_clients
                    )
                )
            )

            num_selected = min(
                num_selected,
                self.total_clients
            )

            # ========================================================
            # Number of clients assigned to each model
            # ========================================================

            base_clients_per_model = (
                    num_selected // self.ME
            )

            remainder = (
                    num_selected % self.ME
            )

            clients_per_model = [
                base_clients_per_model +
                (1 if me < remainder else 0)
                for me in range(self.ME)
            ]

            # ========================================================
            # Candidates for each model
            #
            # Lower client-model participation count has priority.
            # ========================================================

            candidates_m = {}

            for me in range(self.ME):
                candidates_m[me] = sorted(
                    range(self.total_clients),
                    key=lambda cid: (
                        self.client_update_count[me][cid],
                        cid
                    )
                )

            # ========================================================
            # Final assignment
            # ========================================================

            selected_clients_m = [
                []
                for _ in range(self.ME)
            ]

            assigned_clients = set()

            # ========================================================
            # First pass:
            # assign the least-used available client for each model.
            #
            # This guarantees that a client is assigned to at most
            # one model in the current round.
            # ========================================================

            for me in range(self.ME):

                target = clients_per_model[me]

                for cid in candidates_m[me]:

                    if cid in assigned_clients:
                        continue

                    selected_clients_m[me].append(
                        cid
                    )

                    assigned_clients.add(
                        cid
                    )

                    if len(
                            selected_clients_m[me]
                    ) >= target:
                        break

            # ========================================================
            # Sanity check
            # ========================================================

            total_assigned = sum(
                len(clients)
                for clients in selected_clients_m
            )

            if total_assigned != num_selected:
                raise RuntimeError(
                    "FedConD client assignment produced "
                    f"{total_assigned} clients, expected "
                    f"{num_selected}."
                )

            # ========================================================
            # Update model-specific participation counters
            #
            # IMPORTANT:
            # This happens only AFTER the final assignment.
            # ========================================================

            for me in range(self.ME):

                for cid in selected_clients_m[me]:
                    self.client_update_count[me][cid] += 1

            # ========================================================
            # Store selection information
            # ========================================================

            self.selected_clients_m = [
                np.array(
                    clients,
                    dtype=int
                )
                for clients in selected_clients_m
            ]

            self.n_trained_clients = (
                total_assigned
            )

            # ========================================================
            # Debug information
            # ========================================================

            print(
                f"[FedConD] Round {t}: "
                f"selected {total_assigned}/"
                f"{self.total_clients} clients"
            )

            for me in range(self.ME):
                clients = (
                    self.selected_clients_m[me]
                )

                print(
                    f"[FedConD] Round {t} "
                    f"Model {me}: "
                    f"clients={clients.tolist()}"
                )

                print(
                    f"[FedConD] Round {t} "
                    f"Model {me}: "
                    f"participation={{"
                    + ", ".join(
                        f"{int(cid)}:"
                        f"{self.client_update_count[me][int(cid)]}"
                        for cid in clients
                    )
                    + "}"
                )

            return self.selected_clients_m

        except Exception as e:

            print(
                "select_clients error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
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

            print("arquivo nome: ", file_path)

            if train_test == 'test':

                header = self.test_metrics_names
                # print(self.rs_test_acc[me])
                # print(self.rs_test_auc[me])
                # print(self.rs_train_loss[me])
                list_of_metrics = []
                for metric in self.results_test_metrics[me]:
                    # print(me, len(self.results_test_metrics[me][metric]))
                    length = len(self.results_test_metrics[me][metric])
                    list_of_metrics.append(self.results_test_metrics[me][metric])

                data = []
                for i in range(length):
                    row = []
                    for j in range(len(list_of_metrics)):
                        row.append(list_of_metrics[j][i])

                    data.append(row)

            else:
                if mode == '':
                    header = self.train_metrics_names
                    list_of_metrics = []
                    for metric in self.results_train_metrics[me]:
                        # print(me, len(self.results_train_metrics[me][metric]))
                        length = len(self.results_train_metrics[me][metric])
                        list_of_metrics.append(self.results_train_metrics[me][metric])

                    data = []
                    # print("""tamanho: {}    {}""".format(length, list_of_metrics))
                    for i in range(length):
                        row = []
                        for j in range(len(list_of_metrics)):
                            if len(list_of_metrics[j]) > 0:
                                row.append(list_of_metrics[j][i])
                            else:
                                row.append(0)

                        data.append(row)

            # print("File path: " + file_path)
            print("get results data", data, length)

            return file_path, header, data
        except Exception as e:
            print("get_results error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _save_shift_detection_metrics(self, server_round):

        try:
            print("save shift detection metrics")

            result_path = self.get_result_path("test")

            file_path = (
                    result_path
                    + f"shift_detection_metrics_{self.strategy_name}.csv"
            )

            for me in range(self.ME):

                y_true = self.shift_ground_truth_event[me]
                y_pred = self.shift_detected[me]

                if len(y_true) == 0:

                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0

                else:

                    tp = 1 if self.true_detection_round[me] is not None else 0
                    fp = len(self.false_alarm_rounds[me])

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = float(tp)
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if precision + recall > 0
                        else 0.0
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
                    self.shift_rounds[me][0],
                ]]

                self._write_rows(
                    file_path,
                    row
                )

        except Exception as e:

            print("_save_shift_detection_metrics error")

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )

    def _init_shift_detection_files(self):

        result_path = self.get_result_path("test")
        print("inicializou arquivos em ", result_path)
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
            print("saving detection metrics curve")
            result_path = self.get_result_path("test")

            file_path = (
                    result_path
                    + f"shift_detection_curve_{self.strategy_name}.csv"
            )

            for me in range(self.ME):
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

            print("_save_shift_detection_curve error")

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )