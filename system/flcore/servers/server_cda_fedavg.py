from flcore.servers.server_multifedavg import MultiFedAvg
from flcore.clients.client_cda_fedavg import ClientCDAFedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
import copy
import os
import sys
import torch


class CDAFedAvg(MultiFedAvg):

    def __init__(self, args, models, fold_id):
        super().__init__(
            args,
            models,
            fold_id
        )

        self.MIN_TRAINING_CLIENTS = 3

        self.cda_update_buffer = {
            me: []
            for me in range(self.ME)
        }

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
            "Data shift"
        ]
        self.test_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                   "# training clients", "training clients and models", "Model size", "Fold ID", "Alpha", "Drift clients", "Drift rate", "Data shift", "Ground truth shift"]
        self.results_test_metrics = {me: {metric: [] for metric in self.test_metrics_names} for me in range(self.ME)}

        # Detector
        self.detector = self.strategy_name

        # Dataset(s)
        self.dataset = self.args.dataset

        # Shift type
        self.shift_type = (
            "Label"
            if "label_shift" in self.args.experiment_id
            else "Concept"
        )

        # Shift configuration
        self.shift_configuration = (
            self.args.experiment_id
            .replace("label_shift#", "")
            .replace("concept_drift#", "")
            .replace("_sudden", "")
        )

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
            client = ClientCDAFedAvg(
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
        try:

            print("entrou aggregate_fit")

            # ========================================================
            # Reset clients selected for actual training
            # ========================================================

            self.selected_clients_m = [
                []
                for _ in range(self.ME)
            ]

            # ========================================================
            # Separate:
            #
            # 1. all results -> detection
            # 2. training results -> update buffer
            # ========================================================

            all_results_mefl = {
                me: []
                for me in range(self.ME)
            }

            training_results_mefl = {
                me: []
                for me in range(self.ME)
            }

            # Models that receive at least one new local update
            trained_models = []

            # ========================================================
            # Collect all client results
            # ========================================================

            for (
                    parameters,
                    num_examples,
                    result
            ) in results:

                me = result["me"]

                client_id = result["client_id"]

                cda_training = int(
                    result.get(
                        "CDA training",
                        1
                    )
                )

                # ----------------------------------------------------
                # ALL results remain available for drift detection.
                # ----------------------------------------------------

                all_results_mefl[me].append(
                    (
                        parameters,
                        num_examples,
                        result
                    )
                )

                # ----------------------------------------------------
                # Only clients that actually trained participate in
                # the CDA-FedAvg update buffer.
                # ----------------------------------------------------

                if (
                        cda_training == 1
                        and num_examples > 0
                ):

                    training_results_mefl[me].append(
                        (
                            parameters,
                            num_examples,
                            result
                        )
                    )

                    self.selected_clients_m[me].append(
                        client_id
                    )

                    # ------------------------------------------------
                    # Add local update to persistent buffer.
                    #
                    # The update is NOT immediately aggregated.
                    # ------------------------------------------------

                    self.cda_update_buffer[me].append(
                        (
                            parameters,
                            num_examples,
                            result
                        )
                    )

                    if me not in trained_models:
                        trained_models.append(me)

            # ========================================================
            # Aggregate independently for each MEFL model
            #
            # IMPORTANT:
            # A single client is NEVER sufficient to update the
            # global model.
            # ========================================================

            aggregated_ndarrays_mefl = {
                me: None
                for me in range(self.ME)
            }

            models_updated_this_round = []

            for me in range(self.ME):

                buffered_results = self.cda_update_buffer[me]

                n_buffered_updates = len(
                    buffered_results
                )

                print(
                    f"[CDA-FedAvg] Model {me}: "
                    f"{n_buffered_updates} buffered updates "
                    f"(minimum = {self.MIN_TRAINING_CLIENTS})"
                )

                # ----------------------------------------------------
                # Not enough accumulated updates.
                #
                # Keep them in the buffer and preserve the current
                # global model.
                # ----------------------------------------------------

                if n_buffered_updates < self.MIN_TRAINING_CLIENTS:
                    print(
                        f"[CDA-FedAvg] Model {me}: "
                        f"not enough clients for aggregation. "
                        f"Keeping current global model."
                    )

                    continue

                # ----------------------------------------------------
                # Enough clients accumulated.
                #
                # Perform weighted FedAvg over the buffered updates.
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
                    ) in buffered_results
                ]

                aggregated_ndarrays_mefl[me] = aggregate(
                    weights_results
                )

                models_updated_this_round.append(me)

                # ----------------------------------------------------
                # Clear buffer ONLY after successful aggregation.
                # ----------------------------------------------------

                self.cda_update_buffer[me] = []

                print(
                    f"[CDA-FedAvg] Model {me}: "
                    f"global model updated using "
                    f"{len(weights_results)} accumulated clients."
                )

            # ========================================================
            # Update only models for which enough clients accumulated
            # ========================================================

            for me in models_updated_this_round:
                self.parameters_aggregated_mefl[me] = (
                    aggregated_ndarrays_mefl[me]
                )

            # ========================================================
            # Metrics
            # ========================================================

            metrics_aggregated_mefl = {
                me: {}
                for me in range(self.ME)
            }

            for me in range(self.ME):

                # ====================================================
                # Detection is calculated from ALL clients.
                # ====================================================

                model_all_results = (
                    all_results_mefl[me]
                )

                n_clients_detection = len(
                    model_all_results
                )

                n_drift = sum(
                    int(
                        metrics.get(
                            "Drift detected",
                            0
                        )
                    )
                    for (
                        _,
                        _,
                        metrics
                    ) in model_all_results
                )

                self.drift_clients[me] = n_drift

                self.drift_rate[me] = (
                    n_drift
                    / n_clients_detection
                    if n_clients_detection > 0
                    else 0.0
                )

                # ----------------------------------------------------
                # A detection event exists when at least one client
                # detects a local drift.
                # ----------------------------------------------------

                self.data_shift_type[me] = (
                    "DATA_SHIFT"
                    if n_drift > 0
                    else "NO_SHIFT"
                )

                # ====================================================
                # Aggregate normal training metrics only from clients
                # that actually trained THIS round.
                # ====================================================

                model_training_results = (
                    training_results_mefl[me]
                )

                if (
                        len(model_training_results) > 0
                        and self.fit_metrics_aggregation_fn
                ):

                    fit_metrics = [
                        (
                            num_examples,
                            metrics
                        )
                        for (
                            _,
                            num_examples,
                            metrics
                        ) in model_training_results
                    ]

                    aggregated_training_metrics = (
                        self.fit_metrics_aggregation_fn(
                            fit_metrics
                        )
                    )

                    if aggregated_training_metrics:
                        metrics_aggregated_mefl[me].update(
                            aggregated_training_metrics
                        )

                # ====================================================
                # CDA-FedAvg detection metrics
                # ====================================================

                metrics_aggregated_mefl[me][
                    "Drift clients"
                ] = self.drift_clients[me]

                metrics_aggregated_mefl[me][
                    "Drift rate"
                ] = self.drift_rate[me]

                metrics_aggregated_mefl[me][
                    "Data shift"
                ] = self.data_shift_type[me]

                # ====================================================
                # Additional CDA-FedAvg aggregation information
                # ====================================================

                metrics_aggregated_mefl[me][
                    "Training clients"
                ] = len(
                    training_results_mefl[me]
                )

                metrics_aggregated_mefl[me][
                    "Buffered training clients"
                ] = len(
                    self.cda_update_buffer[me]
                )

                metrics_aggregated_mefl[me][
                    "Global model updated"
                ] = int(
                    me in models_updated_this_round
                )

                # ====================================================
                # Data-shift ground truth
                # ====================================================

                ground_truth_state = int(
                    any(
                        server_round >= r
                        for r in self.shift_rounds[me]
                    )
                )

                ground_truth_event = int(
                    server_round
                    in self.shift_rounds[me]
                )

                # ====================================================
                # Detector event
                # ====================================================

                current_state = (
                    self.data_shift_type[me]
                )

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
                # Historical states
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
                # False alarms / correct detection
                # ====================================================

                if self.detection_event[me]:

                    if len(
                            self.shift_rounds[me]
                    ) > 0:

                        shift_round = min(
                            self.shift_rounds[me]
                        )

                        if (
                                server_round
                                < shift_round
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
                                    server_round
                                    - shift_round
                            )

                # ====================================================
                # Detection evaluation metrics
                # ====================================================

                metrics_aggregated_mefl[me][
                    "Ground truth shift"
                ] = ground_truth_state

                metrics_aggregated_mefl[me][
                    "Detection delay"
                ] = self.detection_delay[me]

                metrics_aggregated_mefl[me][
                    "False alarm"
                ] = len(
                    self.false_alarm_rounds[me]
                )

                metrics_aggregated_mefl[me][
                    "Detection rate"
                ] = (
                    1.0
                    if self.true_detection_round[me]
                       is not None
                    else 0.0
                )

            # ========================================================
            # Keep existing CSV / metric writing infrastructure
            # ========================================================

            self._save_data_metrics()

            self._save_shift_detection_metrics(
                server_round
            )

            self._save_shift_detection_curve(
                server_round
            )

            # ========================================================
            # Save current aggregated state
            #
            # If the minimum number of clients was NOT reached,
            # self.parameters_aggregated_mefl remains unchanged.
            # ========================================================

            self.parameters_aggregated_mefl = (
                self.parameters_aggregated_mefl
            )

            self.metrics_aggregated_mefl = (
                metrics_aggregated_mefl
            )

            print(
                f"rodada {server_round} "
                f"metricas agregadas de treino "
                f"{self.metrics_aggregated_mefl}"
            )

            return (
                self.parameters_aggregated_mefl,
                metrics_aggregated_mefl
            )

        except Exception as e:

            print("aggregate_fit error")

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

    def add_metrics(self, server_round, metrics_aggregated, me):
        try:
            print("adicionar metricas")
            metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
            metrics_aggregated[me]["# training clients"] = self.n_trained_clients
            metrics_aggregated[me]["training clients and models"] = self.selected_clients_m[me]
            metrics_aggregated[me]["Fold ID"] = self.fold_id
            metrics_aggregated[me]["Data shift"] = self.data_shift_type[me]
            metrics_aggregated[me]["Drift clients"] = self.drift_clients[me]
            metrics_aggregated[me]["Drift rate"] = self.drift_rate[me]
            metrics_aggregated[me]["Data shift"] = self.data_shift_type[me]

            metrics_aggregated[me]["Ground truth shift"] = (
                self.shift_ground_truth_state[me][-1]
                if len(self.shift_ground_truth_state[me]) > 0
                else 0
            )

            for metric in metrics_aggregated[me]:
                self.results_test_metrics[me][metric].append(metrics_aggregated[me][metric])
        except Exception as e:
            print("add_metrics error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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

    def _cda_drift_adaptation(self, me, t):
        """
        CDA-FedAvg Algorithm 6.

        The current concept is added to long-term memory and
        the model is trained using rehearsal over all concepts.
        """

        # ------------------------------------------------------------
        # We cannot start training until the new concept has enough
        # representative data.
        # ------------------------------------------------------------

        if not self._cda_has_balanced_concept(me):
            self.cda_training[me] = False
            self.cda_new_concept_data[me] = len(
                self.cda_L[me]["y"]
            )

            return False

        self.cda_training[me] = True
        self.cda_training_round[me] = 0

        return True