from flcore.servers.server_multifedavg import MultiFedAvg
from flcore.clients.client_fedcond import ClientFedConD
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
import copy
import os
import sys


class FedConD(MultiFedAvg):

    def __init__(self, args, models, fold_id):
        super().__init__(
            args,
            models,
            fold_id
        )

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
        """Aggregate fit results using weighted average."""
        try:
            print("entrou aggregate_fit")
            self.selected_clients_m = [[] for me in range(self.ME)]

            trained_models = []

            results_mefl = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                parameter, num_examples, result = results[i]
                me = result["me"]
                if me not in trained_models:
                    trained_models.append(me)
                client_id = result["client_id"]
                self.selected_clients_m[me].append(client_id)
                results_mefl[me].append(results[i])


            aggregated_ndarrays_mefl = {me: None for me in range(self.ME)}
            aggregated_ndarrays_mefl = {me: [] for me in range(self.ME)}
            weights_results_mefl = {me: [] for me in range(self.ME)}
            # parameters_aggregated_mefl = {me: [] for me in range(self.ME)}

            for me in trained_models:
                # Convert results
                weights_results = [
                    (parameters, num_examples)
                    for parameters, num_examples, fit_res in results_mefl[me]
                ]
                aggregated_ndarrays_mefl[me] = aggregate(weights_results)
                if len(weights_results) > 1:
                    aggregated_ndarrays_mefl[me] = aggregate(weights_results)
                elif len(weights_results) == 1:
                    aggregated_ndarrays_mefl[me] = results_mefl[me][0][0]

            for me in trained_models:
                self.parameters_aggregated_mefl[me] = aggregated_ndarrays_mefl[me]

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated_mefl = {me: [] for me in range(self.ME)}
            for me in trained_models:
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(num_examples, metrics) for _, num_examples, metrics in results_mefl[me]]
                    metrics_aggregated_mefl[me] = self.fit_metrics_aggregation_fn(fit_metrics)

                    n_drift = sum(
                        m["Drift detected"]
                        for _, _, m in results_mefl[me]
                    )

                    drift_rate = (
                            n_drift /
                            len(results_mefl[me])
                    )

                    self.drift_clients[me] = n_drift

                    self.drift_rate[me] = (
                            n_drift / len(results_mefl[me])
                    )
                    self.drift_rate_history[me].append(
                        self.drift_rate[me]
                    )

                    self.data_shift_type[me] = (
                        "DATA_SHIFT"
                        if self.drift_rate[me] >= 0.4
                        else "NO_SHIFT"
                    )

                    metrics_aggregated_mefl[me]["Drift clients"] = n_drift
                    metrics_aggregated_mefl[me]["Drift rate"] = drift_rate
                    metrics_aggregated_mefl[me]["Data shift"] = self.data_shift_type[me]

                    # =====================================================
                    # Data shift detection evaluation
                    # =====================================================

                    # Estado do shift (para curvas)
                    ground_truth_state = int(
                        any(
                            server_round >= r
                            for r in self.shift_rounds[me]
                        )
                    )

                    # Evento do shift (para Precision/Recall/F1)
                    ground_truth_event = int(
                        server_round in self.shift_rounds[me]
                    )

                    # Estado atual do detector
                    current_state = self.data_shift_type[me]

                    # Evento do detector
                    self.detection_event[me] = int(
                        self.previous_detector_state[me] == "NO_SHIFT"
                        and current_state == "DATA_SHIFT"
                    )

                    # Atualiza estado anterior
                    self.previous_detector_state[me] = current_state

                    # Salva históricos
                    self.shift_ground_truth_state[me].append(
                        ground_truth_state
                    )

                    self.shift_ground_truth_event[me].append(
                        ground_truth_event
                    )

                    self.shift_detected[me].append(
                        self.detection_event[me]
                    )

                    # Atualiza métricas do detector
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

                    metrics_aggregated_mefl[me]["Ground truth shift"] = ground_truth_state
                    metrics_aggregated_mefl[me]["Detection delay"] = self.detection_delay[me]
                    metrics_aggregated_mefl[me]["False alarm"] = len(
                        self.false_alarm_rounds[me]
                    )
                    metrics_aggregated_mefl[me]["Detection rate"] = self.drift_rate[me]

            # if server_round > 10:
            self._save_data_metrics()
            self._save_shift_detection_metrics(server_round)
            self._save_shift_detection_curve(server_round)

            # print("""finalizou aggregated fit""")

            self.parameters_aggregated_mefl = self.parameters_aggregated_mefl
            self.metrics_aggregated_mefl = metrics_aggregated_mefl

            print(f"rodada {server_round} metricas agregadas de treino {self.metrics_aggregated_mefl}")

            return self.parameters_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            print("aggregate_fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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
                self.shift_ground_truth[me][-1]
                if len(self.shift_ground_truth[me]) > 0
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

                if len(y_true) == 0:

                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0

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