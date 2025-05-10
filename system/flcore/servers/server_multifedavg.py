# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import copy
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
import csv
import time
import numpy as np
from flcore.clients.client_multifedavg import MultiFedAvgClient
import sys
import os
from flcore.servers.utils.download_dataset import download_datasets
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flcore.clients.utils.models_utils import get_weights

# Define metric aggregation function
def weighted_average(metrics):
    try:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["Accuracy"] for num_examples, m in metrics]
        balanced_accuracies = [num_examples * m["Balanced accuracy"] for num_examples, m in metrics]
        loss = [num_examples * m["Loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"Accuracy": sum(accuracies) / sum(examples), "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
                "Loss": sum(loss) / sum(examples), "Round (t)": metrics[0][1]["Round (t)"], "Model size": metrics[0][1]["Model size"], "Alpha": metrics[0][1]["Alpha"]}
    except Exception as e:
        print("weighted_average error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def weighted_average_fit(metrics):
    try:
        # Multiply accuracy of each client by number of examples used
        # print(f"metricas recebidas: {metrics}")
        accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
        balanced_accuracies = [num_examples * m["train_balanced_accuracy"] for num_examples, m in metrics]
        loss = [num_examples * m["train_loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"Accuracy": sum(accuracies) / sum(examples), "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
                "Loss": sum(loss) / sum(examples), "Round (t)": metrics[0][1]["Round (t)"], "Model size": metrics[0][1]["Model size"]}
    except Exception as e:
        print("weighted_average_fit error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def weighted_loss_avg(results):
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

class MultiFedAvg:
    def __init__(self, args, models):

        try:
            self.clients = []

            self.evaluate_metrics_aggregation_fn = weighted_average
            self.fit_metrics_aggregation_fn = weighted_average_fit
            self.args = args
            self.global_model = models
            self.local_epochs = args.local_epochs
            self.total_clients = args.total_clients
            self.fraction_new_clients = args.fraction_new_clients
            self.fraction_fit = args.fraction_fit
            self.num_training_clients = int(self.total_clients * self.fraction_fit)
            self.round_new_clients = args.round_new_clients
            self.alpha = [float(i) for i in args.alpha]

            self.dataset = args.dataset
            self.model_name = args.model
            self.ME = len(self.global_model)
            self.number_of_rounds = args.number_of_rounds
            self.cd = f"experiment_id_{args.experiment_id}"
            self.strategy_name = args.strategy
            self.test_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                       "# training clients", "training clients and models", "Model size", "Alpha"]
            self.train_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                        "# training clients", "training clients and models", "Model size", "Alpha"]
            self.rs_test_acc = {me: [] for me in range(self.ME)}
            self.rs_test_auc = {me: [] for me in range(self.ME)}
            self.rs_train_loss = {me: [] for me in range(self.ME)}
            self.results_train_metrics = {me: {metric: [] for metric in self.train_metrics_names} for me in range(self.ME)}
            self.results_train_metrics_w = {me: {metric: [] for metric in self.train_metrics_names} for me in
                                            range(self.ME)}
            self.results_test_metrics = {me: {metric: [] for metric in self.test_metrics_names} for me in range(self.ME)}
            self.results_test_metrics_w = {me: {metric: [] for metric in self.test_metrics_names} for me in range(self.ME)}
            self.clients_results_test_metrics = {me: {metric: [] for metric in self.test_metrics_names} for me in
                                                 range(self.ME)}
            self.selected_clients_m = []
            self.selected_clients_m_ids_random = [[] for me in range(self.ME)]

            print("Dowenload datasets")
            download_datasets(self.dataset, self.alpha, self.total_clients)
            # Concept drift parameters
            self.experiment_id = args.experiment_id
            self.set_clients()
            # self.concept_drift_config = global_concept_dirft_config(self.ME, self.number_of_rounds, self.alpha, self.experiment_id, 0)

        except Exception as e:
            print("__init__ error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def set_clients(self):

        try:
            for i in range(self.total_clients):
                client = MultiFedAvgClient(self.args,
                                id=i,
                                   model=copy.deepcopy(self.global_model))
                self.clients.append(client)

        except Exception as e:
            print("set_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def train(self):
        try:
            self._get_models_size()
            parameters_aggregated_mefl = {me: [] for me in range(self.ME)}
            for me in range(self.ME):
                parameters_aggregated_mefl[me] = get_weights(self.global_model[me])
            for t in range(1, self.number_of_rounds + 1):
                s_t = time.time()
                self.selected_clients = self.select_clients(t)
                print(self.selected_clients)
                fit_results = []

                for me in range(len(self.selected_clients)):

                    for i in range(len(self.selected_clients[me])):
                        fit_results.append(self.clients[self.selected_clients[me][i]].fit(me, t, parameters_aggregated_mefl[me]))

                parameters_aggregated_mefl, metrics_aggregated_mefl = self.aggregate_fit(server_round=t, results=fit_results, failures=[])

                self.evaluate(t, parameters_aggregated_mefl)

        except Exception as e:
            print("configure_fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, t, parameters_aggregated_mefl):

        try:
            evaluate_results = []
            for me in range(self.ME):

                for i in range(len(self.clients)):

                    evaluate_results.append(self.clients[i].evaluate(me, t, parameters_aggregated_mefl[me]))

            loss_aggregated_mefl, metrics_aggregated_mefl = self.aggregate_evaluate(server_round=t, results=evaluate_results, failures=[])

        except Exception as e:
            print("evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate fit results using weighted average."""
        try:

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
            parameters_aggregated_mefl = {me: [] for me in range(self.ME)}

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
                    aggregated_ndarrays_mefl[me] = results_mefl[me][1].parameters

            for me in trained_models:
                parameters_aggregated_mefl[me] = aggregated_ndarrays_mefl[me]

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated_mefl = {me: [] for me in range(self.ME)}
            for me in trained_models:
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(num_examples, metrics) for _, num_examples, metrics in results_mefl[me]]
                    metrics_aggregated_mefl[me] = self.fit_metrics_aggregation_fn(fit_metrics)

            print("""finalizou aggregated fit""")

            self.parameters_aggregated_mefl = parameters_aggregated_mefl
            self.metrics_aggregated_mefl = metrics_aggregated_mefl

            return parameters_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            print("aggregate_fit error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate evaluation losses using weighted average."""
        try:

            print("""inicio aggregate evaluate {}""".format(server_round))

            results_mefl = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                parameters, num_examples, result = results[i]
                me = result[2]["me"]
                results_mefl[me].append(result)


            # Aggregate loss
            # print("""metricas recebidas rodada {}: {}""".format(server_round, results_mefl[0]))
            loss_aggregated_mefl = {me: 0. for me in range(self.ME)}
            for me in results_mefl.keys():
                loss_aggregated = weighted_loss_avg(
                    [
                        (num_examples, loss)
                        for loss, num_examples, metrics in results_mefl[me]
                    ]
                )
                loss_aggregated_mefl[int(me)] = loss_aggregated

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated_mefl = {me: {} for me in range(self.ME)}
            if self.evaluate_metrics_aggregation_fn:
                for me in results_mefl.keys():
                    eval_metrics = [(num_examples, metrics) for loss, num_examples, metrics in results_mefl[me]]
                    metrics_aggregated_mefl[int(me)] = self.evaluate_metrics_aggregation_fn(eval_metrics)

            mode = "w"

            for me in range(self.ME):
                self.add_metrics(server_round, metrics_aggregated_mefl, me)
                self._save_results(mode, me)


            return loss_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            print("aggregate_evaluate error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def select_clients(self, t):

        try:
            selected_clients = list(np.random.choice(self.clients, self.num_training_clients, replace=False))
            selected_clients = [i.client_id for i in selected_clients]

            n = len(selected_clients) // self.ME
            sc = np.array_split(selected_clients, self.ME)

            self.n_trained_clients = sum([len(i) for i in sc])

            return sc

        except Exception as e:
            print("select_clients error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def add_metrics(self, server_round, metrics_aggregated, me):
        try:
            metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
            metrics_aggregated[me]["# training clients"] = self.n_trained_clients
            metrics_aggregated[me]["training clients and models"] = self.selected_clients_m[me]

            for metric in metrics_aggregated[me]:
                self.results_test_metrics[me][metric].append(metrics_aggregated[me][metric])
        except Exception as e:
            print("add_metrics error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _save_results(self, mode, me):

        # train
        try:
            # print("""save results: {}""".format(self.results_test_metrics[me]))
            file_path, header, data = self._get_results('train', '', me)
            # print("""dados: {} {}""".format(data, file_path))
            self._write_header(file_path, header=header, mode=mode)
            self._write_outputs(file_path, data=data)

            # test

            file_path, header, data = self._get_results('test', '', me)
            self._write_header(file_path, header=header, mode=mode)
            self._write_outputs(file_path, data=data)
        except Exception as e:
            print("save_results error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_results(self, train_test, mode, me):

        try:
            algo = self.dataset[me] + "_" + self.strategy_name

            result_path = self.get_result_path(train_test)


            if not os.path.exists(result_path):
                os.makedirs(result_path)

            file_path = result_path + "{}.csv".format(algo)

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
            print(data)

            return file_path, header, data
        except Exception as e:
            print("get_results error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _write_header(self, filename, header, mode):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, mode) as server_log_file:
                writer = csv.writer(server_log_file)
                writer.writerow(header)
        except Exception as e:
            print("_write_header error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _write_outputs(self, filename, data, mode='a'):
        try:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    element = data[i][j]
                    if type(element) == float:
                        element = round(element, 6)
                        data[i][j] = element
            with open(filename, 'a') as server_log_file:
                writer = csv.writer(server_log_file)
                writer.writerows(data)
        except Exception as e:
            print("_write_outputs error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_models_size(self):
        try:
            models_size = []
            for me in range(self.ME):
                model = self.global_model[me]
                parameters = [i.detach().cpu().numpy() for i in model.parameters()]
                size = 0
                for i in range(len(parameters)):
                    size += parameters[i].nbytes
                models_size.append(size)
            self.models_size = models_size
        except Exception as e:
            print("_get_models_size error")
            print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def get_result_path(self, train_test):

        result_path = """results/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
            self.total_clients,
            self.alpha,
            self.dataset,
            self.model_name,
            self.fraction_fit,
            self.number_of_rounds,
            self.local_epochs,
            train_test)

        return result_path

    def write_log(self, result_path):

        # Abra um arquivo para gravação
        result_path = """{}/log_{}""".format(result_path, self.strategy_name)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as f:
            # Redirecione a saída padrão para o arquivo
            original = sys.stdout
            sys.stdout = f
            sys.stdout = original

