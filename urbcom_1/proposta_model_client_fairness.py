# =====================================================
# PARTE 1 — IMPORTS E CONFIGURAÇÕES GERAIS
# =====================================================

import copy
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from models_utils import load_data


# =====================================================
# CONFIGURAÇÕES GLOBAIS
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# ORÇAMENTOS DE RECURSOS POR RODADA
# =====================================================

BASE_SEED = 42
NUM_FOLDS = 1
NUM_CLIENTS = 40
ROUNDS = 100
FRAC = 0.3
K_CLIENTS = int(FRAC * NUM_CLIENTS)   # exemplo: 30% no máximo

LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01

# DIRICHLET_ALPHA = 0.1
DIRICHLET_ALPHA = 1.0
regime = "realistic"
# regime = "benign"
# regime = "severe"

MODEL_COST = {
    "cifar": 0.3,
    "gtsrb": 0.5
}

# =====================================================
# FAIR RESOURCE BUDGET
# =====================================================

LIGHT_MODEL = min(MODEL_COST, key=MODEL_COST.get)

# permitir aproximadamente K_CLIENTS treinamentos por rodada
avg_speed = 0.65

FAIR_RESOURCE_BUDGET = K_CLIENTS * MODEL_COST[LIGHT_MODEL] / avg_speed

Path("results/").mkdir(parents=True, exist_ok=True)

# =====================================================
# PARTE 2 — MODELOS
# =====================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =====================================================
# ESTADOS GLOBAIS (INICIALIZADOS POR FOLD)
# =====================================================

global_models = {}
client_resources = {}
client_acc = {}
client_loss = {}
client_delta_acc = {}
global_acc_history = {}
proposta_logs = {}

# =====================================================
# FAIRNESS STATES
# =====================================================

client_resource_usage = {}

# =====================================================
# REGIMES EXPERIMENTAIS
# =====================================================

def get_regime(name: str):

    regimes = {

        "benign": {
            "TIME_MAX": 3.0
        },

        "realistic": {
            "TIME_MAX": 2.5
        },

        "severe": {
            "TIME_MAX": 2.0
        }
    }

    return regimes[name]

def apply_regime(name: str):

    global TIME_MAX, current_regime

    current_regime = get_regime(name)

    TIME_MAX = current_regime["TIME_MAX"]

    for cid in range(NUM_CLIENTS):

        # heterogeneidade computacional
        client_resources[cid]["speed"] = np.random.uniform(0.3, 1.0)

    print(f"🧪 Regime aplicado: {name.upper()}")

# =====================================================
# RECURSOS E VIABILIDADE
# =====================================================

def estimate_training_time(cid, model_name):

    speed = client_resources[cid]["speed"]

    # tempo heterogêneo baseado na capacidade do cliente
    train_time = MODEL_COST[model_name] / speed

    return train_time

# =====================================================
# RESET DO ESTADO GLOBAL (POR FOLD)
# =====================================================

def reset_experiment_state():

    global global_models
    global client_resources
    global client_acc
    global client_loss
    global client_delta_acc
    global proposta_logs
    global global_acc_history

    # -------- Modelos --------
    global_models = {
        "cifar": SimpleCNN(10).to(DEVICE),
        "gtsrb": SimpleCNN(43).to(DEVICE)
    }

    global_acc_history = {
        "cifar": [],
        "gtsrb": []
    }

    # -------- Recursos --------
    client_resources = {
        cid: {
            "speed": np.random.uniform(0.3, 1.0)
        }
        for cid in range(NUM_CLIENTS)
    }

    # -------- Métricas --------
    client_acc = {
        cid: {"cifar": 0.0, "gtsrb": 0.0}
        for cid in range(NUM_CLIENTS)
    }

    client_loss = {
        cid: {"cifar": float("inf"), "gtsrb": float("inf")}
        for cid in range(NUM_CLIENTS)
    }

    client_delta_acc = {
        cid: {"cifar": 0.0, "gtsrb": 0.0}
        for cid in range(NUM_CLIENTS)
    }

    # -------- Fairness tracking --------
    global client_resource_usage

    client_resource_usage = {
        cid: {
            "cifar": 0.0,
            "gtsrb": 0.0
        }
        for cid in range(NUM_CLIENTS)
    }

    # -------- Logs --------
    # -------- Logs --------
    proposta_logs = {
        "clients_per_model": [],

        "resource_usage_cifar": [],
        "resource_usage_gtsrb": [],

        # fairness entre modelos
        "fairness_resource": [],

        # fairness considerando capacidade dos clientes
        "client_capacity_fairness": []
    }

# =====================================================
# PARTE 7 — TREINO LOCAL E AVALIAÇÃO
# =====================================================

DATASET_INPUT_MAP = {
    "CIFAR10": "img",
    "MNIST": "image",
    "EMNIST": "image",
    "GTSRB": "image",
    "Gowalla": "sequence",
    "WISDM-W": "sequence",
    "ImageNet": "image",
    "ImageNet10": "image",
    "wikitext": "sequence",
    "Foursquare": "sequence"
}


def client_update(dataset_name, model, loader, epochs, lr):

    model = copy.deepcopy(model)
    model.train()

    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    total_loss, total_samples = 0.0, 0
    key = DATASET_INPUT_MAP[dataset_name]

    for _ in range(epochs):
        for batch in loader:

            x = batch[key]
            y = batch["label"]

            x, y = x.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    return model.state_dict(), total_loss / total_samples, total_samples


@torch.no_grad()
def evaluate_model(model, loader, dataset_name):

    model.eval()
    correct, total = 0, 0

    key = DATASET_INPUT_MAP[dataset_name]

    for batch in loader:

        x = batch[key]
        y = batch["label"]

        x, y = x.to(DEVICE), y.to(DEVICE)

        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total


@torch.no_grad()
def evaluate_global_model(model, test_loaders, dataset_name):

    model.eval()
    accs = []

    for cid in range(NUM_CLIENTS):
        acc = evaluate_model(model, test_loaders[cid], dataset_name)
        accs.append(acc)

    return float(np.mean(accs))


def fedavg(updates):

    total = sum(n for _, n in updates)
    avg = {}

    for k in updates[0][0].keys():
        avg[k] = sum(state[k] * (n / total) for state, n in updates)

    return avg

# =====================================================
# PARTE 8 — LOOP FEDERADO COMPLETO
# =====================================================

def append_result_to_csv(row_dict, filename):

    df_row = pd.DataFrame([row_dict])
    file_exists = os.path.exists(filename)

    df_row.to_csv(
        filename,
        mode="a",
        header=not file_exists,
        index=False
    )

def clear_previous_results():
    for model_name in ["cifar", "gtsrb"]:
        filename = f"results/proposta_{model_name}_regime_{regime}_frac_{FRAC}_alpha_{DIRICHLET_ALPHA}.csv"
        if os.path.exists(filename):
            os.remove(filename)



def run_experiment():

    clear_previous_results()

    # -------------------------------------------------
    # Loop por Folds
    # -------------------------------------------------

    for fold in range(NUM_FOLDS):

        print("\n===================================================")
        print(f"🚀 INICIANDO FOLD {fold}")
        print("===================================================")

        fold_seed = BASE_SEED + fold * 1000

        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        reset_experiment_state()
        cumulative_energy = 0.0

        apply_regime(regime)

        # -------------------------------------------------
        # DataLoaders por cliente
        # -------------------------------------------------

        train_loaders = defaultdict(dict)
        test_loaders = defaultdict(dict)

        for cid in range(NUM_CLIENTS):

            train_loader, test_loader = load_data(
                dataset_name="CIFAR10",
                alpha=DIRICHLET_ALPHA,
                partition_id=cid,
                num_partitions=NUM_CLIENTS + 1,
                batch_size=BATCH_SIZE,
                fold_id=fold + 1,
                data_sampling_percentage=0.8,
                get_from_volume=True
            )

            train_loaders[cid]["cifar"] = train_loader
            test_loaders[cid]["cifar"] = test_loader

            train_loader, test_loader = load_data(
                dataset_name="GTSRB",
                alpha=DIRICHLET_ALPHA,
                partition_id=cid,
                num_partitions=NUM_CLIENTS + 1,
                batch_size=BATCH_SIZE,
                fold_id=fold + 1,
                data_sampling_percentage=0.8,
                get_from_volume=True
            )

            train_loaders[cid]["gtsrb"] = train_loader
            test_loaders[cid]["gtsrb"] = test_loader

        # -------------------------------------------------
        # Rodadas
        # -------------------------------------------------



        for rnd in range(1, ROUNDS + 1):

            round_seed = fold_seed + rnd
            random.seed(round_seed)
            np.random.seed(round_seed)
            torch.manual_seed(round_seed)

            # =====================================================
            # containers de updates dos modelos
            # =====================================================

            model_updates = {
                "cifar": [],
                "gtsrb": []
            }

            energy_this_round = 0.0

            print(f"\n🔄 FOLD {fold} | RODADA {rnd}")

            real_training_counter = {m: 0 for m in global_models}

            # =====================================================
            # 1) Balanced Resource Selection + Client Fairness
            # =====================================================

            clients_cifar = []
            clients_gtsrb = []

            resource_usage = {
                "cifar": 0.0,
                "gtsrb": 0.0
            }

            all_clients = list(range(NUM_CLIENTS))
            random.shuffle(all_clients)

            # peso do fairness considerando capacidade
            LAMBDA_CAPACITY = 0.3

            # =====================================================
            # NORMALIZAÇÃO DE USO DE CLIENTES
            # =====================================================

            max_model_usage = {
                "cifar": max(client_resource_usage[c]["cifar"] for c in range(NUM_CLIENTS)),
                "gtsrb": max(client_resource_usage[c]["gtsrb"] for c in range(NUM_CLIENTS))
            }

            max_total_usage = max(
                client_resource_usage[c]["cifar"] + client_resource_usage[c]["gtsrb"]
                for c in range(NUM_CLIENTS)
            )

            # evitar divisão por zero
            for m in max_model_usage:
                if max_model_usage[m] == 0:
                    max_model_usage[m] = 1.0

            if max_total_usage == 0:
                max_total_usage = 1.0

            for cid in all_clients:

                candidates = []

                total_usage = (
                        client_resource_usage[cid]["cifar"] +
                        client_resource_usage[cid]["gtsrb"]
                )

                # -------------------------------------------------
                # CIFAR
                # -------------------------------------------------

                train_time = estimate_training_time(cid, "cifar")

                if (
                        resource_usage["cifar"] + train_time <= FAIR_RESOURCE_BUDGET
                ):

                    new_usage = resource_usage["cifar"] + train_time
                    other_usage = resource_usage["gtsrb"]

                    total = new_usage + other_usage

                    if total > 0:
                        imbalance = abs(new_usage - other_usage) / total
                    else:
                        imbalance = 0.0

                    model_usage = client_resource_usage[cid]["cifar"] / max_model_usage["cifar"]
                    total_usage_norm = total_usage / max_total_usage

                    capacity_penalty = total_usage_norm

                    score = (
                            imbalance
                            + LAMBDA_CAPACITY * capacity_penalty
                    )

                    candidates.append(("cifar", score))

                # -------------------------------------------------
                # GTSRB
                # -------------------------------------------------

                train_time = estimate_training_time(cid, "gtsrb")

                if (
                        resource_usage["gtsrb"] + train_time <= FAIR_RESOURCE_BUDGET
                ):

                    new_usage = resource_usage["gtsrb"] + train_time
                    other_usage = resource_usage["cifar"]

                    total = new_usage + other_usage

                    if total > 0:
                        imbalance = abs(new_usage - other_usage) / total
                    else:
                        imbalance = 0.0

                    model_usage = client_resource_usage[cid]["gtsrb"] / max_model_usage["gtsrb"]
                    total_usage_norm = total_usage / max_total_usage

                    capacity_penalty = total_usage_norm

                    score = (
                            imbalance
                            + LAMBDA_CAPACITY * capacity_penalty
                    )

                    candidates.append(("gtsrb", score))

                if not candidates:
                    continue

                chosen_model = min(candidates, key=lambda x: x[1])[0]

                if chosen_model == "cifar":
                    clients_cifar.append(cid)
                else:
                    clients_gtsrb.append(cid)

                train_time = estimate_training_time(cid, chosen_model)
                resource_usage[chosen_model] += train_time

            # =====================================================
            # 2) Treino CIFAR (proposta puro)
            # =====================================================

            for cid in clients_cifar:
                train_time = estimate_training_time(cid, "cifar")

                local_model = copy.deepcopy(global_models["cifar"])

                state_dict, loss, n_samples = client_update(
                    "CIFAR10",
                    local_model,
                    train_loaders[cid]["cifar"],
                    epochs=LOCAL_EPOCHS,
                    lr=LR
                )

                model_updates["cifar"].append((state_dict, n_samples))
                real_training_counter["cifar"] += 1

                client_resource_usage[cid]["cifar"] += train_time

                acc = evaluate_model(
                    local_model,
                    test_loaders[cid]["cifar"],
                    "CIFAR10"
                )

                client_acc[cid]["cifar"] = acc
                client_loss[cid]["cifar"] = loss

            # =====================================================
            # 3) Treino GTSRB (proposta puro)
            # =====================================================

            for cid in clients_gtsrb:
                train_time = estimate_training_time(cid, "gtsrb")

                local_model = copy.deepcopy(global_models["gtsrb"])

                state_dict, loss, n_samples = client_update(
                    "GTSRB",
                    local_model,
                    train_loaders[cid]["gtsrb"],
                    epochs=LOCAL_EPOCHS,
                    lr=LR
                )

                model_updates["gtsrb"].append((state_dict, n_samples))
                real_training_counter["gtsrb"] += 1

                client_resource_usage[cid]["gtsrb"] += train_time

                acc = evaluate_model(
                    local_model,
                    test_loaders[cid]["gtsrb"],
                    "GTSRB"
                )

                client_acc[cid]["gtsrb"] = acc
                client_loss[cid]["gtsrb"] = loss

            # =====================================================
            # 4) FedAvg (somente se houver updates)
            # =====================================================

            for model_name, updates in model_updates.items():
                if len(updates) > 0:
                    new_state = fedavg(updates)
                    global_models[model_name].load_state_dict(new_state)

            # =====================================================
            # FAIRNESS baseada em custo teórico
            # =====================================================

            resource_cifar = sum(
                estimate_training_time(cid, "cifar")
                for cid in clients_cifar
            )

            resource_gtsrb = sum(
                estimate_training_time(cid, "gtsrb")
                for cid in clients_gtsrb
            )

            total_resource = resource_cifar + resource_gtsrb

            # -------------------------------------------------
            # 1) Fairness Resource (erro relativo)
            # menor = melhor
            # -------------------------------------------------

            if total_resource > 0:
                fairness_resource = abs(resource_cifar - resource_gtsrb) / total_resource
            else:
                fairness_resource = 1.0

            # =====================================================
            # CLIENT CAPACITY FAIRNESS
            # =====================================================

            utilization = []

            for cid in range(NUM_CLIENTS):

                capacity = client_resources[cid]["speed"]

                resource = (
                        client_resource_usage[cid]["cifar"] +
                        client_resource_usage[cid]["gtsrb"]
                )

                if capacity > 0:
                    utilization.append(resource / capacity)

            if len(utilization) > 0:

                num = (sum(utilization) ** 2)
                den = NUM_CLIENTS * sum(u ** 2 for u in utilization)

                if den > 0:
                    client_capacity_fairness = num / den
                else:
                    client_capacity_fairness = 1.0
            else:
                client_capacity_fairness = 1.0

            # -------------------------------------------------
            # salvar métricas
            # -------------------------------------------------

            proposta_logs["resource_usage_cifar"].append(
                real_training_counter["cifar"] * MODEL_COST["cifar"]
            )

            proposta_logs["resource_usage_gtsrb"].append(
                real_training_counter["gtsrb"] * MODEL_COST["gtsrb"]
            )

            # fairness entre modelos
            proposta_logs["fairness_resource"].append(fairness_resource)

            # fairness considerando capacidade dos clientes
            proposta_logs["client_capacity_fairness"].append(client_capacity_fairness)

            # 4) Avaliação global
            for model_name, model in global_models.items():

                dataset_name = {
                    "cifar": "CIFAR10",
                    "gtsrb": "GTSRB"
                }[model_name]

                global_acc = evaluate_global_model(
                    model,
                    {cid: test_loaders[cid][model_name]
                     for cid in range(NUM_CLIENTS)},
                    dataset_name
                )

                global_acc_history[model_name].append(global_acc)

                print(
                    f"📊 FOLD {fold} | Modelo {model_name.upper()} | "
                    f"Acurácia global média: {global_acc:.4f}"
                )

                row_data = {
                    "algorithm": "fair_resource",
                    "fold": fold,
                    "round": rnd,
                    "dataset": model_name,
                    "global_acc": global_acc,

                    "clients_cifar": real_training_counter["cifar"],
                    "clients_gtsrb": real_training_counter["gtsrb"],

                    "clients_selected": real_training_counter[model_name],
                    "clients_selected_total": sum(real_training_counter.values()),

                    # resource usage
                    "resource_usage_cifar": proposta_logs["resource_usage_cifar"][-1],
                    "resource_usage_gtsrb": proposta_logs["resource_usage_gtsrb"][-1],

                    # fairness metrics
                    "fairness_resource": proposta_logs["fairness_resource"][-1],
                    "client_capacity_fairness": proposta_logs["client_capacity_fairness"][-1],
                }

                filename = f"results/proposta_{model_name}_regime_{regime}_frac_{FRAC}_alpha_{DIRICHLET_ALPHA}.csv"
                append_result_to_csv(row_data, filename)

            # =====================================================
            # Atualiza clientes por modelo (TREINAMENTO REAL)
            # =====================================================

            clients_per_model = real_training_counter.copy()

            proposta_logs["clients_per_model"].append(clients_per_model)

            print(
                f"📌 Clientes treinados | "
                f"CIFAR: {real_training_counter['cifar']} | "
                f"GTSRB: {real_training_counter['gtsrb']}"
            )


if __name__ == "__main__":

    run_experiment()