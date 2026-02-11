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

BASE_SEED = 42
NUM_FOLDS = 5
NUM_CLIENTS = 40
ROUNDS = 50

LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01

DIRICHLET_ALPHA = 0.5
regime = "realistic"

# =====================================================
# PARÂMETROS DE UTILIDADE
# =====================================================

ALPHA = 0.4
BETA  = 0.4
GAMMA = 0.2

LAMBDA_ALPHA = 0.5
COLLAPSE_WINDOW = 3

BATTERY_MIN = 0.2

TARGET_ACC = {
    "cifar": 0.75,
    "gtsrb": 0.90
}

MODEL_COST = {
    "cifar": 1.0,
    "gtsrb": 2.0
}

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
rawcs_logs = {}

# =====================================================
# REGIMES EXPERIMENTAIS
# =====================================================

def get_regime(name: str):

    regimes = {

        "benign": {
            "battery_init": (0.8, 1.0),
            "compute": (0.6, 1.0),
            "link": (0.6, 1.0),
            "BATTERY_DECAY": 0.02,
            "TIME_MAX": 3.0,
            "LINK_MIN": 0.3,
        },

        "realistic": {
            "battery_init": (0.6, 1.0),
            "compute": (0.3, 1.0),
            "link": (0.0, 1.0),
            "BATTERY_DECAY": 0.05,
            "TIME_MAX": 2.5,
            "LINK_MIN": 0.3,
        },

        "severe": {
            "battery_init": (0.4, 0.8),
            "compute": (0.2, 0.6),
            "link": (0.0, 0.6),
            "BATTERY_DECAY": 0.08,
            "TIME_MAX": 2.0,
            "LINK_MIN": 0.4,
        }
    }

    return regimes[name]


def apply_regime(name: str):

    global BATTERY_DECAY, TIME_MAX, LINK_MIN, current_regime

    current_regime = get_regime(name)

    BATTERY_DECAY = current_regime["BATTERY_DECAY"]
    TIME_MAX = current_regime["TIME_MAX"]
    LINK_MIN = current_regime["LINK_MIN"]

    for cid in range(NUM_CLIENTS):
        client_resources[cid]["battery"] = np.random.uniform(*current_regime["battery_init"])
        client_resources[cid]["compute"] = np.random.uniform(*current_regime["compute"])
        client_resources[cid]["link"] = np.random.uniform(*current_regime["link"])

    print(f"🧪 Regime aplicado: {name.upper()}")

# =====================================================
# RECURSOS E VIABILIDADE
# =====================================================

def estimate_training_time(cid, model_name):
    return MODEL_COST[model_name] / client_resources[cid]["compute"]


def consume_battery(cid, train_time):
    client_resources[cid]["battery"] -= train_time * BATTERY_DECAY
    client_resources[cid]["battery"] = max(client_resources[cid]["battery"], 0.0)


def update_link(cid):
    low, high = current_regime["link"]
    client_resources[cid]["link"] = np.random.uniform(low, high)


def R(cid, model_name):

    train_time = estimate_training_time(cid, model_name)
    battery_after = client_resources[cid]["battery"] - train_time * BATTERY_DECAY

    return (
        battery_after >= BATTERY_MIN and
        train_time <= TIME_MAX and
        client_resources[cid]["link"] >= LINK_MIN
    )

# =====================================================
# PARTE 5 — UTILIDADE E CONTROLE ADAPTATIVO
# =====================================================

def compute_utility(cid, model_name):

    train_time = estimate_training_time(cid, model_name)

    cost = (
        ALPHA * (1.0 - client_resources[cid]["battery"]) +
        BETA  * min(train_time / TIME_MAX, 1.0) +
        GAMMA * (1.0 - client_resources[cid]["link"])
    )

    acc_gap = max(
        TARGET_ACC[model_name] - client_acc[cid][model_name],
        0.0
    )

    return max((1.0 - cost) * acc_gap, 0.0)


def compute_model_lambda(model_name, round_idx):

    # Round inicial → neutro
    if len(global_acc_history[model_name]) == 0:
        return 1.0

    # Performance deficit
    global_acc = global_acc_history[model_name][-1]
    perf_deficit = max(TARGET_ACC[model_name] - global_acc, 0.0)

    # Participation deficit
    if len(rawcs_logs["clients_per_model"]) == 0:
        part_deficit = 1.0
    else:
        prev_clients = rawcs_logs["clients_per_model"][-1][model_name]
        part_deficit = 1.0 - (prev_clients / NUM_CLIENTS)

    lambda_m = (
        LAMBDA_ALPHA * perf_deficit +
        (1 - LAMBDA_ALPHA) * part_deficit
    )

    return lambda_m


def compute_all_lambdas(round_idx):

    lambdas = {}

    for m in global_models:
        lambdas[m] = compute_model_lambda(m, round_idx)

    total = sum(lambdas.values())

    if total == 0:
        return {m: 1.0 / len(lambdas) for m in lambdas}

    return {m: lambdas[m] / total for m in lambdas}


def check_model_collapse(model_name):

    if len(rawcs_logs["clients_per_model"]) < COLLAPSE_WINDOW:
        return False

    recent = rawcs_logs["clients_per_model"][-COLLAPSE_WINDOW:]
    total_clients = sum(r[model_name] for r in recent)

    return total_clients == 0

# =====================================================
# RESET DO ESTADO GLOBAL (POR FOLD)
# =====================================================

def reset_experiment_state():

    global global_models
    global client_resources
    global client_acc
    global client_loss
    global client_delta_acc
    global rawcs_logs
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
            "battery": np.random.uniform(0.6, 1.0),
            "compute": np.random.uniform(0.3, 1.0),
            "link": 1.0
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

    # -------- Logs RAWCS --------
    rawcs_logs = {
        "viable_pairs": [],
        "viable_clients": [],
        "viable_pairs_per_model": [],
        "clients_per_model": [],
        "avg_battery": [],
        "avg_link": [],
        "avg_cost": [],
        "avg_train_time": [],
        "max_train_time": [],
        "fallback_rate": []
    }

# =====================================================
# PARTE 6 — RAWCS MULTI-MODELO
# =====================================================

def rawcs_multi_model(t):

    lambdas = compute_all_lambdas(t)

    assignments = {}

    viable_pairs = 0
    viable_clients = 0
    fallback_count = 0

    viable_pairs_per_model = {m: 0 for m in global_models}
    costs_per_model = {m: [] for m in global_models}
    train_times_per_model = {m: [] for m in global_models}

    # Atualiza link (volátil)
    for cid in range(NUM_CLIENTS):
        update_link(cid)

    # Decisão cliente → modelo
    for cid in range(NUM_CLIENTS):

        utilities = {}
        client_viable = False

        for model_name in global_models:

            if R(cid, model_name):

                client_viable = True
                viable_pairs += 1
                viable_pairs_per_model[model_name] += 1

                train_time = estimate_training_time(cid, model_name)

                cost = (
                    ALPHA * (1.0 - client_resources[cid]["battery"]) +
                    BETA  * min(train_time / TIME_MAX, 1.0) +
                    GAMMA * (1.0 - client_resources[cid]["link"])
                )

                base_utility = compute_utility(cid, model_name)
                utility = base_utility * lambdas[model_name]

                utilities[model_name] = utility
                costs_per_model[model_name].append(cost)
                train_times_per_model[model_name].append(train_time)

        if client_viable:
            viable_clients += 1

        if utilities:

            models = list(utilities.keys())
            probs = np.array(list(utilities.values()))

            if probs.sum() == 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()

            chosen_model = np.random.choice(models, p=probs)

        else:
            fallback_count += 1
            chosen_model = max(
                global_models.keys(),
                key=lambda m: client_loss[cid][m]
            )

        assignments[cid] = chosen_model

    # -------- Logs --------

    rawcs_logs["viable_pairs"].append(viable_pairs)
    rawcs_logs["viable_clients"].append(viable_clients)
    rawcs_logs["viable_pairs_per_model"].append(viable_pairs_per_model)
    rawcs_logs["fallback_rate"].append(fallback_count / NUM_CLIENTS)

    rawcs_logs["avg_cost"].append({
        m: float(np.mean(costs_per_model[m])) if costs_per_model[m] else None
        for m in global_models
    })

    rawcs_logs["avg_train_time"].append({
        m: float(np.mean(train_times_per_model[m])) if train_times_per_model[m] else None
        for m in global_models
    })

    rawcs_logs["max_train_time"].append({
        m: float(np.max(train_times_per_model[m])) if train_times_per_model[m] else None
        for m in global_models
    })

    rawcs_logs["avg_battery"].append(
        float(np.mean([client_resources[cid]["battery"] for cid in range(NUM_CLIENTS)]))
    )

    rawcs_logs["avg_link"].append(
        float(np.mean([client_resources[cid]["link"] for cid in range(NUM_CLIENTS)]))
    )

    return assignments

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


def run_experiment():
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

            print(f"\n🔄 FOLD {fold} | RODADA {rnd}")

            real_training_counter = {m: 0 for m in global_models}

            # 1) RAWCS
            assignments = rawcs_multi_model(rnd)

            model_updates = {
                "cifar": [],
                "gtsrb": []
            }

            # 2) Treino local
            for cid, model_name in assignments.items():

                if not R(cid, model_name):
                    continue

                real_training_counter[model_name] += 1

                dataset_name = {
                    "cifar": "CIFAR10",
                    "gtsrb": "GTSRB"
                }[model_name]

                local_model = copy.deepcopy(global_models[model_name])

                state_dict, loss, n_samples = client_update(
                    dataset_name,
                    local_model,
                    train_loaders[cid][model_name],
                    epochs=LOCAL_EPOCHS,
                    lr=LR
                )

                model_updates[model_name].append((state_dict, n_samples))

                train_time = estimate_training_time(cid, model_name)
                consume_battery(cid, train_time)

                prev_acc = client_acc[cid][model_name]

                acc = evaluate_model(
                    local_model,
                    test_loaders[cid][model_name],
                    dataset_name
                )

                client_acc[cid][model_name] = acc
                client_loss[cid][model_name] = loss
                client_delta_acc[cid][model_name] = acc - prev_acc

            # 3) FedAvg
            for model_name, updates in model_updates.items():
                if updates:
                    new_state = fedavg(updates)
                    global_models[model_name].load_state_dict(new_state)

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
                    "fold": fold,
                    "round": rnd,
                    "dataset": model_name,
                    "global_acc": global_acc,
                    "viable_pairs": rawcs_logs["viable_pairs"][-1],
                    "viable_clients": rawcs_logs["viable_clients"][-1],
                    "fallback_rate": rawcs_logs["fallback_rate"][-1],
                    "clients_selected": real_training_counter[model_name]
                }

                filename = f"results/proposta_{model_name}_regime_{regime}.csv"
                append_result_to_csv(row_data, filename)

            # Atualiza clientes por modelo
            clients_per_model = {m: 0 for m in global_models}
            for m in assignments.values():
                clients_per_model[m] += 1

            rawcs_logs["clients_per_model"].append(clients_per_model)

            print("🧠 RAWCS | "
                  f"Viable pairs: {rawcs_logs['viable_pairs'][-1]} | "
                  f"Viable clients: {rawcs_logs['viable_clients'][-1]} | "
                  f"Fallback rate: {rawcs_logs['fallback_rate'][-1]:.2f}")

            print("🧠 Viable pairs por modelo:",
                  rawcs_logs["viable_pairs_per_model"][-1])

            print("⚙️ Custo médio por modelo:",
                  rawcs_logs["avg_cost"][-1])

            print("⏱️ Tempo médio por modelo:",
                  rawcs_logs["avg_train_time"][-1])

if __name__ == "__main__":

    run_experiment()