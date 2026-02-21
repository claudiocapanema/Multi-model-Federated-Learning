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
NUM_FOLDS = 1
NUM_CLIENTS = 40
ROUNDS = 100
FRAC = 0.3
K_CLIENTS = int(FRAC * NUM_CLIENTS)   # exemplo: 30% no máximo

LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01

DIRICHLET_ALPHA = 0.1
regime = "realistic"
# regime = "benign"
regime = "severe"


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
    "cifar": 0.3,
    "gtsrb": 0.5
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
            "BATTERY_DECAY": 0.09,
            "TIME_MAX": 2.5,
            "LINK_MIN": 0.3,
        },

        "severe": {
            "battery_init": (0.4, 0.8),
            "compute": (0.2, 0.6),
            "link": (0.0, 0.6),
            "BATTERY_DECAY": 0.1,
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
    energy_spent = train_time * BATTERY_DECAY

    client_resources[cid]["battery"] -= energy_spent
    client_resources[cid]["battery"] = max(client_resources[cid]["battery"], 0.0)

    return energy_spent


def update_link(cid):
    low, high = current_regime["link"]
    client_resources[cid]["link"] = np.random.uniform(low, high)


def R(cid, model_name):

    train_time = estimate_training_time(cid, model_name)
    battery_after = client_resources[cid]["battery"] - train_time * BATTERY_DECAY

    return (
        battery_after > 0 and
        train_time <= TIME_MAX and
        client_resources[cid]["link"] >= LINK_MIN
    )


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
        "fallback_rate": [],
        "energy_consumed_round": [],
        "cumulative_energy": [],
        "drained_clients_round": [],
        "avg_battery_remaining": []
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
        filename = f"results/baseline_{model_name}_regime_{regime}_frac_{FRAC}_alpha_{DIRICHLET_ALPHA}.csv"
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

            energy_this_round = 0.0

            print(f"\n🔄 FOLD {fold} | RODADA {rnd}")

            for cid in range(NUM_CLIENTS):
                update_link(cid)

            real_training_counter = {m: 0 for m in global_models}

            # =====================================================
            # 1) Seleção Aleatória de Clientes (Baseline)
            # =====================================================

            # -------------------------------------------------
            # 1) Determinar clientes viáveis por modelo
            # -------------------------------------------------

            viable_cifar_clients = [
                cid for cid in range(NUM_CLIENTS)
                if R(cid, "cifar")
            ]

            viable_gtsrb_clients = [
                cid for cid in range(NUM_CLIENTS)
                if R(cid, "gtsrb")
            ]

            # =====================================================
            # 1) Seleção Aleatória de Clientes (Baseline)
            #    (TODOS os clientes são elegíveis)
            # =====================================================

            all_clients = list(range(NUM_CLIENTS))
            selected_clients = random.sample(
                all_clients,
                min(K_CLIENTS, NUM_CLIENTS)
            )

            half = K_CLIENTS // 2
            clients_cifar = selected_clients[:half]
            clients_gtsrb = selected_clients[half:]

            selected_clients = list(set(clients_cifar + clients_gtsrb))
            if len(selected_clients) == 0:

                print("⚠️ Nenhum cliente viável nesta rodada.")

                # Energia não muda
                energy_this_round = 0.0
                cumulative_energy += 0.0

                # Clientes drenados continuam drenados
                drained_clients = sum(
                    1 for cid in range(NUM_CLIENTS)
                    if client_resources[cid]["battery"] <= 0.0
                )

                avg_battery_remaining = np.mean([
                    client_resources[cid]["battery"]
                    for cid in range(NUM_CLIENTS)
                ])

                rawcs_logs["energy_consumed_round"].append(0.0)
                rawcs_logs["cumulative_energy"].append(cumulative_energy)
                rawcs_logs["drained_clients_round"].append(drained_clients)
                rawcs_logs["avg_battery_remaining"].append(avg_battery_remaining)

                rawcs_logs["viable_clients"].append(0)
                rawcs_logs["viable_pairs"].append(0)
                rawcs_logs["viable_pairs_per_model"].append({
                    "cifar": 0,
                    "gtsrb": 0
                })

                rawcs_logs["fallback_rate"].append(0.0)

                # Avaliação global SEM treino
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

                    row_data = {
                        "algorithm": "baseline",
                        "fold": fold,
                        "round": rnd,
                        "dataset": model_name,
                        "global_acc": global_acc,
                        "clients_selected": 0,
                        "clients_selected_total": 0,
                        "viable_clients_total": 0,
                        "viable_pairs_total": 0,
                        "viable_cifar": 0,
                        "viable_gtsrb": 0,
                        "energy_consumed_round": 0.0,
                        "cumulative_energy": cumulative_energy,
                        "drained_clients_round": drained_clients,
                        "avg_battery_remaining": avg_battery_remaining,
                    }

                    filename = f"results/baseline_{model_name}_regime_{regime}_frac_{FRAC}_alpha_{DIRICHLET_ALPHA}.csv"
                    append_result_to_csv(row_data, filename)

                rawcs_logs["clients_per_model"].append({
                    "cifar": 0,
                    "gtsrb": 0
                })

                continue

            # =====================================================
            # Cálculo de clientes viáveis na rodada
            # =====================================================

            viable_cifar = sum(R(cid, "cifar") for cid in range(NUM_CLIENTS))
            viable_gtsrb = sum(R(cid, "gtsrb") for cid in range(NUM_CLIENTS))

            # Cliente considerado viável se for viável para pelo menos um modelo
            viable_clients_total = sum(
                1 for cid in range(NUM_CLIENTS)
                if R(cid, "cifar") or R(cid, "gtsrb")
            )

            half = K_CLIENTS // 2
            clients_cifar = selected_clients[:half]
            clients_gtsrb = selected_clients[half:]

            # Se K_CLIENTS for ímpar
            if len(clients_cifar) + len(clients_gtsrb) < K_CLIENTS:
                clients_cifar.append(selected_clients[-1])

            model_updates = {
                "cifar": [],
                "gtsrb": []
            }

            real_training_counter = {
                "cifar": 0,
                "gtsrb": 0
            }

            # =====================================================
            # 2) Treino CIFAR (apenas viáveis)
            # =====================================================

            for cid in clients_cifar:

                if not R(cid, "cifar"):
                    continue

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

                train_time = estimate_training_time(cid, "cifar")
                energy_spent = consume_battery(cid, train_time)
                energy_this_round += energy_spent

                acc = evaluate_model(
                    local_model,
                    test_loaders[cid]["cifar"],
                    "CIFAR10"
                )

                client_acc[cid]["cifar"] = acc
                client_loss[cid]["cifar"] = loss

            # =====================================================
            # 3) Treino GTSRB (apenas viáveis)
            # =====================================================

            for cid in clients_gtsrb:

                if not R(cid, "gtsrb"):
                    continue

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

                train_time = estimate_training_time(cid, "gtsrb")
                energy_spent = consume_battery(cid, train_time)
                energy_this_round += energy_spent

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

            # -------------------------------------------------
            # MÉTRICAS ENERGÉTICAS
            # -------------------------------------------------

            cumulative_energy += energy_this_round

            drained_clients = sum(
                1 for cid in range(NUM_CLIENTS)
                if client_resources[cid]["battery"] <= 0.0
            )

            avg_battery_remaining = np.mean([
                client_resources[cid]["battery"]
                for cid in range(NUM_CLIENTS)
            ])

            rawcs_logs["energy_consumed_round"].append(energy_this_round)
            rawcs_logs["cumulative_energy"].append(cumulative_energy)
            rawcs_logs["drained_clients_round"].append(drained_clients)
            rawcs_logs["avg_battery_remaining"].append(avg_battery_remaining)

            rawcs_logs["viable_clients"].append(viable_clients_total)
            rawcs_logs["viable_pairs"].append(viable_cifar + viable_gtsrb)

            rawcs_logs["viable_pairs_per_model"].append({
                "cifar": viable_cifar,
                "gtsrb": viable_gtsrb
            })

            rawcs_logs["avg_battery"].append(
                np.mean([client_resources[c]["battery"] for c in range(NUM_CLIENTS)])
            )

            rawcs_logs["avg_link"].append(
                np.mean([client_resources[c]["link"] for c in range(NUM_CLIENTS)])
            )

            rawcs_logs["avg_cost"].append({
                "cifar": MODEL_COST["cifar"],
                "gtsrb": MODEL_COST["gtsrb"]
            })

            rawcs_logs["avg_train_time"].append({
                "cifar": np.mean([estimate_training_time(c, "cifar") for c in range(NUM_CLIENTS)]),
                "gtsrb": np.mean([estimate_training_time(c, "gtsrb") for c in range(NUM_CLIENTS)])
            })

            rawcs_logs["fallback_rate"].append(0.0)

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
                    "algorithm": "rawcs",
                    "fold": fold,
                    "round": rnd,
                    "dataset": model_name,
                    "global_acc": global_acc,

                    "clients_selected": real_training_counter[model_name],
                    "clients_selected_total": sum(real_training_counter.values()),

                    "viable_clients_total": rawcs_logs["viable_clients"][-1],
                    "viable_pairs_total": rawcs_logs["viable_pairs"][-1],
                    "viable_cifar": rawcs_logs["viable_pairs_per_model"][-1]["cifar"],
                    "viable_gtsrb": rawcs_logs["viable_pairs_per_model"][-1]["gtsrb"],

                    "avg_battery": rawcs_logs["avg_battery"][-1],
                    "avg_link": rawcs_logs["avg_link"][-1],

                    "avg_cost_cifar": rawcs_logs["avg_cost"][-1]["cifar"],
                    "avg_cost_gtsrb": rawcs_logs["avg_cost"][-1]["gtsrb"],

                    "avg_train_time_cifar": rawcs_logs["avg_train_time"][-1]["cifar"],
                    "avg_train_time_gtsrb": rawcs_logs["avg_train_time"][-1]["gtsrb"],

                    "fallback_rate": rawcs_logs["fallback_rate"][-1],
                    "energy_consumed_round": rawcs_logs["energy_consumed_round"][-1],
                    "cumulative_energy": rawcs_logs["cumulative_energy"][-1],
                    "drained_clients_round": rawcs_logs["drained_clients_round"][-1],
                    "avg_battery_remaining": rawcs_logs["avg_battery_remaining"][-1],

                }

                filename = f"results/baseline_{model_name}_regime_{regime}_frac_{FRAC}_alpha_{DIRICHLET_ALPHA}.csv"
                append_result_to_csv(row_data, filename)

            # =====================================================
            # Atualiza clientes por modelo (TREINAMENTO REAL)
            # =====================================================

            clients_per_model = real_training_counter.copy()

            rawcs_logs["clients_per_model"].append(clients_per_model)

            print(
                f"📌 Clientes treinados | "
                f"CIFAR: {real_training_counter['cifar']} | "
                f"GTSRB: {real_training_counter['gtsrb']}"
            )


if __name__ == "__main__":

    run_experiment()