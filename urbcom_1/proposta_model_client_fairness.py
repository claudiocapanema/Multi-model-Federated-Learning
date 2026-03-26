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

global FAIR_RESOURCE_BUDGET
global loss_cifar_norm, loss_gtsrb_norm
global data_cifar_norm, data_gtsrb_norm

MODEL_COST = {
    "cifar": {
        "flops_per_sample": 5e6,   # modelo leve
    },
    "gtsrb": {
        "flops_per_sample": 1.2e7, # modelo mais pesado
    }
}

# =====================================================
# FAIRNESS WEIGHTS (GLOBAL EXPERIMENT CONFIG)
# =====================================================

LAMBDA_CAPACITY = 0.3
LAMBDA_INTRA = 0.3

assert LAMBDA_CAPACITY + LAMBDA_INTRA <= 1.0

# =====================================================
# FAIR RESOURCE BUDGET
# =====================================================

LIGHT_MODEL = min(
    MODEL_COST,
    key=lambda m: MODEL_COST[m]["flops_per_sample"]
)

# permitir aproximadamente K_CLIENTS treinamentos por rodada
avg_speed = 0.65

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
logs = {}

# =====================================================
# FAIRNESS STATES
# =====================================================

client_resource_usage = {}

# =====================================================
# TRAINING TIME (REALISTA + DIRICHLET)
# =====================================================

def estimate_training_time(cid, model_name):

    speed = client_resources[cid]["speed"]

    if model_name == "cifar":
        data_size = client_resources[cid]["data_size_cifar"]
    else:
        data_size = client_resources[cid]["data_size_gtsrb"]

    flops_per_sample = MODEL_COST[model_name]["flops_per_sample"]

    LOCAL_EPOCHS = 1

    # custo computacional total
    total_flops = flops_per_sample * data_size * LOCAL_EPOCHS

    # converter para tempo (normalizado)
    train_time = total_flops / (1e7 * speed)

    # ruído realista (CPU jitter)
    noise = np.random.uniform(0.9, 1.1)

    return train_time * noise

# =====================================================
# RESET DO ESTADO GLOBAL (POR FOLD)
# =====================================================

def reset_experiment_state(train_loaders):

    global global_models
    global client_resources
    global client_acc
    global client_loss
    global client_delta_acc
    global logs
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
    # =====================================================
    # CLIENT RESOURCES (ALINHADO COM DIRICHLET)
    # =====================================================

    # =====================================================
    # CLIENT RESOURCES (CORRETO COM DATALOADER)
    # =====================================================

    client_resources = {
        cid: {
            "speed": float(np.clip(np.random.lognormal(-0.3, 0.6), 0.2, 2.0)),

            "data_size_cifar": len(train_loaders[cid]["cifar"].dataset),
            "data_size_gtsrb": len(train_loaders[cid]["gtsrb"].dataset),
        }
        for cid in range(NUM_CLIENTS)
    }

    # =====================================================
    # SANITY CHECK (DEBUG)
    # =====================================================

    for cid in range(NUM_CLIENTS):
        assert client_resources[cid]["data_size_cifar"] >= 0
        assert client_resources[cid]["data_size_gtsrb"] >= 0

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
    logs = {
        "clients_per_model": [],

        "resource_usage_cifar": [],
        "resource_usage_gtsrb": [],

        # fairness padronizadas (↑ melhor)
        "inter_model_fairness": [],
        "inter_client_fairness": [],
        "intra_client_fairness": []
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
        filename = (
    f"results/proposta_{model_name}"
    f"_frac_{FRAC}"
    f"_alpha_{DIRICHLET_ALPHA}"
    f"_lambdaCap_{LAMBDA_CAPACITY}"
    f"_lambdaIntra_{LAMBDA_INTRA}.csv"
)
        if os.path.exists(filename):
            os.remove(filename)

def compute_inter_client_fairness_with_delta(
        cid,
        train_time,
        loss_cifar_norm,
        loss_gtsrb_norm,
        data_cifar_norm,
        data_gtsrb_norm
):

    utilization = []
    eps = 1e-8

    for c in range(NUM_CLIENTS):

        capacity = client_resources[c]["speed"]

        usage = (
            client_resource_usage[c]["cifar"] +
            client_resource_usage[c]["gtsrb"]
        )

        # 👉 simula adição para o cliente candidato
        if c == cid:
            usage += train_time

        # -------- UTILITY --------
        loss_cifar = client_loss[c]["cifar"]
        loss_gtsrb = client_loss[c]["gtsrb"]

        loss_cifar = loss_cifar if np.isfinite(loss_cifar) else 1.0
        loss_gtsrb = loss_gtsrb if np.isfinite(loss_gtsrb) else 1.0

        data_cifar = client_resources[c]["data_size_cifar"]
        data_gtsrb = client_resources[c]["data_size_gtsrb"]

        # -------- UTILITY NORMALIZADA --------

        utility = (
                data_cifar_norm[c] * loss_cifar_norm[c] +
                data_gtsrb_norm[c] * loss_gtsrb_norm[c]
        )

        value = usage / (capacity * (utility + eps))
        utilization.append(value)

    if len(utilization) > 0:
        num = (sum(utilization) ** 2)
        den = NUM_CLIENTS * sum(u ** 2 for u in utilization)
        return num / den if den > 0 else 1.0
    else:
        return 1.0

def compute_inter_client_fairness(loss_cifar_norm,
        loss_gtsrb_norm,
        data_cifar_norm,
        data_gtsrb_norm):

    utilization = []
    eps = 1e-8

    for cid in range(NUM_CLIENTS):

        capacity = client_resources[cid]["speed"]

        usage = (
            client_resource_usage[cid]["cifar"] +
            client_resource_usage[cid]["gtsrb"]
        )

        loss_cifar = client_loss[cid]["cifar"]
        loss_gtsrb = client_loss[cid]["gtsrb"]

        loss_cifar = loss_cifar if np.isfinite(loss_cifar) else 1.0
        loss_gtsrb = loss_gtsrb if np.isfinite(loss_gtsrb) else 1.0

        data_cifar = client_resources[cid]["data_size_cifar"]
        data_gtsrb = client_resources[cid]["data_size_gtsrb"]

        utility = (
                data_cifar_norm[cid] * loss_cifar_norm[cid] +
                data_gtsrb_norm[cid] * loss_gtsrb_norm[cid]
        )

        value = usage / (capacity * (utility + eps))
        utilization.append(value)

    if len(utilization) > 0:
        num = (sum(utilization) ** 2)
        den = NUM_CLIENTS * sum(u ** 2 for u in utilization)
        return num / den if den > 0 else 1.0
    else:
        return 1.0

def run_experiment():

    # =====================================================
    # LIMPEZA INICIAL
    # Remove arquivos CSV antigos para evitar mistura de resultados
    # =====================================================
    clear_previous_results()

    # =====================================================
    # LOOP SOBRE FOLDS (cross-validation ou repetição com seeds)
    # =====================================================
    for fold in range(NUM_FOLDS):

        print("\n===================================================")
        print(f"🚀 INICIANDO FOLD {fold}")
        print("===================================================")

        # -------------------------------------------------
        # Definição de seed reprodutível por fold
        # -------------------------------------------------
        fold_seed = BASE_SEED + fold * 1000

        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        # -------------------------------------------------
        # Reset completo do estado global:
        # - modelos
        # - métricas
        # - recursos acumulados
        # -------------------------------------------------

        # =====================================================
        # CRIAÇÃO DOS DATALOADERS POR CLIENTE
        # Cada cliente recebe uma partição não-iid (Dirichlet)
        # =====================================================
        train_loaders = defaultdict(dict)
        test_loaders = defaultdict(dict)

        for cid in range(NUM_CLIENTS):

            # ---------- CIFAR ----------
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

            # ---------- GTSRB ----------
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

        reset_experiment_state(train_loaders)

        global FAIR_RESOURCE_BUDGET

        avg_data_cifar = np.mean([client_resources[c]["data_size_cifar"] for c in range(NUM_CLIENTS)])
        avg_data_gtsrb = np.mean([client_resources[c]["data_size_gtsrb"] for c in range(NUM_CLIENTS)])

        avg_data = min(avg_data_cifar, avg_data_gtsrb)

        FAIR_RESOURCE_BUDGET = (
                K_CLIENTS *
                MODEL_COST[LIGHT_MODEL]["flops_per_sample"] *
                avg_data /
                (1e7 * avg_speed)
        )

        # =====================================================
        # LOOP PRINCIPAL DE RODADAS FEDERADAS
        # =====================================================
        for rnd in range(1, ROUNDS + 1):

            # -------------------------------------------------
            # Seed por rodada (garante reprodutibilidade completa)
            # -------------------------------------------------
            round_seed = fold_seed + rnd
            random.seed(round_seed)
            np.random.seed(round_seed)
            torch.manual_seed(round_seed)

            # =====================================================
            # CACHE DE TEMPO DE TREINO (CRÍTICO)
            # =====================================================
            training_time_cache = {}

            def get_train_time(cid, model_name):
                if (cid, model_name) not in training_time_cache:
                    training_time_cache[(cid, model_name)] = estimate_training_time(cid, model_name)
                return training_time_cache[(cid, model_name)]

            # =====================================================
            # Estrutura para armazenar updates locais (FedAvg)
            # =====================================================
            model_updates = {
                "cifar": [],
                "gtsrb": []
            }

            print(f"\n🔄 FOLD {fold} | RODADA {rnd}")

            # Contador REAL de clientes que treinaram
            real_training_counter = {m: 0 for m in global_models}

            # =====================================================
            # NORMALIZAÇÃO POR DATASET (CRÍTICO)
            # =====================================================

            loss_cifar_vals = []
            loss_gtsrb_vals = []
            data_cifar_vals = []
            data_gtsrb_vals = []

            for cid in range(NUM_CLIENTS):
                lc = client_loss[cid]["cifar"]
                lg = client_loss[cid]["gtsrb"]

                lc = lc if np.isfinite(lc) else 1.0
                lg = lg if np.isfinite(lg) else 1.0

                loss_cifar_vals.append(lc)
                loss_gtsrb_vals.append(lg)

                data_cifar_vals.append(client_resources[cid]["data_size_cifar"])
                data_gtsrb_vals.append(client_resources[cid]["data_size_gtsrb"])

            def minmax(arr):
                mn, mx = min(arr), max(arr)
                if mx - mn < 1e-8:
                    return [1.0 for _ in arr]
                return [(x - mn) / (mx - mn + 1e-8) for x in arr]

            loss_cifar_norm = minmax(loss_cifar_vals)
            loss_gtsrb_norm = minmax(loss_gtsrb_vals)

            data_cifar_norm = minmax(data_cifar_vals)
            data_gtsrb_norm = minmax(data_gtsrb_vals)

            # =====================================================
            # 1) SELEÇÃO DE CLIENTES (CORE DO MÉTO DO)
            # Balanced Resource + Fairness entre clientes
            # =====================================================

            clients_cifar = []
            clients_gtsrb = []

            # controle de budget por rodada
            resource_usage = {
                "cifar": 0.0,
                "gtsrb": 0.0
            }

            # embaralha clientes (ordem aleatória)
            all_clients = list(range(NUM_CLIENTS))
            random.shuffle(all_clients)

            # =====================================================
            # LOOP DE DECISÃO POR CLIENTE (NOVO — MAX FAIRNESS)
            # =====================================================
            # =====================================================
            # LOOP DE DECISÃO POR CLIENTE (FIXED — STABLE FAIRNESS)
            # =====================================================

            total_cifar_usage = sum(client_resource_usage[c]["cifar"] for c in range(NUM_CLIENTS))
            total_gtsrb_usage = sum(client_resource_usage[c]["gtsrb"] for c in range(NUM_CLIENTS))

            max_total_clients = K_CLIENTS * 2  # garante mínimo de updates

            for cid in all_clients:

                # 🔒 garante limite de clientes (evita under-training)
                if len(clients_cifar) + len(clients_gtsrb) >= max_total_clients:
                    break

                candidates = []

                # =========================
                # TESTA CIFAR
                # =========================
                train_time = get_train_time(cid, "cifar")

                if resource_usage["cifar"] + train_time <= FAIR_RESOURCE_BUDGET:

                    new_usage = total_cifar_usage + train_time
                    other_usage = total_gtsrb_usage
                    total = new_usage + other_usage

                    if total > 0:
                        imbalance = abs(new_usage - other_usage) / total
                        inter_model_fairness = 1.0 - imbalance
                    else:
                        inter_model_fairness = 1.0

                    # -------- Intra-client
                    client_cifar = client_resource_usage[cid]["cifar"] + train_time
                    client_gtsrb = client_resource_usage[cid]["gtsrb"]
                    total_client = client_cifar + client_gtsrb

                    if total_client > 0:
                        intra_imbalance = abs(client_cifar - client_gtsrb) / total_client
                        intra_client_fairness = 1.0 - intra_imbalance
                    else:
                        intra_client_fairness = 1.0

                    score = (
                            (1 - LAMBDA_CAPACITY - LAMBDA_INTRA) * inter_model_fairness
                            + LAMBDA_CAPACITY * inter_client_fairness
                            + LAMBDA_INTRA * intra_client_fairness
                    )

                    candidates.append(("cifar", score, train_time))

                # =========================
                # TESTA GTSRB
                # =========================
                train_time = get_train_time(cid, "gtsrb")

                if resource_usage["gtsrb"] + train_time <= FAIR_RESOURCE_BUDGET:

                    new_usage = total_gtsrb_usage + train_time
                    other_usage = total_cifar_usage
                    total = new_usage + other_usage

                    if total > 0:
                        imbalance = abs(new_usage - other_usage) / total
                        inter_model_fairness = 1.0 - imbalance
                    else:
                        inter_model_fairness = 1.0

                    # -------- Inter-client fairness (MULTI-DIMENSIONAL) --------
                    inter_client_fairness = compute_inter_client_fairness_with_delta(
                        cid,
                        train_time,
                        loss_cifar_norm,
                        loss_gtsrb_norm,
                        data_cifar_norm,
                        data_gtsrb_norm
                    )

                    client_cifar = client_resource_usage[cid]["cifar"]
                    client_gtsrb = client_resource_usage[cid]["gtsrb"] + train_time
                    total_client = client_cifar + client_gtsrb

                    if total_client > 0:
                        intra_imbalance = abs(client_cifar - client_gtsrb) / total_client
                        intra_client_fairness = 1.0 - intra_imbalance
                    else:
                        intra_client_fairness = 1.0

                    score = (
                            (1 - LAMBDA_CAPACITY - LAMBDA_INTRA) * inter_model_fairness
                            + LAMBDA_CAPACITY * inter_client_fairness
                            + LAMBDA_INTRA * intra_client_fairness
                    )

                    candidates.append(("gtsrb", score, train_time))

                if not candidates:
                    continue

                # =========================
                # ESCOLHA ÓTIMA
                # =========================
                chosen_model, _, train_time = max(candidates, key=lambda x: x[1])

                # -------- atualiza uso global
                if chosen_model == "cifar":
                    total_cifar_usage += train_time
                    clients_cifar.append(cid)
                else:
                    total_gtsrb_usage += train_time
                    clients_gtsrb.append(cid)

                # -------- atualiza recursos
                resource_usage[chosen_model] += train_time
                client_resource_usage[cid][chosen_model] += train_time

            # =====================================================
            # 2) TREINAMENTO LOCAL — CIFAR
            # =====================================================
            for cid in clients_cifar:

                train_time = get_train_time(cid, "cifar")
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

            # =====================================================
            # 3) TREINAMENTO LOCAL — GTSRB
            # =====================================================
            for cid in clients_gtsrb:

                train_time = get_train_time(cid, "gtsrb")
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

            # =====================================================
            # 4) AGREGAÇÃO FEDAVG
            # =====================================================
            for model_name, updates in model_updates.items():
                if len(updates) > 0:
                    new_state = fedavg(updates)
                    global_models[model_name].load_state_dict(new_state)

            # =====================================================
            # 5) MÉTRICAS DE FAIRNESS (PADRONIZADAS ↑)
            # =====================================================

            # -------- Inter-model fairness --------
            resource_cifar = sum(client_resource_usage[c]["cifar"] for c in range(NUM_CLIENTS))
            resource_gtsrb = sum(client_resource_usage[c]["gtsrb"] for c in range(NUM_CLIENTS))

            total_resource = resource_cifar + resource_gtsrb

            if total_resource > 0:
                imbalance = abs(resource_cifar - resource_gtsrb) / total_resource
                inter_model_fairness = 1.0 - imbalance
            else:
                inter_model_fairness = 1.0

            # -------- Inter-client fairness (Jain) --------
            inter_client_fairness = compute_inter_client_fairness(
                loss_cifar_norm,
                loss_gtsrb_norm,
                data_cifar_norm,
                data_gtsrb_norm
            )

            # -------- Intra-client fairness --------
            imbalances = []

            for cid in range(NUM_CLIENTS):

                u_cifar = client_resource_usage[cid]["cifar"]
                u_gtsrb = client_resource_usage[cid]["gtsrb"]

                total = u_cifar + u_gtsrb

                if total > 0:
                    imbalance = abs(u_cifar - u_gtsrb) / total
                    imbalances.append(imbalance)

            if len(imbalances) > 0:
                intra_client_fairness = 1.0 - (sum(imbalances) / len(imbalances))
            else:
                intra_client_fairness = 1.0

            # =====================================================
            # 6) LOGS DE RECURSOS E FAIRNESS
            # =====================================================
            resource_cifar_real = sum(
                estimate_training_time(cid, "cifar")
                for cid in clients_cifar
            )

            resource_gtsrb_real = sum(
                estimate_training_time(cid, "gtsrb")
                for cid in clients_gtsrb
            )

            logs["resource_usage_cifar"].append(resource_cifar_real)
            logs["resource_usage_gtsrb"].append(resource_gtsrb_real)

            logs["inter_model_fairness"].append(inter_model_fairness)
            logs["inter_client_fairness"].append(inter_client_fairness)
            logs["intra_client_fairness"].append(intra_client_fairness)
            # =====================================================
            # 7) AVALIAÇÃO GLOBAL
            # =====================================================
            # =====================================================
            # 7) AVALIAÇÃO GLOBAL
            # =====================================================
            for model_name, model in global_models.items():
                dataset_name = {
                    "cifar": "CIFAR10",
                    "gtsrb": "GTSRB"
                }[model_name]

                global_acc = evaluate_global_model(
                    model,
                    {cid: test_loaders[cid][model_name] for cid in range(NUM_CLIENTS)},
                    dataset_name
                )

                global_acc_history[model_name].append(global_acc)

                # 🔥 PRINT DA ACURÁCIA POR RODADA
                print(
                    f"📊 Rodada {rnd:03d} | {model_name.upper()} Accuracy: {global_acc:.4f}"
                )

                # salvar linha no CSV
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

                    "resource_usage_cifar": logs["resource_usage_cifar"][-1],
                    "resource_usage_gtsrb": logs["resource_usage_gtsrb"][-1],

                    "inter_model_fairness": logs["inter_model_fairness"][-1],
                    "inter_client_fairness": logs["inter_client_fairness"][-1],
                    "intra_client_fairness": logs["intra_client_fairness"][-1],
                }

                filename = (
                    f"results/proposta_{model_name}"
                    f"_frac_{FRAC}"
                    f"_alpha_{DIRICHLET_ALPHA}"
                    f"_lambdaCap_{LAMBDA_CAPACITY}"
                    f"_lambdaIntra_{LAMBDA_INTRA}.csv"
                )

                append_result_to_csv(row_data, filename)

            # =====================================================
            # 8) LOG FINAL DA RODADA
            # =====================================================
            clients_per_model = real_training_counter.copy()
            logs["clients_per_model"].append(clients_per_model)

            print(
                f"📌 Clientes treinados | "
                f"CIFAR: {real_training_counter['cifar']} | "
                f"GTSRB: {real_training_counter['gtsrb']}"
            )


if __name__ == "__main__":

    run_experiment()