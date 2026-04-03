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

# =====================================================
# MODEL COST (FLOPs POR AMOSTRA)
# =====================================================

MODEL_COST_SETUPS = {

    # 🔹 Setup atual (≈2.4x)
    "cost_2_4x": {
        "cifar": {
            "flops_per_sample": 5e6,
        },
        "gtsrb": {
            "flops_per_sample": 1.2e7,
        }
    },

    # 🔹 Novo setup (4x)
    "cost_4x": {
        "cifar": {
            "flops_per_sample": 5e6,
        },
        "gtsrb": {
            "flops_per_sample": 2.0e7,
        }
    }
}

# =====================================================
# SELECT COST SETUP
# =====================================================

COST_SETUP_NAME = "cost_2_4x"  # 🔥 troque aqui
COST_SETUP_NAME = "cost_4x"  # 🔥 troque aqui


MODEL_COST = MODEL_COST_SETUPS[COST_SETUP_NAME]

# =====================================================
# FAIRNESS WEIGHTS (GLOBAL EXPERIMENT CONFIG)
# =====================================================

LAMBDA_CAPACITY = 0.3
LAMBDA_INTRA = 0.3

assert LAMBDA_CAPACITY + LAMBDA_INTRA <= 1.0

# =====================================================
# OORT STATES
# =====================================================

client_last_round = {
    cid: {"cifar": 1, "gtsrb": 1}
    for cid in range(NUM_CLIENTS)
}

client_duration = {
    cid: {"cifar": 1.0, "gtsrb": 1.0}
    for cid in range(NUM_CLIENTS)
}
client_utility = {cid: 0.0 for cid in range(NUM_CLIENTS)}

explored_clients = set()

# Pacer
oort_T = 1.0
oort_delta = 0.2
oort_window = 5

# histórico de utilidade por rodada
oort_round_utilities = []

# parâmetros
EPSILON = 0.2
ALPHA_OORT = 1.0
CUTOFF_PERCENTILE = 0.95

# =====================================================
# FAIR RESOURCE BUDGET
# =====================================================

LIGHT_MODEL = min(
    MODEL_COST,
    key=lambda m: MODEL_COST[m]["flops_per_sample"]
)

# =====================================================
# COST RATIO (AUTOMÁTICO)
# =====================================================

def compute_cost_ratio():
    cifar_cost = MODEL_COST["cifar"]["flops_per_sample"]
    gtsrb_cost = MODEL_COST["gtsrb"]["flops_per_sample"]

    ratio = gtsrb_cost / cifar_cost

    # 🔹 versão para nome de pasta (segura)
    ratio_str_file = f"{ratio:.1f}x"

    # 🔹 versão para exibição (PT-BR)
    ratio_str_br = f"{ratio:.1f}".replace(".", ",") + "x"

    return ratio, ratio_str_file, ratio_str_br


COST_RATIO, COST_RATIO_STR_FILE, COST_RATIO_STR_BR = compute_cost_ratio()

# =====================================================
# DIRETÓRIO DE RESULTADOS (COM RATIO)
# =====================================================

RESULTS_DIR = f"results/gtsrb_{COST_RATIO_STR_FILE}_cifar/"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

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
# RECURSOS E VIABILIDADE
# =====================================================

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

def compute_oort_utility(cid, rnd):

    # -------- STATISTICAL UTILITY --------
    U = client_utility[cid]

    # -------- UNCERTAINTY BONUS --------
    last = client_last_round[cid]
    bonus = np.sqrt(0.1 * np.log(rnd + 1) / (last + 1))

    U = U + bonus

    # -------- SYSTEM UTILITY --------
    duration = client_duration[cid]

    if duration > oort_T:
        U = U * (oort_T / duration) ** ALPHA_OORT

    return U

def update_pacer(oort_round_utilities, oort_T, delta, window):

    if len(oort_round_utilities) < 2 * window:
        return oort_T

    prev = sum(oort_round_utilities[-2*window:-window])
    curr = sum(oort_round_utilities[-window:])

    if prev > curr:
        oort_T += delta

    return oort_T

def oort_multi_model_selection(
    all_clients,
    models,   # ["cifar", "gtsrb"]
    K,
    rnd,
    epsilon,
    oort_T,
    alpha,
    client_loss,
    client_resources,
    client_last_round,
    client_duration,
    explored_pairs   # (cid, model)
):

    candidates = []

    # =====================================================
    # 1) COMPUTE UTILITY POR (CLIENT, MODEL)
    # =====================================================
    for cid in all_clients:
        for m in models:

            if m == "cifar":
                loss = client_loss[cid]["cifar"]
                data = client_resources[cid]["data_size_cifar"]
            else:
                loss = client_loss[cid]["gtsrb"]
                data = client_resources[cid]["data_size_gtsrb"]

            loss = loss if np.isfinite(loss) else 1.0
            U = data * abs(loss)

            # -------- uncertainty --------
            last = client_last_round[cid][m]
            bonus = np.sqrt(0.1 * np.log(rnd + 1) / (last + 1))
            U += bonus

            # -------- system --------
            duration = client_duration[cid][m]
            if duration > oort_T:
                U *= (oort_T / duration) ** alpha

            candidates.append((cid, m, U))

    # =====================================================
    # 2) EXPLOITATION
    # =====================================================
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    exploit_K = int((1 - epsilon) * K)
    exploit_K = max(1, exploit_K)

    cutoff_value = candidates[min(exploit_K-1, len(candidates)-1)][2]
    threshold = 0.95 * cutoff_value

    W = [c for c in candidates if c[2] >= threshold]

    # sampling proporcional
    probs = np.array([c[2] for c in W])
    probs = probs / probs.sum()

    exploit_selected = list(np.random.choice(
        len(W),
        size=min(len(W), exploit_K),
        replace=False,
        p=probs
    ))

    exploit_pairs = [W[i] for i in exploit_selected]

    # =====================================================
    # 3) EXPLORATION
    # =====================================================
    unexplored = [
        (cid, m)
        for cid in all_clients
        for m in models
        if (cid, m) not in explored_pairs
    ]

    explore_K = K - len(exploit_pairs)

    explore_pairs = []

    if len(unexplored) > 0 and explore_K > 0:

        speeds = np.array([
            1.0 / (client_duration[cid][m] + 1e-6)
            for cid, m in unexplored
        ])

        probs = speeds / speeds.sum()

        idx = np.random.choice(
            len(unexplored),
            size=min(len(unexplored), explore_K),
            replace=False,
            p=probs
        )

        explore_pairs = [
            (cid, m, 0.0)  # utility dummy
            for cid, m in [unexplored[i] for i in idx]
        ]

    all_pairs = exploit_pairs + explore_pairs  # ainda tem (cid, m, u)

    used = set()
    final = []

    for cid, m, u in all_pairs:
        if cid not in used:
            final.append((cid, m))
            used.add(cid)

    return final[:K]

# =====================================================
# RESET DO ESTADO GLOBAL (POR FOLD)
# =====================================================

def reset_experiment_state(train_loaders):

    global global_models
    global client_resources
    global client_acc
    global client_loss
    global client_delta_acc
    global fedbalancer_logs
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
    global logs

    logs = {
        "clients_per_model": [],

        "resource_usage_cifar": [],
        "resource_usage_gtsrb": [],

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
            f"{RESULTS_DIR}/oort_{model_name}"
            f"_frac_{FRAC}"
            f"_alpha_{DIRICHLET_ALPHA}"
            f"_lambdaCap_{LAMBDA_CAPACITY}"
            f"_lambdaIntra_{LAMBDA_INTRA}.csv"
        )
        if os.path.exists(filename):
            os.remove(filename)

def compute_inter_client_fairness(
        loss_cifar_norm,
        loss_gtsrb_norm,
        data_cifar_norm,
        data_gtsrb_norm):

    utilization = []
    eps = 1e-8

    for cid in range(NUM_CLIENTS):

        # -------- CAPACIDADE --------
        capacity = client_resources[cid]["speed"]

        # -------- USO ACUMULADO --------
        usage = (
            client_resource_usage[cid]["cifar"] +
            client_resource_usage[cid]["gtsrb"]
        )

        # -------- UTILITY NORMALIZADA (ALINHADA COM PROPOSTA) --------
        utility = (
            data_cifar_norm[cid] * loss_cifar_norm[cid] +
            data_gtsrb_norm[cid] * loss_gtsrb_norm[cid]
        )

        # evitar divisão por zero
        value = usage / (capacity * (utility + eps))

        utilization.append(value)

    # -------- ÍNDICE DE JAIN --------
    if len(utilization) > 0:
        num = (sum(utilization) ** 2)
        den = NUM_CLIENTS * sum(u ** 2 for u in utilization)
        return num / den if den > 0 else 1.0
    else:
        return 1.0

def compute_normalized_metrics():

    loss_cifar_vals = []
    loss_gtsrb_vals = []
    data_cifar_vals = []
    data_gtsrb_vals = []

    for cid in range(NUM_CLIENTS):

        lc = client_loss[cid]["cifar"]
        lg = client_loss[cid]["gtsrb"]

        # tratar valores inválidos
        lc = lc if np.isfinite(lc) else 1.0
        lg = lg if np.isfinite(lg) else 1.0

        loss_cifar_vals.append(lc)
        loss_gtsrb_vals.append(lg)

        data_cifar_vals.append(client_resources[cid]["data_size_cifar"])
        data_gtsrb_vals.append(client_resources[cid]["data_size_gtsrb"])

    # -------- FUNÇÃO MIN-MAX --------
    def minmax(arr):
        mn, mx = min(arr), max(arr)

        if mx - mn < 1e-8:
            return [1.0 for _ in arr]

        return [(x - mn) / (mx - mn + 1e-8) for x in arr]

    # -------- NORMALIZAÇÃO --------
    loss_cifar_norm = minmax(loss_cifar_vals)
    loss_gtsrb_norm = minmax(loss_gtsrb_vals)

    data_cifar_norm = minmax(data_cifar_vals)
    data_gtsrb_norm = minmax(data_gtsrb_vals)

    return (
        loss_cifar_norm,
        loss_gtsrb_norm,
        data_cifar_norm,
        data_gtsrb_norm
    )

def compute_oort_utility_model(cid, model_name, rnd):

    # -------- STATISTICAL UTILITY --------
    if model_name == "cifar":
        loss = client_loss[cid]["cifar"]
        data = client_resources[cid]["data_size_cifar"]
    else:
        loss = client_loss[cid]["gtsrb"]
        data = client_resources[cid]["data_size_gtsrb"]

    loss = loss if np.isfinite(loss) else 1.0
    U = data * abs(loss)

    # -------- UNCERTAINTY BONUS (POR MODELO) --------
    last = client_last_round[cid][model_name]
    bonus = np.sqrt(0.1 * np.log(rnd + 1) / (last + 1))
    U += bonus

    # -------- SYSTEM COST (POR MODELO) --------
    duration = client_duration[cid][model_name]

    if duration > oort_T:
        U = U * (oort_T / duration) ** ALPHA_OORT

    return U

def oort_select_participants(
    all_clients,
    K,
    rnd,
    epsilon,
    oort_T,
    alpha,
    client_utility,
    client_last_round,
    client_duration,
    explored_clients
):

    utilities = {}

    # =====================================================
    # 1) UPDATE + COMPUTE UTILITY (explored clients only)
    # =====================================================
    for cid in explored_clients:

        U = client_utility[cid]

        # -------- uncertainty bonus --------
        last = client_last_round[cid]
        bonus = np.sqrt(0.1 * np.log(rnd + 1) / (last + 1))
        U += bonus

        # -------- system penalty --------
        duration = client_duration[cid]
        if duration > oort_T:
            U *= (oort_T / duration) ** alpha

        utilities[cid] = U

    # =====================================================
    # 2) EXPLOITATION
    # =====================================================

    # ordena por utility
    sorted_clients = sorted(utilities, key=utilities.get, reverse=True)

    exploit_K = int((1 - epsilon) * K)
    exploit_K = max(1, exploit_K)

    if len(sorted_clients) > 0:

        # cutoff (c%)
        cutoff_value = utilities[sorted_clients[min(exploit_K - 1, len(sorted_clients)-1)]]
        threshold = 0.95 * cutoff_value

        # pool W
        W = [cid for cid in sorted_clients if utilities[cid] >= threshold]

        # -------- sampling probabilístico --------
        probs = np.array([utilities[cid] for cid in W])
        probs = probs / probs.sum()

        exploit_selected = list(np.random.choice(
            W,
            size=min(len(W), exploit_K),
            replace=False,
            p=probs
        ))

    else:
        exploit_selected = []

    # =====================================================
    # 3) EXPLORATION (UNEXPLORED CLIENTS)
    # =====================================================
    unexplored = list(set(all_clients) - explored_clients)

    explore_K = K - len(exploit_selected)

    if len(unexplored) > 0 and explore_K > 0:

        # -------- por speed (paper sugere isso) --------
        speeds = np.array([
            1.0 / (client_duration[cid] + 1e-6)
            for cid in unexplored
        ])

        probs = speeds / speeds.sum()

        explore_selected = list(np.random.choice(
            unexplored,
            size=min(len(unexplored), explore_K),
            replace=False,
            p=probs
        ))

    else:
        explore_selected = []

    # =====================================================
    # 4) FINAL
    # =====================================================
    selected = exploit_selected + explore_selected

    # update explored
    for cid in selected:
        explored_clients.add(cid)

    return selected

def compute_intra_client_fairness_utility(
    loss_cifar_norm,
    loss_gtsrb_norm,
    data_cifar_norm,
    data_gtsrb_norm,
    beta=1.0,
    utility_smoothing=0.1,
    eps=1e-8
):
    """
    Intra-client fairness com awareness de utility.

    Parâmetros:
    - beta: controla peso da utility (0 = ignora utility, 1 = total)
    - utility_smoothing: suavização para evitar divisão por zero
    - eps: estabilidade numérica
    """

    vals = []

    for cid in range(NUM_CLIENTS):

        # -------- USAGE --------
        u_cifar = client_resource_usage[cid]["cifar"]
        u_gtsrb = client_resource_usage[cid]["gtsrb"]

        # -------- UTILITY --------
        util_cifar = data_cifar_norm[cid] * loss_cifar_norm[cid]
        util_gtsrb = data_gtsrb_norm[cid] * loss_gtsrb_norm[cid]

        # -------- SMOOTHING (ESSENCIAL) --------
        util_cifar = utility_smoothing + (1 - utility_smoothing) * util_cifar
        util_gtsrb = utility_smoothing + (1 - utility_smoothing) * util_gtsrb

        # -------- CONTROLE DE INFLUÊNCIA --------
        util_cifar = util_cifar ** beta
        util_gtsrb = util_gtsrb ** beta

        # -------- NORMALIZAÇÃO POR UTILITY --------
        v1 = u_cifar / (util_cifar + eps)
        v2 = u_gtsrb / (util_gtsrb + eps)

        total = v1 + v2

        if total == 0:
            continue

        # -------- JAIN --------
        num = total ** 2
        den = 2 * (v1 ** 2 + v2 ** 2)

        vals.append(num / den if den > 0 else 1.0)

    return float(np.mean(vals)) if vals else 1.0

def run_experiment():

    global oort_T

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

        explored_pairs = set()

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

            # =====================================================
            # OORT MULTI-MODEL SELECTION
            # =====================================================

            # =====================================================
            # OORT MULTI-MODEL SELECTION (CORRIGIDO)
            # =====================================================

            # -------- construir candidatos --------
            selected_pairs = oort_multi_model_selection(
                all_clients=list(range(NUM_CLIENTS)),
                models=["cifar", "gtsrb"],
                K=K_CLIENTS,
                rnd=rnd,
                epsilon=EPSILON,
                oort_T=oort_T,
                alpha=ALPHA_OORT,
                client_loss=client_loss,
                client_resources=client_resources,
                client_last_round=client_last_round,
                client_duration=client_duration,
                explored_pairs=explored_pairs
            )

            # 🔥 atualizar exploration memory
            for cid, m in selected_pairs:
                explored_pairs.add((cid, m))

            # -------- separa por modelo (ESSENCIAL) --------
            clients_cifar = []
            clients_gtsrb = []

            for cid, model_name in selected_pairs:
                if model_name == "cifar":
                    clients_cifar.append(cid)
                else:
                    clients_gtsrb.append(cid)

            # =====================================================
            # 2) TREINAMENTO LOCAL — CIFAR
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

                client_loss[cid]["cifar"] = loss

                model_updates["cifar"].append((state_dict, n_samples))
                real_training_counter["cifar"] += 1

                # -------- UPDATE OORT FEEDBACK --------
                client_last_round[cid]["cifar"] = rnd
                client_duration[cid]["cifar"] = train_time

                # -------- UTILITY POR MODELO --------

                loss_cifar = client_loss[cid]["cifar"]
                loss_gtsrb = client_loss[cid]["gtsrb"]

                loss_cifar = loss_cifar if np.isfinite(loss_cifar) else 1.0
                loss_gtsrb = loss_gtsrb if np.isfinite(loss_gtsrb) else 1.0

                data_cifar = client_resources[cid]["data_size_cifar"]
                data_gtsrb = client_resources[cid]["data_size_gtsrb"]

                utility_cifar = data_cifar * abs(loss_cifar)
                utility_gtsrb = data_gtsrb * abs(loss_gtsrb)

                explored_clients.add(cid)

                client_resource_usage[cid]["cifar"] += train_time

            # =====================================================
            # 3) TREINAMENTO LOCAL — GTSRB
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

                client_loss[cid]["gtsrb"] = loss

                model_updates["gtsrb"].append((state_dict, n_samples))
                real_training_counter["gtsrb"] += 1

                # -------- UPDATE OORT FEEDBACK --------
                client_last_round[cid]["gtsrb"] = rnd
                client_duration[cid]["gtsrb"] = train_time

                # utility estatística (loss-based, paper)
                data_size = (
                        client_resources[cid]["data_size_cifar"] +
                        client_resources[cid]["data_size_gtsrb"]
                )

                # -------- UTILITY POR MODELO --------

                loss_cifar = client_loss[cid]["cifar"]
                loss_gtsrb = client_loss[cid]["gtsrb"]

                loss_cifar = loss_cifar if np.isfinite(loss_cifar) else 1.0
                loss_gtsrb = loss_gtsrb if np.isfinite(loss_gtsrb) else 1.0

                data_cifar = client_resources[cid]["data_size_cifar"]
                data_gtsrb = client_resources[cid]["data_size_gtsrb"]

                utility_cifar = data_cifar * abs(loss_cifar)
                utility_gtsrb = data_gtsrb * abs(loss_gtsrb)

                client_utility[cid] = utility_cifar + utility_gtsrb

                explored_clients.add(cid)

                client_resource_usage[cid]["gtsrb"] += train_time

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
            loss_cifar_norm, loss_gtsrb_norm, data_cifar_norm, data_gtsrb_norm = compute_normalized_metrics()

            inter_client_fairness = compute_inter_client_fairness(
                loss_cifar_norm,
                loss_gtsrb_norm,
                data_cifar_norm,
                data_gtsrb_norm
            )

            # -------- Intra-client fairness --------
            intra_client_fairness = compute_intra_client_fairness_utility(
                loss_cifar_norm,
                loss_gtsrb_norm,
                data_cifar_norm,
                data_gtsrb_norm,
                beta=0.7,
                utility_smoothing=0.1
            )

            round_util = sum(
                compute_oort_utility_model(cid, m, rnd)
                for cid, m in selected_pairs
            )
            oort_round_utilities.append(round_util)
            oort_T = update_pacer(
                oort_round_utilities,
                oort_T,
                oort_delta,
                oort_window
            )

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

                # =========================
                # 🔥 PRINT DA ACURÁCIA
                # =========================
                print(f"📊 {model_name.upper()} | Round {rnd} | Acc: {global_acc:.4f}")

                # salvar linha no CSV
                row_data = {
                    "algorithm": "Oort",
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
                    f"{RESULTS_DIR}/oort_{model_name}"
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