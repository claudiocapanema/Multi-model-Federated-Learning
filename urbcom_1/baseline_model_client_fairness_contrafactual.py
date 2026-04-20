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
TOTAL_CLIENTS = 50

ROUNDS = 100
FRAC = 0.3

LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01

# =====================================================
# MODEL COST SETUPS
# =====================================================

MODEL_COST_SETUPS = {

    # 🔹 BASELINE (1x)
    "cost_1x": {
        "cifar": {
            "flops_per_sample": 5e6,
        },
        "gtsrb": {
            "flops_per_sample": 5e6,
        }
    },

    # 🔹 EXISTENTE (2x)
    "cost_2x": {
        "cifar": {
            "flops_per_sample": 5e6,
        },
        "gtsrb": {
            "flops_per_sample": 1.0e7,
        }
    },

    # 🔹 EXISTENTE (4x)
    "cost_4x": {
        "cifar": {
            "flops_per_sample": 5e6,
        },
        "gtsrb": {
            "flops_per_sample": 2.0e7,
        }
    },

    # 🔥 NOVO (6x)
    "cost_6x": {
        "cifar": {
            "flops_per_sample": 5e6,
        },
        "gtsrb": {
            "flops_per_sample": 3.0e7,
        }
    },

    # 🔥 NOVO (8x)
    "cost_8x": {
        "cifar": {
            "flops_per_sample": 5e6,
        },
        "gtsrb": {
            "flops_per_sample": 4.0e7,
        }
    },

    # 🔥 NOVO (10x)
    "cost_10x": {
        "cifar": {
            "flops_per_sample": 5e6,
        },
        "gtsrb": {
            "flops_per_sample": 5.0e7,
        }
    },
}

# =====================================================
# SELECT COST SETUP
# =====================================================

# =====================================================
# CONTRAFACTUAL FRACTIONS (SENSITIVITY ANALYSIS)
# =====================================================

REMOVAL_FRACTIONS = [0.0, 0.1, 0.2, 0.3]

# BETA = 0.1
# BETA = 0.5
BETA = 1.0

VERSIONS = [
    "traditional",

    "remove_random_inter",
    "remove_top_gamma_inter",

    "remove_random_intra",
    "remove_top_gamma_intra",
]

version = VERSIONS[1]

# DIRICHLET_ALPHA = 0.1
DIRICHLET_ALPHA = 1.0

# COST_SETUP_NAME = "cost_1x"
# COST_SETUP_NAME = "cost_2x"
COST_SETUP_NAME = "cost_4x"
# COST_SETUP_NAME = "cost_6x"
# COST_SETUP_NAME = "cost_8x"
# COST_SETUP_NAME = "cost_10x"

print(f"Versão: {version}")

MODEL_COST = MODEL_COST_SETUPS[COST_SETUP_NAME]

# =====================================================
# TRAINING TIME CACHE (POR RODADA)
# =====================================================

training_time_cache = {}

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

    # -------- COMPUTE-BASED TIME PROXY --------
    total_flops = flops_per_sample * data_size * LOCAL_EPOCHS

    # speed já entra aqui → NÃO repetir na fairness
    train_time = total_flops / (1e7 * speed)

    return train_time

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
        for cid in range(TOTAL_CLIENTS)
    }

    # -------- Métricas --------
    client_acc = {
        cid: {"cifar": 0.0, "gtsrb": 0.0}
        for cid in range(TOTAL_CLIENTS)
    }

    client_loss = {
        cid: {"cifar": float("inf"), "gtsrb": float("inf")}
        for cid in range(TOTAL_CLIENTS)
    }

    client_delta_acc = {
        cid: {"cifar": 0.0, "gtsrb": 0.0}
        for cid in range(TOTAL_CLIENTS)
    }

    # -------- Fairness tracking --------
    global client_resource_usage

    client_resource_usage = {
        cid: {
            "cifar": 0.0,
            "gtsrb": 0.0
        }
        for cid in range(TOTAL_CLIENTS)
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

    global training_time_cache

    training_time_cache = {
        cid: {
            "cifar": 0.0,
            "gtsrb": 0.0
        }
        for cid in range(TOTAL_CLIENTS)
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

    for cid in range(TOTAL_CLIENTS):
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

def clear_previous_results(removal_fraction):
    for model_name in ["cifar", "gtsrb"]:
        directory = f"{RESULTS_DIR}/frac_{FRAC}/alpha_dirichlet_{DIRICHLET_ALPHA}/beta_{BETA}/"
        filename = (
            f"{directory}"
            f"baseline_contrafactual_{version}_frac{removal_fraction}_{model_name}.csv"
        )
        if os.path.exists(filename):
            os.remove(filename)

def compute_utilities():
    """
    Utility clássica baseada em tamanho total de dados
    (multi-model aware, sem misturar loss)
    """

    util_cifar = []
    util_gtsrb = []

    for cid in range(TOTAL_CLIENTS):

        n_cifar = client_resources[cid]["data_size_cifar"]
        n_gtsrb = client_resources[cid]["data_size_gtsrb"]

        util_cifar.append(n_cifar)
        util_gtsrb.append(n_gtsrb)

    return util_cifar, util_gtsrb

def get_random_clients_from_pool(pool, fraction):
    n = int(len(pool) * fraction)
    if n == 0:
        return set()
    return set(random.sample(pool, n))

def compute_caf_inter(
    client_resources,
    client_resource_usage,
    gamma_inter,
    model_names,
    eps=1e-16
):

    gamma = []
    usage = []

    # -----------------------------
    # coleta γ_k e u_k
    # -----------------------------
    for cid in client_resources:

        gamma_k = gamma_inter[cid]

        u_k = sum(client_resource_usage[cid][m] for m in model_names)

        gamma.append(gamma_k)
        usage.append(u_k)

    if len(gamma) == 0:
        return 1.0

    total_gamma = sum(gamma) + eps
    total_usage = sum(usage) + eps

    rho = []

    # -----------------------------
    # ρ_k = observed / expected
    # -----------------------------
    for g, u in zip(gamma, usage):

        u_hat = g / total_gamma
        u_tilde = u / total_usage

        rho_k = u_tilde / (u_hat + eps)

        rho.append(rho_k)

    # -----------------------------
    # Jain Index
    # -----------------------------
    num = (sum(rho)) ** 2
    den = len(rho) * sum(r ** 2 for r in rho) + eps

    fairness = num / den

    return max(0.0, min(1.0, fairness))

def compute_caf_intra(
    client_resources,
    client_resource_usage,
    gamma_intra,
    model_names,
    eps=1e-16
):

    total_weight = 0.0
    weighted_sum = 0.0

    for cid in client_resources:

        gamma_km = []
        usage_km = []

        # peso = total de dados do cliente
        total_data_client = sum(
            client_resources[cid][f"data_size_{m}"]
            for m in model_names
        )

        if total_data_client <= 0:
            continue

        # -----------------------------
        # coleta γ_km e u_km
        # -----------------------------
        for m in model_names:

            g = gamma_intra[cid][m]
            u = client_resource_usage[cid][m]

            gamma_km.append(g)
            usage_km.append(u)

        total_gamma = sum(gamma_km) + eps
        total_usage = sum(usage_km) + eps

        rho = []

        # -----------------------------
        # ρ_km
        # -----------------------------
        for g, u in zip(gamma_km, usage_km):

            u_hat = g / total_gamma
            u_tilde = u / total_usage

            rho_km = u_tilde / (u_hat + eps)

            rho.append(rho_km)

        # -----------------------------
        # Jain por cliente
        # -----------------------------
        num = (sum(rho)) ** 2
        den = len(rho) * sum(r ** 2 for r in rho) + eps

        F_k = num / den
        F_k = max(0.0, min(1.0, F_k))

        # -----------------------------
        # média ponderada (paper)
        # -----------------------------
        weighted_sum += total_data_client * F_k
        total_weight += total_data_client

    return weighted_sum / (total_weight + eps)

def compute_gamma_inter_static(beta=BETA, eps=1e-16):

    model_names = list(MODEL_COST.keys())

    # ---------------------------------------
    # Normalizações globais
    # ---------------------------------------
    total_speed = sum(client_resources[cid]["speed"] for cid in client_resources) + eps

    inv_flops = {m: 1.0 / MODEL_COST[m]["flops_per_sample"] for m in model_names}
    total_inv_flops = sum(inv_flops.values()) + eps

    # normalização por modelo (dados)
    total_data_per_model = {
        m: sum(client_resources[cid][f"data_size_{m}"] for cid in client_resources) + eps
        for m in model_names
    }

    gamma_inter = {}

    for cid in client_resources:

        v_k = client_resources[cid]["speed"]
        v_tilde = v_k / total_speed

        gamma_k = 0.0

        for m in model_names:

            n_km = client_resources[cid][f"data_size_{m}"]

            # ñ_k^me (normalização por modelo)
            n_tilde = n_km / total_data_per_model[m]

            # c̃^me
            c_tilde = inv_flops[m] / total_inv_flops

            term = n_tilde * v_tilde * c_tilde

            if term > 0:
                gamma_k += (term) ** beta

        gamma_inter[cid] = max(gamma_k, eps)

    return gamma_inter

def compute_gamma_intra_static(beta=BETA, eps=1e-16):

    model_names = list(MODEL_COST.keys())

    # ---------------------------------------
    # Normalizações globais
    # ---------------------------------------
    total_speed = sum(client_resources[cid]["speed"] for cid in client_resources) + eps

    inv_flops = {m: 1.0 / MODEL_COST[m]["flops_per_sample"] for m in model_names}
    total_inv_flops = sum(inv_flops.values()) + eps

    gamma_intra = {}

    for cid in client_resources:

        gamma_intra[cid] = {}

        v_k = client_resources[cid]["speed"]
        v_tilde = v_k / total_speed

        # normalização por cliente (dados)
        total_data_client = sum(
            client_resources[cid][f"data_size_{m}"] for m in model_names
        ) + eps

        for m in model_names:

            n_km = client_resources[cid][f"data_size_{m}"]

            # ñ_k^me (normalização intra)
            n_tilde = n_km / total_data_client

            # c̃^me
            c_tilde = inv_flops[m] / total_inv_flops

            term = n_tilde * v_tilde * c_tilde

            if term > 0:
                gamma_km = (term) ** beta
            else:
                gamma_km = eps

            gamma_intra[cid][m] = max(gamma_km, eps)

    return gamma_intra

def generate_base_selection(seed, total_clients, rounds, frac):
    """
    Gera seleção base FIXA para todas as versões contrafactuais.
    """
    base_selection = {}

    for rnd in range(1, rounds + 1):
        rng = random.Random(seed + rnd)

        clients = list(range(total_clients))
        rng.shuffle(clients)

        k = max(1, int(frac * total_clients))

        base_selection[rnd] = clients[:k]

    return base_selection

def aggregate_gamma(gamma_km):

    gamma_k = {}

    for cid in gamma_km:

        # 🔥 OPÇÃO PRINCIPAL
        gamma_k[cid] = max(gamma_km[cid].values())

        # alternativas:
        # sum(...)
        # mean(...)
        # weighted sum

    return gamma_k

def get_top_gamma_clients(gamma_dict, fraction=0.2):
    sorted_clients = sorted(gamma_dict.items(), key=lambda x: x[1], reverse=True)
    n = int(len(sorted_clients) * fraction)
    return set([cid for cid, _ in sorted_clients[:n]])


def get_random_clients(fraction=0.2):
    n = int(TOTAL_CLIENTS * fraction)
    if n == 0:
        return set()
    return set(random.sample(range(TOTAL_CLIENTS), n))

def get_eligible_clients(version, base_clients, gamma_inter, gamma_intra, removal_fraction):

    if version == "traditional":
        return base_clients

    elif version == "remove_top_gamma_inter":
        removed = get_top_gamma_clients(
            {cid: gamma_inter[cid] for cid in base_clients},
            removal_fraction
        )
        return [cid for cid in base_clients if cid not in removed]

    elif version == "remove_random_inter":
        removed = get_random_clients_from_pool(base_clients, removal_fraction)
        return [cid for cid in base_clients if cid not in removed]

    elif version == "remove_top_gamma_intra":
        removed = get_top_gamma_clients(
            {cid: max(gamma_intra[cid].values()) for cid in base_clients},
            removal_fraction
        )
        return [cid for cid in base_clients if cid not in removed]

    elif version == "remove_random_intra":
        removed = get_random_clients_from_pool(base_clients, removal_fraction)
        return [cid for cid in base_clients if cid not in removed]

    else:
        return base_clients

def get_blocked_models(version, selected_clients, gamma_intra, removal_fraction, round_seed):

    blocked = {}

    k = int(len(selected_clients) * removal_fraction)
    if k == 0:
        return blocked

    # 🔹 RNG global só para escolher quem será afetado
    rng_global = random.Random(round_seed)

    affected = rng_global.sample(selected_clients, k)

    # =====================================================
    # REMOVE TOP GAMMA INTRA
    # =====================================================
    if version == "remove_top_gamma_intra":

        for cid in affected:
            gamma_km = gamma_intra[cid]

            # 🔥 tie-break determinístico por cliente (evita padrões globais)
            items = list(gamma_km.items())

            items.sort(
                key=lambda x: (
                    -x[1],                          # maior gamma primeiro
                    hash((cid, round_seed, x[0]))   # desempate consistente
                )
            )

            blocked[cid] = items[0][0]

    # =====================================================
    # REMOVE RANDOM INTRA
    # =====================================================
    elif version == "remove_random_intra":

        for cid in affected:

            # 🔥 RNG independente por cliente (remove correlação)
            local_rng = random.Random(hash((cid, round_seed)))

            options = list(gamma_intra[cid].keys())

            blocked[cid] = local_rng.choice(options)

    return blocked

def assign_clients_to_models(all_clients, blocked_model, round_seed):

    rng = random.Random(round_seed)

    k = int(FRAC * TOTAL_CLIENTS)
    half = k // 2

    # -------------------------------------------------
    # 1. separa clientes por restrição
    # -------------------------------------------------
    free_clients = []
    only_cifar = []
    only_gtsrb = []

    for cid in all_clients:

        if cid in blocked_model:
            blocked = blocked_model[cid]

            if blocked == "cifar":
                only_gtsrb.append(cid)

            elif blocked == "gtsrb":
                only_cifar.append(cid)

        else:
            free_clients.append(cid)

    # embaralha tudo (IMPORTANTE)
    rng.shuffle(free_clients)
    rng.shuffle(only_cifar)
    rng.shuffle(only_gtsrb)

    # -------------------------------------------------
    # 2. alocação forçada (clientes com restrição)
    # -------------------------------------------------
    clients_cifar = []
    clients_gtsrb = []

    # primeiro, encaixa os restritos
    clients_cifar.extend(only_cifar[:half])
    clients_gtsrb.extend(only_gtsrb[:half])

    # -------------------------------------------------
    # 3. completa com livres
    # -------------------------------------------------
    needed_cifar = half - len(clients_cifar)
    needed_gtsrb = half - len(clients_gtsrb)

    if needed_cifar > 0:
        clients_cifar.extend(free_clients[:needed_cifar])
        free_clients = free_clients[needed_cifar:]

    if needed_gtsrb > 0:
        clients_gtsrb.extend(free_clients[:needed_gtsrb])
        free_clients = free_clients[needed_gtsrb:]

    # -------------------------------------------------
    # 4. ajuste final se k for ímpar
    # -------------------------------------------------
    if len(clients_cifar) + len(clients_gtsrb) < k and len(free_clients) > 0:
        clients_cifar.append(free_clients.pop())

    # -------------------------------------------------
    # 5. sanity check forte
    # -------------------------------------------------
    assert len(clients_cifar) + len(clients_gtsrb) == k, \
        f"Erro: esperado {k}, obtido {len(clients_cifar) + len(clients_gtsrb)}"

    return clients_cifar, clients_gtsrb

def get_blocked_model_top_gamma(gamma_intra_static, clients):

    blocked = {}

    for cid in clients:
        gamma_km = gamma_intra_static[cid]
        blocked[cid] = max(gamma_km, key=gamma_km.get)

    return blocked

def deterministic_choice(cid, round_seed, options):
    local_rng = random.Random(hash((cid, round_seed)))
    return local_rng.choice(options)

def get_removed_clients(version, all_clients, gamma_inter, removal_fraction, round_seed):

    removed = set()

    k = int(len(all_clients) * removal_fraction)
    if k == 0:
        return removed

    rng = random.Random(round_seed)

    # =====================================================
    # REMOVE RANDOM INTER
    # =====================================================
    if version == "remove_random_inter":

        removed = set(rng.sample(all_clients, k))

    # =====================================================
    # REMOVE TOP GAMMA INTER
    # =====================================================
    elif version == "remove_top_gamma_inter":

        gamma_pool = {cid: gamma_inter[cid] for cid in all_clients}

        # ordena determinístico (tie-break incluído)
        sorted_clients = sorted(
            gamma_pool.items(),
            key=lambda x: (-x[1], hash((x[0], round_seed)))
        )

        removed = set([cid for cid, _ in sorted_clients[:k]])

    return removed

def run_experiment():

    # =====================================================
    # LIMPEZA INICIAL
    # Remove arquivos CSV antigos para evitar mistura de resultados
    # =====================================================

    # =====================================================
    # LOOP SOBRE FOLDS (cross-validation ou repetição com seeds)
    # =====================================================
    for removal_fraction in REMOVAL_FRACTIONS:

        clear_previous_results(removal_fraction)

        print("\n========================================")
        print(f"🚨 CONTRAFACTUAL FRACTION: {removal_fraction}")
        print("========================================")

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
            # 🔥 BASE SELECTION (CONTRAFATUAL CORRETO)
            # =====================================================
            all_clients = list(range(TOTAL_CLIENTS))

            # =====================================================
            # CRIAÇÃO DOS DATALOADERS POR CLIENTE
            # Cada cliente recebe uma partição não-iid (Dirichlet)
            # =====================================================
            train_loaders = defaultdict(dict)
            test_loaders = defaultdict(dict)

            for cid in range(TOTAL_CLIENTS):

                # ---------- CIFAR ----------
                train_loader, test_loader = load_data(
                    dataset_name="CIFAR10",
                    alpha=DIRICHLET_ALPHA,
                    partition_id=cid,
                    num_partitions=TOTAL_CLIENTS + 1,
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
                    num_partitions=TOTAL_CLIENTS + 1,
                    batch_size=BATCH_SIZE,
                    fold_id=fold + 1,
                    data_sampling_percentage=0.8,
                    get_from_volume=True
                )

                train_loaders[cid]["gtsrb"] = train_loader
                test_loaders[cid]["gtsrb"] = test_loader

            reset_experiment_state(train_loaders)

            # =====================================================
            # 🔥 GAMMA FIXO (CONTRAFACTUAL CORRETO)
            # =====================================================

            gamma_inter_static = compute_gamma_inter_static()
            gamma_intra_static = compute_gamma_intra_static()

            # =====================================================
            # 🔥 STATIC CONTRAFACTUAL SETUP (INTER + INTRA)
            # =====================================================

            rng_static = random.Random(fold_seed)

            all_clients = list(range(TOTAL_CLIENTS))
            k = int(len(all_clients) * removal_fraction)

            # ---------------------------
            # INTER (clientes removidos)
            # ---------------------------
            removed_clients_static = set()

            if version == "remove_random_inter":

                if k > 0:
                    removed_clients_static = set(rng_static.sample(all_clients, k))

            elif version == "remove_top_gamma_inter":

                sorted_clients = sorted(
                    gamma_inter_static.items(),
                    key=lambda x: -x[1]
                )

                removed_clients_static = set([cid for cid, _ in sorted_clients[:k]])

            # ---------------------------
            # INTRA (clientes afetados)
            # ---------------------------
            affected_clients_static = set()
            blocked_model_static = {}

            if version in ["remove_random_intra", "remove_top_gamma_intra"]:

                if k > 0:
                    affected_clients_static = set(rng_static.sample(all_clients, k))

                for cid in affected_clients_static:

                    if version == "remove_random_intra":
                        blocked_model_static[cid] = rng_static.choice(list(MODEL_COST.keys()))

                    elif version == "remove_top_gamma_intra":
                        gamma_km = gamma_intra_static[cid]
                        blocked_model_static[cid] = max(gamma_km, key=gamma_km.get)

            # =====================================================
            # TRACK DE EFICIÊNCIA POR MODELO (CORRETO)
            # =====================================================
            efficiency_history = {
                "cifar": [],
                "gtsrb": []
            }

            # =====================================================
            # TRACK GLOBAL DE EFICIÊNCIA
            # =====================================================
            total_training_time = 0.0
            total_accuracy_gain = 0.0

            random_removed_static = get_random_clients(removal_fraction)

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

                print(f"\n🔄 FOLD {fold} | RODADA {rnd} | REMOVAL FRACTION {removal_fraction} | VERSION {version}")

                # =====================================================
                # ATUALIZA TRAINING TIME CACHE (1x por rodada)
                # =====================================================
                for cid in range(TOTAL_CLIENTS):
                    training_time_cache[cid]["cifar"] = estimate_training_time(cid, "cifar")
                    training_time_cache[cid]["gtsrb"] = estimate_training_time(cid, "gtsrb")

                # Contador REAL de clientes que treinaram
                real_training_counter = {m: 0 for m in global_models}

                # =====================================================
                # 1) SELEÇÃO DE CLIENTES (CORE DO MÉTO DO)
                # Balanced Resource + Fairness entre clientes
                # =====================================================

                removed_clients = set()
                affected_clients = set()
                blocked_model = {}

                rng = random.Random(round_seed)

                # ---------------------------
                # INTER (fixo)
                # ---------------------------
                removed_clients = {
                    cid for cid in removed_clients_static
                }

                selected_clients = [
                    cid for cid in all_clients
                    if cid not in removed_clients
                ]

                # ---------------------------
                # INTRA (fixo)
                # ---------------------------
                blocked_model = {
                    cid: blocked_model_static[cid]
                    for cid in selected_clients
                    if cid in blocked_model_static
                }

                # -------------------------------------------------
                # ALOCAÇÃO FINAL
                # -------------------------------------------------
                clients_cifar, clients_gtsrb = assign_clients_to_models(
                    selected_clients,
                    blocked_model,
                    round_seed
                )

                all_assigned = set(clients_cifar) | set(clients_gtsrb)

                missing = set(all_clients) - all_assigned

                if len(missing) > 0:
                    print("🚨 CLIENTES PERDIDOS:", missing)

                # controle de recurso (mantido só para logging)
                resource_usage = {
                    "cifar": 0.0,
                    "gtsrb": 0.0
                }

                # atualiza usage (IMPORTANTE para manter fairness tracking funcionando)
                for cid in clients_cifar:
                    train_time = training_time_cache[cid]["cifar"]
                    resource_usage["cifar"] += train_time
                    client_resource_usage[cid]["cifar"] += train_time

                for cid in clients_gtsrb:
                    train_time = training_time_cache[cid]["gtsrb"]
                    resource_usage["gtsrb"] += train_time
                    client_resource_usage[cid]["gtsrb"] += train_time

                # =====================================================
                # 2) TREINAMENTO LOCAL — CIFAR
                # =====================================================
                for cid in clients_cifar:

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

                    client_loss[cid]["cifar"] = loss

                # =====================================================
                # 3) TREINAMENTO LOCAL — GTSRB
                # =====================================================
                for cid in clients_gtsrb:

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

                    client_loss[cid]["gtsrb"] = loss

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
                resource_cifar = sum(client_resource_usage[c]["cifar"] for c in range(TOTAL_CLIENTS))
                resource_gtsrb = sum(client_resource_usage[c]["gtsrb"] for c in range(TOTAL_CLIENTS))

                total_resource = resource_cifar + resource_gtsrb

                if total_resource > 0:
                    imbalance = abs(resource_cifar - resource_gtsrb) / total_resource
                    inter_model_fairness = 1.0 - imbalance
                else:
                    inter_model_fairness = 1.0

                model_names = list(global_models.keys())

                inter_client_fairness = compute_caf_inter(
                    client_resources,
                    client_resource_usage,
                    gamma_inter_static,  # ✅ CORRETO
                    model_names
                )

                intra_client_fairness = compute_caf_intra(
                    client_resources,
                    client_resource_usage,
                    gamma_intra_static,  # ✅ CORRETO
                    model_names
                )

                # =====================================================
                # 6) LOGS DE RECURSOS E FAIRNESS
                # =====================================================
                resource_cifar_real = sum(
                    training_time_cache[cid]["cifar"]
                    for cid in clients_cifar
                )

                resource_gtsrb_real = sum(
                    training_time_cache[cid]["gtsrb"]
                    for cid in clients_gtsrb
                )

                round_time = resource_cifar_real + resource_gtsrb_real
                total_training_time += round_time

                logs["resource_usage_cifar"].append(resource_cifar_real)
                logs["resource_usage_gtsrb"].append(resource_gtsrb_real)

                logs["inter_model_fairness"].append(inter_model_fairness)
                logs["inter_client_fairness"].append(inter_client_fairness)
                logs["intra_client_fairness"].append(intra_client_fairness)

                prev_acc = {
                    m: global_acc_history[m][-1] if len(global_acc_history[m]) > 0 else 0.0
                    for m in global_models
                }

                # =====================================================
                # 7) AVALIAÇÃO GLOBAL + EFICIÊNCIA POR RODADA
                # =====================================================
                for model_name, model in global_models.items():

                    dataset_name = {
                        "cifar": "CIFAR10",
                        "gtsrb": "GTSRB"
                    }[model_name]

                    global_acc = evaluate_global_model(
                        model,
                        {cid: test_loaders[cid][model_name] for cid in range(TOTAL_CLIENTS)},
                        dataset_name
                    )

                    global_acc_history[model_name].append(global_acc)

                    delta = global_acc - prev_acc[model_name]

                    if model_name == "cifar":
                        time = resource_cifar_real
                    else:
                        time = resource_gtsrb_real

                    if time > 0:
                        eff_round = delta / time
                    else:
                        eff_round = 0.0

                    efficiency_history[model_name].append(eff_round)

                    print(f"⚡ Efficiency ({model_name}): {eff_round:.6f}")
                    print(f"📊 {model_name.upper()} | Round {rnd} | Acc: {global_acc:.4f}")

                # =====================================================
                # EFICIÊNCIA MÉDIA (AGORA CORRETA)
                # =====================================================
                efficiency_mean_cifar = np.mean(efficiency_history["cifar"]) if len(
                    efficiency_history["cifar"]) > 0 else 0.0

                efficiency_mean_gtsrb = np.mean(efficiency_history["gtsrb"]) if len(
                    efficiency_history["gtsrb"]) > 0 else 0.0

                total_time_models = resource_cifar_real + resource_gtsrb_real

                if total_time_models > 0:
                    efficiency_global_mean = np.mean([
                        efficiency_mean_cifar,
                        efficiency_mean_gtsrb
                    ])
                else:
                    efficiency_global_mean = 0.0

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

                # =====================================================
                # SALVAR RESULTADOS (CSV)
                # =====================================================
                for model_name in global_models:
                    global_acc = global_acc_history[model_name][-1]

                    row_data = {
                        "algorithm": "MultiFedAvg",
                        "fold": fold,
                        "round": rnd,
                        "dataset": model_name,
                        "global_acc": global_acc,

                        "clients_cifar": real_training_counter["cifar"],
                        "clients_gtsrb": real_training_counter["gtsrb"],

                        "clients_selected": real_training_counter[model_name],
                        "clients_selected_total": sum(real_training_counter.values()),

                        "resource_usage_cifar": resource_cifar_real,
                        "resource_usage_gtsrb": resource_gtsrb_real,

                        "inter_model_fairness": inter_model_fairness,
                        "inter_client_fairness": inter_client_fairness,
                        "intra_client_fairness": intra_client_fairness,

                        "total_training_time": total_training_time,

                        "version": version,
                        "removal_fraction": removal_fraction,

                        # 🔥 MÉTRICAS CORRETAS
                        "efficiency_mean_cifar": efficiency_mean_cifar,
                        "efficiency_mean_gtsrb": efficiency_mean_gtsrb,
                        "efficiency_global_mean": efficiency_global_mean,
                    }

                    directory = f"{RESULTS_DIR}/frac_{FRAC}/alpha_dirichlet_{DIRICHLET_ALPHA}/beta_{BETA}/"
                    filename = (
                        f"{directory}"
                        f"baseline_contrafactual_{version}_frac{removal_fraction}_{model_name}.csv"
                    )

                    os.makedirs(directory, exist_ok=True)

                    append_result_to_csv(row_data, filename)


if __name__ == "__main__":

    run_experiment()