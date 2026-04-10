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

# =====================================================
# MODEL COST (FLOPs POR AMOSTRA)
# =====================================================

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

DIRICHLET_ALPHA = 0.1
# DIRICHLET_ALPHA = 1.0

COST_SETUP_NAME = "cost_1x"
# COST_SETUP_NAME = "cost_2x"
# COST_SETUP_NAME = "cost_4x"
# COST_SETUP_NAME = "cost_6x"
# COST_SETUP_NAME = "cost_8x"
# COST_SETUP_NAME = "cost_10x"

MODEL_COST = MODEL_COST_SETUPS[COST_SETUP_NAME]

# =====================================================
# TRAINING TIME CACHE (POR RODADA)
# =====================================================

training_time_cache = {}

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

    # ruído realista
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

    global training_time_cache

    training_time_cache = {
        cid: {
            "cifar": 0.0,
            "gtsrb": 0.0
        }
        for cid in range(NUM_CLIENTS)
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
            f"{RESULTS_DIR}/baseline_{model_name}"
            f"_frac_{FRAC}"
            f"_alpha_{DIRICHLET_ALPHA}"
            f"_lambdaCap_{LAMBDA_CAPACITY}"
            f"_lambdaIntra_{LAMBDA_INTRA}.csv"
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

    for cid in range(NUM_CLIENTS):

        n_cifar = client_resources[cid]["data_size_cifar"]
        n_gtsrb = client_resources[cid]["data_size_gtsrb"]

        util_cifar.append(n_cifar)
        util_gtsrb.append(n_gtsrb)

    return util_cifar, util_gtsrb

def compute_caf_inter(
    client_resources,
    client_resource_usage,
    training_time_cache,
    model_names,
    gamma=0.5,
    eps=1e-8,
    alpha_eff=0.5
):

    utility = []
    usage = []

    for cid in client_resources:

        w_data = 0.0

        for m in model_names:
            n = client_resources[cid][f"data_size_{m}"]
            speed = client_resources[cid]["speed"]
            flops = MODEL_COST[m]["flops_per_sample"]

            if flops < eps:
                continue

            # 🔥 NOVA MÉTRICA CORRETA
            w_data += ((n * speed) / flops) ** alpha_eff

        if w_data < eps:
            continue

        use = sum(client_resource_usage[cid][m] for m in model_names)

        utility.append(w_data)
        usage.append(use)

    if len(utility) == 0:
        return 1.0

    total_util = sum(utility) + eps
    total_usage = sum(usage) + eps

    ratios = []
    for i in range(len(utility)):
        expected = utility[i] / total_util
        actual = usage[i] / total_usage
        ratios.append(actual / (expected + eps))

    num = (sum(ratios) ** 2)
    den = len(ratios) * sum(r**2 for r in ratios)

    return num / den if den > 0 else 1.0

def compute_caf_intra(
    client_resources,
    client_resource_usage,
    training_time_cache,
    model_names,
    eps=1e-8,
    alpha_eff=0.5
):

    total_weight = 0.0
    weighted_sum = 0.0

    for cid in client_resources:

        weights = []
        usages = []

        for m in model_names:
            n = client_resources[cid][f"data_size_{m}"]
            u = client_resource_usage[cid][m]

            speed = client_resources[cid]["speed"]
            flops = MODEL_COST[m]["flops_per_sample"]

            if flops < eps:
                continue

            # 🔥 NOVA DEFINIÇÃO
            weights.append(((n * speed) / flops) ** alpha_eff)
            usages.append(u)

        if len(weights) == 0:
            continue

        total_w = sum(weights)
        total_u = sum(usages)

        if total_w < eps or total_u < eps:
            continue

        ratios = []
        for w, u in zip(weights, usages):
            exp = w / total_w
            act = u / total_u
            ratios.append(act / (exp + eps))

        M = len(ratios)
        num = (sum(ratios) ** 2)
        den = M * sum(r**2 for r in ratios)

        J = num / den if den > 0 else 1.0

        total_data = sum(
            client_resources[cid][f"data_size_{m}"]
            for m in model_names
        )

        weighted_sum += total_data * J
        total_weight += total_data

    return weighted_sum / total_weight if total_weight > 0 else 1.0

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

            # =====================================================
            # ATUALIZA TRAINING TIME CACHE (1x por rodada)
            # =====================================================
            for cid in range(NUM_CLIENTS):
                training_time_cache[cid]["cifar"] = estimate_training_time(cid, "cifar")
                training_time_cache[cid]["gtsrb"] = estimate_training_time(cid, "gtsrb")

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

            # embaralha clientes (ordem aleatória)
            all_clients = list(range(NUM_CLIENTS))
            random.shuffle(all_clients)

            # =====================================================
            # LOOP DE DECISÃO POR CLIENTE (NOVO — MAX FAIRNESS)
            # =====================================================
            total_cifar_usage = sum(client_resource_usage[c]["cifar"] for c in range(NUM_CLIENTS))
            total_gtsrb_usage = sum(client_resource_usage[c]["gtsrb"] for c in range(NUM_CLIENTS))

            # =====================================================
            # MULTIFEDAVG CLIENT SELECTION (BASELINE)
            # =====================================================

            all_clients = list(range(NUM_CLIENTS))
            random.shuffle(all_clients)

            selected_clients = all_clients[:K_CLIENTS]

            # -------------------------------------------------
            # GARANTIR: 1 cliente → 1 modelo
            # -------------------------------------------------
            clients_cifar = []
            clients_gtsrb = []

            for i, cid in enumerate(selected_clients):
                if i % 2 == 0:
                    clients_cifar.append(cid)
                else:
                    clients_gtsrb.append(cid)

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
            resource_cifar = sum(client_resource_usage[c]["cifar"] for c in range(NUM_CLIENTS))
            resource_gtsrb = sum(client_resource_usage[c]["gtsrb"] for c in range(NUM_CLIENTS))

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
                training_time_cache,
                model_names,
                gamma=0.5
            )

            intra_client_fairness = compute_caf_intra(
                client_resources,
                client_resource_usage,
                training_time_cache,
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
                    "algorithm": "MultiFedAvg",
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
                    f"{RESULTS_DIR}/baseline_{model_name}"
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