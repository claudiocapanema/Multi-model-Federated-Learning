# ===============================
# PARTE 1 — IMPORTS E CONFIG
# ===============================

import copy
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transforms
from models_utils import load_data
# ===============================
# CONFIGURAÇÕES GERAIS
# ===============================

import pandas as pd

BASE_SEED = 42
NUM_FOLDS = 5


def reset_experiment_state():

    global global_models
    global client_resources
    global client_acc
    global client_loss
    global client_delta_acc
    global proposta_logs
    global global_acc_history

    # reset modelos
    global_models = {
        "cifar": SimpleCNN(10).to(DEVICE),
        "gtsrb": SimpleCNN(43).to(DEVICE)
    }

    # reset histórico global
    global_acc_history = {
        "cifar": [],
        "gtsrb": []
    }

    # reset recursos clientes
    client_resources = {}
    for cid in range(NUM_CLIENTS):
        client_resources[cid] = {
            "battery": np.random.uniform(0.6, 1.0),
            "compute": np.random.uniform(0.3, 1.0),
            "link": 1.0
        }

    # reset métricas cliente
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

    # reset logs RAWCS
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 40
ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
regime = "realistic"

global_acc_history = {
    "cifar": [],
    "gtsrb": []
}


# ===============================
# REGIMES EXPERIMENTAIS RAWCS
# ===============================

def get_regime(name: str):
    """
    Retorna um dicionário com a configuração
    do regime experimental.
    """

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

    if name not in regimes:
        raise ValueError(f"Regime desconhecido: {name}")

    return regimes[name]

LAMBDA_ALPHA = 0.5

def compute_model_lambda(model_name, round_idx):

    # -------------------------------------------------
    # CASO INICIAL (ROUND 1) → comportamento neutro
    # -------------------------------------------------
    if len(global_acc_history[model_name]) == 0:
        return 1.0

    # -------------------------------------------------
    # PERFORMANCE DEFICIT
    # -------------------------------------------------
    global_acc = global_acc_history[model_name][-1]
    perf_deficit = max(TARGET_ACC[model_name] - global_acc, 0.0)

    # -------------------------------------------------
    # PARTICIPATION DEFICIT
    # -------------------------------------------------
    if len(proposta_logs["clients_per_model"]) == 0:
        part_deficit = 1.0
    else:
        prev_clients = proposta_logs["clients_per_model"][-1][model_name]
        part_deficit = 1.0 - (prev_clients / NUM_CLIENTS)

    # -------------------------------------------------
    # COMBINAÇÃO
    # -------------------------------------------------
    lambda_m = (
        LAMBDA_ALPHA * perf_deficit +
        (1 - LAMBDA_ALPHA) * part_deficit
    )

    return lambda_m

COLLAPSE_WINDOW = 3

def check_model_collapse(model_name):
    if len(proposta_logs["clients_per_model"]) < COLLAPSE_WINDOW:
        return False

    recent = proposta_logs["clients_per_model"][-COLLAPSE_WINDOW:]

    total_clients = sum(r[model_name] for r in recent)

    if total_clients == 0:
        return True

    return False


def compute_all_lambdas(round_idx):
    lambdas = {}

    for m in global_models:
        lambdas[m] = compute_model_lambda(m, round_idx)

    total = sum(lambdas.values())

    if total == 0:
        return {m: 1.0 / len(lambdas) for m in lambdas}

    return {m: lambdas[m] / total for m in lambdas}


# ===============================
# APLICAR REGIME AO EXPERIMENTO
# ===============================

def apply_regime(name: str):
    global BATTERY_DECAY, TIME_MAX, LINK_MIN

    cfg = get_regime(name)

    # parâmetros globais
    BATTERY_DECAY = cfg["BATTERY_DECAY"]
    TIME_MAX = cfg["TIME_MAX"]
    LINK_MIN = cfg["LINK_MIN"]

    # estado inicial dos clientes
    for cid in range(NUM_CLIENTS):
        client_resources[cid]["battery"] = np.random.uniform(
            *cfg["battery_init"]
        )

        client_resources[cid]["compute"] = np.random.uniform(
            *cfg["compute"]
        )

        client_resources[cid]["link"] = np.random.uniform(
            *cfg["link"]
        )

    print(f"🧪 Regime experimental aplicado: {name.upper()}")

def update_link(cid):
    low, high = current_regime["link"]
    client_resources[cid]["link"] = np.random.uniform(low, high)



# ===============================
# MODELOS
# ===============================

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

global_models = {
    "cifar": SimpleCNN(10).to(DEVICE),
    "gtsrb": SimpleCNN(43).to(DEVICE)
}

# ===============================
# ESTADO DE RECURSOS POR CLIENTE
# ===============================

client_resources = {}

for cid in range(NUM_CLIENTS):
    client_resources[cid] = {
        "battery": np.random.uniform(0.6, 1.0),     # cb inicial
        "compute": np.random.uniform(0.3, 1.0),     # capacidade fixa
        "link": 1.0                                 # atualizado por rodada
    }

# ===============================
# CUSTO DOS MODELOS
# ===============================

MODEL_COST = {
    "cifar": 1.0,
    "gtsrb": 2.0   # deliberadamente mais pesado
}

def estimate_training_time(cid, model_name):
    return MODEL_COST[model_name] / client_resources[cid]["compute"]

# ===============================
# CONSUMO DE BATERIA
#ROUND ===============================

BATTERY_DECAY = 0.05  # θ do artigo

def consume_battery(cid, train_time):
    client_resources[cid]["battery"] -= train_time * BATTERY_DECAY
    client_resources[cid]["battery"] = max(
        client_resources[cid]["battery"], 0.0
    )

def update_link(cid):
    client_resources[cid]["link"] = np.random.uniform(0.0, 1.0)

# ===============================
# FUNÇÃO DE VIABILIDADE (RAWCS)
# ===============================

BATTERY_MIN = 0.2
TIME_MAX = 2.5
LINK_MIN = 0.3

def R(cid, model_name):
    train_time = estimate_training_time(cid, model_name)
    battery_after = client_resources[cid]["battery"] - train_time * BATTERY_DECAY

    return (
        battery_after >= BATTERY_MIN and
        train_time <= TIME_MAX and
        client_resources[cid]["link"] >= LINK_MIN
    )

import os
import pandas as pd

def append_result_to_csv(row_dict, filename):

    df_row = pd.DataFrame([row_dict])

    file_exists = os.path.exists(filename)

    df_row.to_csv(
        filename,
        mode="a",
        header=not file_exists,
        index=False
    )

from pathlib import Path

Path("results/").mkdir(parents=True, exist_ok=True)

# Remover CSVs antigos
for model_name in ["cifar", "gtsrb"]:
    old_file = f"results/proposta_{model_name}_regime_{regime}.csv"
    if os.path.exists(old_file):
        os.remove(old_file)
        print(f"🗑 Removido arquivo antigo: {old_file}")



# ===============================
# FUNÇÃO DE UTILIDADE
# ===============================

ALPHA = 0.4
BETA  = 0.4
GAMMA = 0.2

TARGET_ACC = {
    "cifar": 0.75,
    "gtsrb": 0.90
}

# ===============================
# FUNÇÃO DE UTILIDADE (MODELO-AWARE)
# ===============================

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

    # utilidade RAWCS (sem colapsar modelos)
    return max((1.0 - cost) * acc_gap, 0.0)

# ===============================
# MÉTRICAS POR CLIENTE E MODELO
# ===============================

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

# ===============================
# LOGS RAWCS (MULTI-MODELO)
# ===============================

proposta_logs = {
    "viable_pairs": [],
    "viable_clients": [],
    "viable_pairs_per_model": [],   # <-- NOVO
    "clients_per_model": [],
    "avg_battery": [],
    "avg_link": [],
    "avg_cost": [],                 # agora por modelo
    "avg_train_time": [],           # agora por modelo
    "max_train_time": [],           # agora por modelo
    "fallback_rate": []
}


# ===============================
# RAWCS MULTI-MODELO (PADRÃO ARTIGO)
# ===============================

# ===============================
# RAWCS MULTI-MODELO (PADRÃO RAWCS REAL)
# ===============================

def rawcs_multi_model(t):

    lambdas = compute_all_lambdas(t)

    assignments = {}

    viable_pairs = 0
    viable_clients = 0
    fallback_count = 0

    viable_pairs_per_model = {m: 0 for m in global_models}
    costs_per_model = {m: [] for m in global_models}
    train_times_per_model = {m: [] for m in global_models}

    # 1) atualizar link (volátil)
    for cid in range(NUM_CLIENTS):
        update_link(cid)

    # 2) decisão cliente -> modelo
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

    # 3) LOGS RAWCS (SEM COLAPSAR MODELOS)
    proposta_logs["viable_pairs"].append(viable_pairs)
    proposta_logs["viable_clients"].append(viable_clients)
    proposta_logs["viable_pairs_per_model"].append(viable_pairs_per_model)
    proposta_logs["fallback_rate"].append(fallback_count / NUM_CLIENTS)

    proposta_logs["avg_cost"].append({
        m: float(np.mean(costs_per_model[m])) if costs_per_model[m] else None
        for m in global_models
    })

    proposta_logs["avg_train_time"].append({
        m: float(np.mean(train_times_per_model[m])) if train_times_per_model[m] else None
        for m in global_models
    })

    proposta_logs["max_train_time"].append({
        m: float(np.max(train_times_per_model[m])) if train_times_per_model[m] else None
        for m in global_models
    })

    proposta_logs["avg_battery"].append(
        float(np.mean([client_resources[cid]["battery"] for cid in range(NUM_CLIENTS)]))
    )

    proposta_logs["avg_link"].append(
        float(np.mean([client_resources[cid]["link"] for cid in range(NUM_CLIENTS)]))
    )

    return assignments

# ===============================
# DATASETS E DATALOADERS
# ===============================

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def split_client_train_test(indices, train_ratio=0.8):

    indices = np.array(indices)
    np.random.shuffle(indices)

    split_point = int(len(indices) * train_ratio)

    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]

    return train_idx.tolist(), test_idx.tolist()


def dirichlet_split(dataset, num_clients, alpha):

    # ----------------------------------------
    # Extração robusta de labels
    # ----------------------------------------

    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)

    elif hasattr(dataset, "_samples"):
        # GTSRB
        labels = np.array([s[1] for s in dataset._samples])

    else:
        raise ValueError("Dataset não suportado para Dirichlet split.")

    num_classes = len(np.unique(labels))

    class_indices = [
        np.where(labels == y)[0]
        for y in range(num_classes)
    ]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):

        np.random.shuffle(class_indices[c])

        proportions = np.random.dirichlet(
            alpha * np.ones(num_clients)
        )

        proportions = (
            np.cumsum(proportions) *
            len(class_indices[c])
        ).astype(int)[:-1]

        split = np.split(class_indices[c], proportions)

        for cid in range(num_clients):
            client_indices[cid].extend(split[cid])

    return client_indices

DIRICHLET_ALPHA = 0.5

# ===============================
# TREINO LOCAL
# ===============================

def client_update(dataset_name, model, loader, epochs, lr):
    model = copy.deepcopy(model)
    model.train()

    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    total_loss, total_samples = 0.0, 0

    DATASET_INPUT_MAP = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image", "Gowalla": "sequence",
                         "WISDM-W": "sequence", "ImageNet": "image", "ImageNet10": "image", "wikitext": "sequence",
                         "Foursquare": "sequence"}

    key = DATASET_INPUT_MAP[dataset_name]

    for _ in range(epochs):
        for batch in loader:
            # logger.info("""dentro {} labels {}""".format(images, labels))
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

    DATASET_INPUT_MAP = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image", "Gowalla": "sequence",
                         "WISDM-W": "sequence", "ImageNet": "image", "ImageNet10": "image", "wikitext": "sequence",
                         "Foursquare": "sequence"}

    key = DATASET_INPUT_MAP[dataset_name]

    for batch in loader:
        x = batch[key]
        y = batch["label"]
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total

# ===============================
# AVALIAÇÃO GLOBAL (MODELO EM TODOS OS CLIENTES)
# ===============================

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

# ===============================
# LOOP FEDERADO (PADRÃO RAWCS)
# ===============================

# ===============================
# CARREGAMENTO DOS DATASETS BASE
# ===============================

cifar_train = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

gtsrb_train = torchvision.datasets.GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=transform_train
)

from pathlib import Path

Path("results/").mkdir(parents=True, exist_ok=True)


current_regime = get_regime(regime)
apply_regime(regime)

results_rows = []

for fold in range(NUM_FOLDS):

    print("\n===================================================")
    print(f"🚀 INICIANDO FOLD {fold}")
    print("===================================================")

    # Seed fixa por fold
    fold_seed = BASE_SEED + fold * 1000
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)

    # ===============================
    # DIRICHLET POR DATASET (UMA VEZ)
    # ===============================

    cifar_client_indices = dirichlet_split(
        cifar_train,
        NUM_CLIENTS,
        DIRICHLET_ALPHA
    )

    gtsrb_client_indices = dirichlet_split(
        gtsrb_train,
        NUM_CLIENTS,
        DIRICHLET_ALPHA
    )

    train_loaders = defaultdict(dict)
    test_loaders = defaultdict(dict)

    for cid in range(NUM_CLIENTS):
        # ---------------- CIFAR ----------------
        cifar_train_idx, cifar_test_idx = split_client_train_test(
            cifar_client_indices[cid],
            train_ratio=0.8
        )

        train_loaders[cid]["cifar"] = DataLoader(
            Subset(cifar_train, cifar_train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        test_loaders[cid]["cifar"] = DataLoader(
            Subset(cifar_train, cifar_test_idx),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # ---------------- GTSRB ----------------
        gtsrb_train_idx, gtsrb_test_idx = split_client_train_test(
            gtsrb_client_indices[cid],
            train_ratio=0.8
        )

        train_loaders[cid]["gtsrb"] = DataLoader(
            Subset(gtsrb_train, gtsrb_train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        test_loaders[cid]["gtsrb"] = DataLoader(
            Subset(gtsrb_train, gtsrb_test_idx),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

    reset_experiment_state()

    current_regime = get_regime(regime)
    apply_regime(regime)

    # ===============================
    # NOVA PARTIÇÃO POR FOLD
    # ===============================

    # seed específica do fold para partição
    np.random.seed(fold_seed)

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

    for rnd in range(1, ROUNDS + 1):

        # Seed fixa por rodada
        round_seed = fold_seed + rnd
        random.seed(round_seed)
        np.random.seed(round_seed)
        torch.manual_seed(round_seed)

        print(f"\n🔄 FOLD {fold} | RODADA {rnd}")

        real_training_counter = {m: 0 for m in global_models}

        # ---------------------------
        # 1) RAWCS
        # ---------------------------
        assignments = rawcs_multi_model(rnd)

        model_updates = {
            "cifar": [],
            "gtsrb": []
        }

        # ---------------------------
        # 2) Treino local
        # ---------------------------
        for cid, model_name in assignments.items():

            if not R(cid, model_name):
                continue

            real_training_counter[model_name] += 1

            local_model = copy.deepcopy(global_models[model_name])
            dataset_name = {"cifar": "CIFAR10", "gtsrb": "GTSRB"}[model_name]
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

        # ---------------------------
        # 3) FedAvg
        # ---------------------------
        for model_name, updates in model_updates.items():
            if updates:
                new_state = fedavg(updates)
                global_models[model_name].load_state_dict(new_state)

        if sum(real_training_counter.values()) == 0:
            print("💀 COLAPSO GLOBAL DO SISTEMA")

        for m in global_models:
            if real_training_counter[m] == 0:
                print(f"💀 Modelo {m.upper()} sem treino real nesta rodada")

        # ---------------------------
        # 4) Avaliação global
        # ---------------------------
        for model_name, model in global_models.items():
            dataset_name = {"cifar": "CIFAR10", "gtsrb": "GTSRB"}[model_name]
            global_acc = evaluate_global_model(
                model,
                {
                    cid: test_loaders[cid][model_name]
                    for cid in range(NUM_CLIENTS)
                },
                dataset_name
            )

            print(
                f"📊 FOLD {fold} | Modelo {model_name.upper()} | "
                f"Acurácia global média: {global_acc:.4f}"
            )

            global_acc_history[model_name].append(global_acc)

            row_data = {
                "fold": fold,
                "round": rnd,
                "dataset": model_name,
                "global_acc": global_acc,
                "viable_pairs": proposta_logs["viable_pairs"][-1],
                "viable_clients": proposta_logs["viable_clients"][-1],
                "fallback_rate": proposta_logs["fallback_rate"][-1],
                "clients_selected": real_training_counter[model_name]
            }

            filename = f"results/proposta_{model_name}_regime_{regime}.csv"

            append_result_to_csv(row_data, filename)

        clients_per_model = {m: 0 for m in global_models}
        for m in assignments.values():
            clients_per_model[m] += 1

        proposta_logs["clients_per_model"].append(clients_per_model)

        print(
            f"🧠 RAWCS | "
            f"Viable pairs: {proposta_logs['viable_pairs'][-1]} | "
            f"Viable clients: {proposta_logs['viable_clients'][-1]} | "
            f"Fallback rate: {proposta_logs['fallback_rate'][-1]:.2f}"
        )

        print("🧠 Viable pairs por modelo:",
              proposta_logs["viable_pairs_per_model"][-1])

        print("⚙️ Custo médio por modelo:",
              proposta_logs["avg_cost"][-1])

        print("⏱️ Tempo médio por modelo:",
              proposta_logs["avg_train_time"][-1])


