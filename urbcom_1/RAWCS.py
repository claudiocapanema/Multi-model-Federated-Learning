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
# ===============================
# CONFIGURAÇÕES GERAIS
# ===============================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 40
ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01

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
# ===============================

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

rawcs_logs = {
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

def rawcs_multi_model():
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

                utility = compute_utility(cid, model_name)

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

# CIFAR-10
cifar_train = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
cifar_test = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

# GTSRB
gtsrb_train = torchvision.datasets.GTSRB(
    root="./data", split="train", download=True, transform=transform_train
)
gtsrb_test = torchvision.datasets.GTSRB(
    root="./data", split="test", download=True, transform=transform_test
)

def split_dataset(dataset, num_clients):
    idx = np.random.permutation(len(dataset))
    splits = np.array_split(idx, num_clients)
    return splits

cifar_train_splits = split_dataset(cifar_train, NUM_CLIENTS)
cifar_test_splits  = split_dataset(cifar_test, NUM_CLIENTS)
gtsrb_train_splits = split_dataset(gtsrb_train, NUM_CLIENTS)
gtsrb_test_splits  = split_dataset(gtsrb_test, NUM_CLIENTS)

train_loaders = defaultdict(dict)
test_loaders  = defaultdict(dict)

for cid in range(NUM_CLIENTS):
    train_loaders[cid]["cifar"] = DataLoader(
        Subset(cifar_train, cifar_train_splits[cid]),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loaders[cid]["cifar"] = DataLoader(
        Subset(cifar_test, cifar_test_splits[cid]),
        batch_size=BATCH_SIZE, shuffle=False
    )

    train_loaders[cid]["gtsrb"] = DataLoader(
        Subset(gtsrb_train, gtsrb_train_splits[cid]),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loaders[cid]["gtsrb"] = DataLoader(
        Subset(gtsrb_test, gtsrb_test_splits[cid]),
        batch_size=BATCH_SIZE, shuffle=False
    )

# ===============================
# TREINO LOCAL
# ===============================

def client_update(model, loader, epochs, lr):
    model = copy.deepcopy(model)
    model.train()

    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    total_loss, total_samples = 0.0, 0

    for _ in range(epochs):
        for x, y in loader:
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
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total

# ===============================
# AVALIAÇÃO GLOBAL (MODELO EM TODOS OS CLIENTES)
# ===============================

@torch.no_grad()
def evaluate_global_model(model, test_loaders):
    model.eval()

    accs = []

    for cid in range(NUM_CLIENTS):
        acc = evaluate_model(model, test_loaders[cid])
        accs.append(acc)

    return float(np.mean(accs))


def fedavg(updates):
    total = sum(n for _, n in updates)
    avg = {}

    for k in updates[0][0].keys():
        avg[k] = sum(state[k] * (n / total) for state, n in updates)

    return avg


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total

# ===============================
# LOOP FEDERADO (PADRÃO RAWCS)
# ===============================

current_regime = get_regime("realistic")
apply_regime("realistic")

for rnd in range(1, ROUNDS + 1):

    print(f"\n🔄 RODADA {rnd}")

    # ---------------------------
    # 1) RAWCS: decide cliente → modelo
    # ---------------------------
    assignments = rawcs_multi_model()

    # buffers de atualização por modelo
    model_updates = {
        "cifar": [],
        "gtsrb": []
    }

    # ---------------------------
    # 2) Treino local + consumo
    # ---------------------------
    for cid, model_name in assignments.items():

        # cliente pode ter sido selecionado mesmo em fallback
        if not R(cid, model_name):
            continue

        local_model = copy.deepcopy(global_models[model_name])

        state_dict, loss, n_samples = client_update(
            local_model,
            train_loaders[cid][model_name],
            epochs=LOCAL_EPOCHS,
            lr=LR
        )

        model_updates[model_name].append((state_dict, n_samples))

        # -----------------------
        # consumo de bateria (RAWCS Eq. 4)
        # -----------------------
        train_time = estimate_training_time(cid, model_name)
        consume_battery(cid, train_time)

        # -----------------------
        # métricas locais
        # -----------------------
        prev_acc = client_acc[cid][model_name]

        acc = evaluate_model(
            local_model,
            test_loaders[cid][model_name]
        )

        client_acc[cid][model_name] = acc
        client_loss[cid][model_name] = loss
        client_delta_acc[cid][model_name] = acc - prev_acc

    # ---------------------------
    # 3) Agregação FedAvg (por modelo)
    # ---------------------------
    for model_name, updates in model_updates.items():
        if updates:
            new_state = fedavg(updates)
            global_models[model_name].load_state_dict(new_state)

    # ---------------------------
    # 4) Avaliação global
    # ---------------------------
    for model_name, model in global_models.items():
        global_acc = evaluate_global_model(
            model,
            test_loaders={
                cid: test_loaders[cid][model_name]
                for cid in range(NUM_CLIENTS)
            }
        )

        print(
            f"📊 Modelo {model_name.upper()} | "
            f"Acurácia global média: {global_acc:.4f}"
        )

    print(
        f"🧠 RAWCS | "
        f"Viable pairs: {rawcs_logs['viable_pairs'][-1]} | "
        f"Viable clients: {rawcs_logs['viable_clients'][-1]} | "
        f"Fallback rate: {rawcs_logs['fallback_rate'][-1]:.2f}"
    )

    print("🧠 Viable pairs por modelo:",
          rawcs_logs["viable_pairs_per_model"][-1])

    print("⚙️ Custo médio por modelo:",
          rawcs_logs["avg_cost"][-1])

    print("⏱️ Tempo médio por modelo:",
          rawcs_logs["avg_train_time"][-1])

