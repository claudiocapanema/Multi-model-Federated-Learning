# ==============================
# PART 1 - Imports, Metrics, Model, Datasets
# ==============================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import DataLoader, Subset

import csv
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 40
CLIENT_FRAC = 0.3
NUM_SELECTED = int(NUM_CLIENTS * CLIENT_FRAC)  # 12
ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01

# Rotation hyperparameters
LAMBDA = 0.3   # penalize repetition
BETA = 0.2     # reward rotation

BASE_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(BASE_SEED)

# ==============================
# Metrics
# ==============================

def jain_fairness(x):
    x = np.array(x)
    return (x.sum() ** 2) / (len(x) * (x ** 2).sum() + 1e-8)

def avg_efficiency(clients, model_name):
    return np.mean([c.efficiency[model_name] for c in clients])

def print_client_ids(client_list):
    return [c.cid for c in client_list]

def client_entropy(client):
    total = client.train_count["cifar"] + client.train_count["gtsrb"]
    if total == 0:
        return 0.0
    p_cifar = client.train_count["cifar"] / total
    p_gtsrb = client.train_count["gtsrb"] / total

    entropy = 0
    for p in [p_cifar, p_gtsrb]:
        if p > 0:
            entropy -= p * np.log(p + 1e-8)
    return entropy


def compute_switch_rate(prev_assignments, curr_assignments):
    switches = 0
    total = 0
    for cid in curr_assignments:
        if cid in prev_assignments:
            if prev_assignments[cid] != curr_assignments[cid]:
                switches += 1
            total += 1
    return switches / max(total, 1)


# ==============================
# Simple CNN Model
# ==============================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==============================
# Load datasets
# ==============================

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

gtsrb_train = torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=transform)
gtsrb_test = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=transform)
# ==============================
# PART 2 - Dirichlet Split, Client, Training, Evaluation
# ==============================

def dirichlet_split(dataset, num_clients, alpha=0.5):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        split = np.split(class_indices[c], proportions)

        for i in range(num_clients):
            client_indices[i].extend(split[i])

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


# Create client splits
cifar_clients = dirichlet_split(cifar_train, NUM_CLIENTS, alpha=0.1)
gtsrb_clients = dirichlet_split(gtsrb_train, NUM_CLIENTS, alpha=0.1)


# ==============================
# Client class
# ==============================

class Client:
    def __init__(self, cid, cifar_idxs, gtsrb_idxs):
        self.cid = cid

        self.cifar_loader = DataLoader(
            Subset(cifar_train, cifar_idxs),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        self.gtsrb_loader = DataLoader(
            Subset(gtsrb_train, gtsrb_idxs),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # efficiency per model
        self.efficiency = {
            "cifar": 1.0,
            "gtsrb": 1.0
        }

        # history for rotation
        self.train_count = {"cifar": 0, "gtsrb": 0}
        self.last_trained = {"cifar": -1, "gtsrb": -1}

    def train_local(self, model, dataloader):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0

        for _ in range(LOCAL_EPOCHS):
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()

        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        return avg_loss, acc

    def update_efficiency(self, model_name, loss_before, loss_after, cost):
        # normalized relative improvement
        delta = (loss_before - loss_after) / (loss_before + 1e-8)
        new_eff = delta / cost
        self.efficiency[model_name] = max(new_eff, 1e-6)

    def update_history(self, model_name, round_id):
        self.train_count[model_name] += 1
        self.last_trained[model_name] = round_id


# ==============================
# Initialize clients
# ==============================

clients = []
for i in range(NUM_CLIENTS):
    clients.append(Client(i, cifar_clients[i], gtsrb_clients[i]))


# ==============================
# Evaluation function
# ==============================

def evaluate_global(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

    return total_loss / len(loader), correct / total

# ==============================
# PART 3 - BASELINE Federated Loop (Random Selection)
# ==============================

# Cost of each model
COST = {
    "cifar": 1.0,
    "gtsrb": 2.0
}

NUM_CIFAR_CLIENTS = int(NUM_SELECTED / 2)   # 6
NUM_GTSRB_CLIENTS = int(NUM_SELECTED / 2)   # 6

# ==============================
# Global models
# ==============================

global_cifar = SimpleCNN(num_classes=10).to(DEVICE)
global_gtsrb = SimpleCNN(num_classes=43).to(DEVICE)

# ==============================
# Federated Averaging
# ==============================

def fedavg(weights):
    avg = {}
    for k in weights[0].keys():
        avg[k] = sum(w[k] for w in weights) / len(weights)
    return avg

# ==============================
# Federated training loop (Baseline)
# ==============================

CIFAR_CSV = "baseline_cifar.csv"
GTSRB_CSV = "baseline_gtsrb.csv"

csv_header = [
    "round",
    "loss",
    "accuracy",
    "avg_efficiency",
    "effective_resource",
    "fairness",
    "switch_rate",
    "avg_client_entropy",
    "min_client_entropy",
    "client_jain_fairness"
]

# Clean CSVs at start
for file in [CIFAR_CSV, GTSRB_CSV]:
    with open(file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)


cifar_log = []
gtsrb_log = []

prev_assignments = {}

for rnd in range(ROUNDS):
    set_seed(BASE_SEED + rnd)
    print(f"\n================ ROUND {rnd+1} ================")

    # Step 1: randomly select clients (availability)
    selected = random.sample(clients, NUM_SELECTED)

    # Step 2: random split into two groups
    random.shuffle(selected)
    cifar_selected = selected[:NUM_CIFAR_CLIENTS]
    gtsrb_selected = selected[NUM_CIFAR_CLIENTS:NUM_CIFAR_CLIENTS + NUM_GTSRB_CLIENTS]

    # Print selected clients
    print(f"CIFAR selected clients ({len(cifar_selected)}): {print_client_ids(cifar_selected)}")
    print(f"GTSRB selected clients ({len(gtsrb_selected)}): {print_client_ids(gtsrb_selected)}")

    # Current assignments
    current_assignments = {}
    for c in cifar_selected:
        current_assignments[c.cid] = "cifar"
    for c in gtsrb_selected:
        current_assignments[c.cid] = "gtsrb"

    # Diversity metrics
    if rnd == 0:
        switch_rate = 0.0
    else:
        switch_rate = compute_switch_rate(prev_assignments, current_assignments)

    entropies = [client_entropy(c) for c in clients]
    avg_entropy = np.mean(entropies)
    min_entropy = np.min(entropies)

    client_usage = [c.train_count["cifar"] + c.train_count["gtsrb"] for c in clients]
    jain_clients = jain_fairness(client_usage)

    prev_assignments = current_assignments.copy()

    cifar_weights = []
    gtsrb_weights = []

    # ================= CIFAR =================
    for client in cifar_selected:
        local_model = SimpleCNN(num_classes=10).to(DEVICE)
        local_model.load_state_dict(global_cifar.state_dict())

        loss_before, _ = evaluate_global(local_model, cifar_test)
        loss_after, acc = client.train_local(local_model, client.cifar_loader)

        client.update_efficiency("cifar", loss_before, loss_after, COST["cifar"])
        client.update_history("cifar", rnd)

        cifar_weights.append(local_model.state_dict())

    # ================= GTSRB =================
    for client in gtsrb_selected:
        local_model = SimpleCNN(num_classes=43).to(DEVICE)
        local_model.load_state_dict(global_gtsrb.state_dict())

        loss_before, _ = evaluate_global(local_model, gtsrb_test)
        loss_after, acc = client.train_local(local_model, client.gtsrb_loader)

        client.update_efficiency("gtsrb", loss_before, loss_after, COST["gtsrb"])
        client.update_history("gtsrb", rnd)

        gtsrb_weights.append(local_model.state_dict())

    # ================= Aggregate =================
    if cifar_weights:
        global_cifar.load_state_dict(fedavg(cifar_weights))
    if gtsrb_weights:
        global_gtsrb.load_state_dict(fedavg(gtsrb_weights))

    # ================= Evaluate =================
    cifar_loss, cifar_acc = evaluate_global(global_cifar, cifar_test)
    gtsrb_loss, gtsrb_acc = evaluate_global(global_gtsrb, gtsrb_test)

    # ================= Metrics =================
    R_eff_cifar = sum(c.efficiency["cifar"] * COST["cifar"] for c in cifar_selected)
    R_eff_gtsrb = sum(c.efficiency["gtsrb"] * COST["gtsrb"] for c in gtsrb_selected)

    fairness = jain_fairness([R_eff_cifar, R_eff_gtsrb])

    avg_eff_cifar = avg_efficiency(cifar_selected, "cifar")
    avg_eff_gtsrb = avg_efficiency(gtsrb_selected, "gtsrb")
    worst_acc = min(cifar_acc, gtsrb_acc)

    # ================= CSV Logging =================

    # ================= CSV Logging (per round) =================

    with open(CIFAR_CSV, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            rnd + 1,
            cifar_loss,
            cifar_acc,
            avg_eff_cifar,
            R_eff_cifar,
            fairness,
            switch_rate,
            avg_entropy,
            min_entropy,
            jain_clients
        ])

    with open(GTSRB_CSV, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            rnd + 1,
            gtsrb_loss,
            gtsrb_acc,
            avg_eff_gtsrb,
            R_eff_gtsrb,
            fairness,
            switch_rate,
            avg_entropy,
            min_entropy,
            jain_clients
        ])

    print("\n--- Metrics (Baseline) ---")
    print(f"CIFAR   -> Loss: {cifar_loss:.4f}, Acc: {cifar_acc:.4f}")
    print(f"GTSRB   -> Loss: {gtsrb_loss:.4f}, Acc: {gtsrb_acc:.4f}")
    print(f"Worst-model Accuracy: {worst_acc:.4f}")

    print(f"Jain Fairness (effective): {fairness:.4f}")
    print(f"Avg efficiency CIFAR: {avg_eff_cifar:.6f}")
    print(f"Avg efficiency GTSRB: {avg_eff_gtsrb:.6f}")

    print(f"Switch rate (client rotation): {switch_rate:.4f}")
    print(f"Avg client entropy: {avg_entropy:.4f}")
    print(f"Min client entropy: {min_entropy:.4f}")
    print(f"Client Jain Fairness (usage): {jain_clients:.4f}")