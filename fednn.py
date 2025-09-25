"""
Client-side adaptation to concept drift (simulation)
Implements the algorithm from:
"Client-Side Adaptation to Concept Drift in Federated Learning"
(EMAs on local loss to compute per-client learning rates).
Uses MNIST and simulates sudden drift (class-swap) or incremental drift (Gaussian blur).

Requirements:
    pip install torch torchvision numpy

Run:
    python client_side_adapt.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import GaussianBlur
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import random
import copy
from collections import defaultdict
import math

# ---------------------------
# Config
# ---------------------------
NUM_CLIENTS = 10
PARTICIPATION_RATE = 0.3  # 30%
CLIENTS_PER_ROUND = max(1, int(NUM_CLIENTS * PARTICIPATION_RATE))
NUM_ROUNDS = 20            # total FL rounds (increase for fuller sim)
LOCAL_EPOCHS = 1            # epochs per client update
BATCH_SIZE = 32
LR_SERVER_INIT = 0.2        # eta0 from paper
LR_DECAY = 0.99             # d (server-side decay per round)
L_EST = 20                  # number of forward-pass samples for LR estimation per client
BETA1 = 0.7                 # EMA decay for mean loss
BETA2 = 0.3                 # EMA decay for variance
BETA3 = 0.9                 # EMA decay for variance ratio
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose drift type and parameters
DRIFT_TYPE = "sudden"       # "sudden" or "incremental" or None
DRIFT_ROUND = 30            # r0: when drift starts
# sudden drift parameters (swap pairs)
CLASS_SWAP_PAIRS = [(0, 1), (2, 3)]
# incremental drift parameters
TD = 20                     # duration of incremental drift
SIGMA_MAX = 5.0             # gaussian blur max sigma

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Dataset and client partition
# ---------------------------
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_base)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_base)
test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

# Partition training indices among clients (simple random split)
indices = np.random.permutation(len(mnist_train))
client_indices = np.array_split(indices, NUM_CLIENTS)

# Helper: dataset wrapper that can apply transformations (e.g., blur) and label mapping (for swap)
class ClientDataset(Dataset):
    def __init__(self, base_dataset, indices, blur_sigma=0.0, label_map=None):
        self.base = base_dataset
        self.indices = list(indices)
        self.blur_sigma = blur_sigma
        self.label_map = label_map  # dict {old: new}
        # We'll reuse base transform for normalize/tensor; apply blur before ToTensor to PIL image.

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        # img is already a tensor because base transform included ToTensor.
        # To apply GaussianBlur, convert to PIL-like or use tensor blur approx.
        if self.blur_sigma and self.blur_sigma > 0:
            # convert tensor back to PIL-like via torchvision.functional
            # But GaussianBlur expects PIL or torch.Tensor CHW float in torchvision >=0.9 supports Tensor
            # apply blur on tensor (C,H,W)
            img = TF.gaussian_blur(img, kernel_size=11, sigma=(self.blur_sigma, self.blur_sigma))
        if self.label_map is not None:
            label = self.label_map.get(label, label)
        return img, label

# ---------------------------
# Model architecture (as in paper: small conv net)
# ---------------------------
class SmallCNN(nn.Module):
    def __init__(self, channels=200):  # use smaller to be faster; paper used 200 for MNIST or 32 for CIFAR
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(2, 2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=1)
        # compute flattened size approx: for MNIST 28x28 -> after conv+pool ~ ...
        # We'll flatten dynamically in forward
        self.fc1 = nn.Linear(channels * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Federated utilities
# ---------------------------
def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += float(criterion(out, y))
            preds = out.argmax(dim=1)
            correct += int((preds == y).sum())
            total += y.size(0)
    return loss_sum / total, correct / total

def local_train(model, train_loader, local_lr, local_epochs, device):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=local_lr)
    criterion = nn.CrossEntropyLoss()
    for e in range(local_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    # evaluate on local data
    return evaluate_model(model, train_loader, device)

def federated_average(models_state_dicts, weights=None):
    # simple average (weights optional)
    avg = copy.deepcopy(models_state_dicts[0])
    for k in avg.keys():
        avg[k] = avg[k].float() * 0.0
    n = len(models_state_dicts)
    for st in models_state_dicts:
        for k in st.keys():
            avg[k] += st[k].float() / n
    return avg

# ---------------------------
# Client state (EMAs etc.)
# ---------------------------
class ClientState:
    def __init__(self):
        # EMAs initialized to 0 as in paper
        self.M = 0.0
        self.V = 0.0
        self.R = 0.0
        self.rounds_seen = 0  # used for bias correction power
        # store last bias-corrected values for denominator (for ratio)
        self.M_hat_prev = 0.0
        self.V_hat_prev = 0.0

# Initialize client states
client_states = [ClientState() for _ in range(NUM_CLIENTS)]

# ---------------------------
# Server-side main loop
# ---------------------------

# initialize global model
global_model = SmallCNN().to(DEVICE)
# initial server LR
eta0 = LR_SERVER_INIT

# Precompute per-client static datasets (we will wrap them with dynamic drift transforms per round)
# store the indices; we'll create ClientDataset instances as needed to apply blur/swap
client_train_datasets_base = [client_indices[i] for i in range(NUM_CLIENTS)]

print(f"Federated simulation: {NUM_CLIENTS} clients, {CLIENTS_PER_ROUND} selected/round, {NUM_ROUNDS} rounds.")
print(f"Drift type: {DRIFT_TYPE}, drift start round {DRIFT_ROUND}")

criterion = nn.CrossEntropyLoss()

for r in range(1, NUM_ROUNDS + 1):
    # server LR decay (stepwise via multiplicative decay)
    eta_r = eta0 * (LR_DECAY ** (r-1))
    if eta_r > eta0:  # just in case
        eta_r = eta0

    # ---------- Additional stage: update learning rate on each client ----------
    # For each client, server requests UpdateLearningRate: client computes average loss over L samples (forward pass)
    for cid in range(NUM_CLIENTS):
        # Prepare client dataset with drift applied according to current round and selected DRIFT_TYPE
        label_map = None
        blur_sigma = 0.0

        if DRIFT_TYPE == "sudden" and r >= DRIFT_ROUND:
            # apply class-swap mapping to ALL client data permanently (as in paper)
            # build mapping dict once: swap pairs
            label_map = {}
            for a, b in CLASS_SWAP_PAIRS:
                label_map[a] = b
                label_map[b] = a
        elif DRIFT_TYPE == "incremental" and r >= DRIFT_ROUND:
            # compute sigma increasing linearly from 0 to SIGMA_MAX over TD rounds
            t_since = r - DRIFT_ROUND
            if t_since >= TD:
                blur_sigma = SIGMA_MAX
            else:
                blur_sigma = (t_since / TD) * SIGMA_MAX

        client_dataset = ClientDataset(mnist_train, client_train_datasets_base[cid], blur_sigma=blur_sigma, label_map=label_map)
        # sample L_EST random samples (or all if fewer)
        n_samples = min(L_EST, len(client_dataset))
        if n_samples == 0:
            l_k = 0.0
        else:
            sample_idxs = random.sample(range(len(client_dataset)), n_samples)
            loader_est = DataLoader(Subset(client_dataset, sample_idxs), batch_size=n_samples, shuffle=False)
            # forward-pass loss estimate using global model (no grad)
            global_model.eval()
            l_sum = 0.0
            with torch.no_grad():
                for x_s, y_s in loader_est:
                    x_s, y_s = x_s.to(DEVICE), y_s.to(DEVICE)
                    out = global_model(x_s)
                    l_sum += float(nn.CrossEntropyLoss(reduction='sum')(out, y_s))
            l_k = l_sum / n_samples

        # Update EMAs for client cid following equations in paper
        st = client_states[cid]
        st.rounds_seen += 1
        # Mr = lr * (1 - beta1) + Mr-1 * beta1
        st.M = l_k * (1.0 - BETA1) + st.M * BETA1
        # bias-corrected M_hat = M / (1 - beta1^r)
        denom1 = (1 - (BETA1 ** st.rounds_seen)) if st.rounds_seen > 0 else 1.0
        M_hat = st.M / (denom1 + 1e-12)

        # Vr = (lr - M_{r-1})^2 * (1 - beta2) + Vr-1 * beta2
        # note: paper uses Mr-1 inside; using M_hat_prev to follow bias-corrected prev
        prev_M = st.M_hat_prev if st.rounds_seen > 1 else 0.0
        st.V = ((l_k - prev_M) ** 2) * (1.0 - BETA2) + st.V * BETA2
        denom2 = (1 - (BETA2 ** st.rounds_seen)) if st.rounds_seen > 0 else 1.0
        V_hat = st.V / (denom2 + 1e-12)

        # Rr = (V_hat / V_hat_prev) * (1 - beta3) + R_{r-1} * beta3  (if V_hat_prev == 0 use 1*(1-beta3)+R_prev*beta3)
        if st.V_hat_prev == 0.0:
            ratio = 1.0
        else:
            ratio = (V_hat / (st.V_hat_prev + 1e-12))
        st.R = ratio * (1.0 - BETA3) + st.R * BETA3
        denom3 = (1 - (BETA3 ** st.rounds_seen)) if st.rounds_seen > 0 else 1.0
        R_hat = st.R / (denom3 + 1e-12)

        # compute local lr: eta_lr = min(eta0, eta_r * R_hat)
        eta_lr = min(eta0, eta_r * (R_hat if R_hat > 0 else 1.0))

        # store bias-corrected prev values for next round
        st.M_hat_prev = M_hat
        st.V_hat_prev = V_hat

        # store computed local lr in client state for use when selected to train
        st.local_lr = float(eta_lr)

    # ---------- Select participating clients ----------
    selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)

    # ---------- Each selected client performs ClientUpdate (local training) ----------
    local_states = []
    local_state_sizes = []
    print(f"\nRound {r:03d} | server eta_r={eta_r:.5f} | selected clients: {selected}")
    for cid in selected:
        # rebuild client dataset with drift applied same as above
        label_map = None
        blur_sigma = 0.0
        if DRIFT_TYPE == "sudden" and r >= DRIFT_ROUND:
            label_map = {}
            for a, b in CLASS_SWAP_PAIRS:
                label_map[a] = b
                label_map[b] = a
        elif DRIFT_TYPE == "incremental" and r >= DRIFT_ROUND:
            t_since = r - DRIFT_ROUND
            if t_since >= TD:
                blur_sigma = SIGMA_MAX
            else:
                blur_sigma = (t_since / TD) * SIGMA_MAX

        client_dataset = ClientDataset(mnist_train, client_train_datasets_base[cid], blur_sigma=blur_sigma, label_map=label_map)
        client_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # local model copy
        local_model = copy.deepcopy(global_model).to(DEVICE)

        # get client's computed local lr
        eta_local = getattr(client_states[cid], "local_lr", eta_r)
        # ensure small lower bound
        eta_local = max(1e-6, eta_local)

        # ClientUpdate: do local epochs with local_lr
        local_train_loss, local_train_acc = local_train(local_model, client_loader, eta_local, LOCAL_EPOCHS, DEVICE)
        print(f"  Client {cid:02d} -> local loss: {local_train_loss:.4f} | local acc: {local_train_acc:.4f} | eta_local: {eta_local:.5f}")

        local_states.append(copy.deepcopy(local_model.state_dict()))
        local_state_sizes.append(len(client_dataset))

    # ---------- Aggregate (FedAvg simple average) ----------
    if len(local_states) > 0:
        global_state = federated_average(local_states)
        global_model.load_state_dict(global_state)

    # ---------- Evaluate global model on test set ----------
    test_loss, test_acc = evaluate_model(global_model, test_loader, DEVICE)
    print(f"  GLOBAL EVAL -> test loss: {test_loss:.4f} | test acc: {test_acc:.4f}")

print("\nSimulation finished.")
