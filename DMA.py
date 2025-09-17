# federated_cifar10_label_drift_with_mitigation.py
# Execute em ambiente com PyTorch; GPU recomendada.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

# ---------------- Config ----------------
seed = 42
torch.manual_seed(seed)
random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_clients = 8
rounds = 20
clients_per_round = 4             # número de clientes selecionados por rodada (síncrono)
batch_size = 64
drift_round = 8                   # rodada em que o label drift ocorre
label_shift = 3                   # exemplo: mapping y -> (y + label_shift) % 10
drift_detection_window = 1        # quantas últimas avaliações usar para estimar mudança (simples)
initial_local_epochs = 1
mitigation_extra_epochs = 2       # epochs extras para clientes com alto drift
mitigation_weight_factor = 2.0    # quanto a atualização do cliente de alto-drift é 'pesada' na agregação
drift_threshold = 0.05            # mínima mudança relativa na loss para considerar 'alto drift'
# ----------------------------------------

# ---------- Transform & datasets ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# ---------- Util: RelabelledSubset ----------
class RelabelledSubset(torch.utils.data.Dataset):
    """Subset that applies a label mapping (mapping: list of length num_classes)"""
    def __init__(self, base_dataset, indices, mapping=None):
        self.base = base_dataset
        self.indices = list(indices)
        self.mapping = mapping  # e.g., [1,2,3,...] or None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.base[real_idx]
        if self.mapping is not None:
            y = self.mapping[y]
        return x, y

# ---------- Model ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- Federated elements ----------
class Client:
    def __init__(self, cid, train_indices, val_indices, global_models, initial_mapping_taskB=None):
        self.id = cid
        self.train_indices = train_indices
        self.val_indices = val_indices  # small validation set to estimate client loss
        self.global_models = global_models  # dict of model instances
        # inicial loaders
        self.loader_taskA = DataLoader(Subset(trainset, train_indices), batch_size=batch_size, shuffle=True)
        dsB = RelabelledSubset(trainset, train_indices, mapping=initial_mapping_taskB)
        self.loader_taskB = DataLoader(dsB, batch_size=batch_size, shuffle=True)
        # validation loaders (taskB)
        dsB_val = RelabelledSubset(trainset, val_indices, mapping=initial_mapping_taskB)
        self.val_loader_taskB = DataLoader(dsB_val, batch_size=batch_size, shuffle=False)
        # histórico simples para drift detection: guarda últimas perdas (avg)
        self.last_val_loss = None

    def set_taskB_mapping(self, mapping):
        dsB = RelabelledSubset(trainset, self.train_indices, mapping=mapping)
        self.loader_taskB = DataLoader(dsB, batch_size=batch_size, shuffle=True)
        dsB_val = RelabelledSubset(trainset, self.val_indices, mapping=mapping)
        self.val_loader_taskB = DataLoader(dsB_val, batch_size=batch_size, shuffle=False)

    def local_train(self, task, epochs=1, lr=0.01):
        model = deepcopy(self.global_models[task]).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        model.train()

        loader = self.loader_taskA if task == "taskA" else self.loader_taskB
        for _ in range(epochs):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def evaluate_taskB_val_loss(self):
        # calcula perda média no val_loader_taskB (usado para detectar drift)
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        model = deepcopy(self.global_models["taskB"]).to(device)
        model.eval()
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for X, y in self.val_loader_taskB:
                X, y = X.to(device), y.to(device)
                out = model(X)
                total_loss += loss_fn(out, y).item()
                total += y.size(0)
        if total == 0:
            return None
        avg = total_loss / total
        return avg

class Server:
    def __init__(self, initial_models):
        self.global_models = initial_models

    def aggregate_weighted(self, updates, weights):
        # updates: list of state_dicts (selected clients), weights: list of scalars
        # produce weighted average
        total_w = sum(weights)
        if total_w == 0:
            return
        avg = deepcopy(updates[0])
        for k in avg.keys():
            # initialize to zero
            avg[k] = torch.zeros_like(avg[k], dtype=torch.float32)
        for sd, w in zip(updates, weights):
            for k in avg.keys():
                avg[k] += sd[k].float() * (w/total_w)
        # load averaged states into global_models (same for both tasks individually)
        return avg

    def evaluate(self, test_loaders):
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        results = {}
        for task, loader in test_loaders.items():
            model = self.global_models[task].to(device)
            model.eval()
            correct = 0
            total = 0
            loss_sum = 0.0
            with torch.no_grad():
                for X, y in loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    loss_sum += loss_fn(out, y).item()
                    pred = out.argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            results[task] = {"acc": correct/total, "loss": loss_sum/total}
        return results

# ---------- Prepare client partitions ----------
all_indices = list(range(len(trainset)))
random.shuffle(all_indices)

# partition: give each client a disjoint chunk (simple)
chunks = [all_indices[i::num_clients] for i in range(num_clients)]
# small validation indices per client (take 5% of client's chunk)
val_chunks = []
for c_idx in range(num_clients):
    c_indices = chunks[c_idx]
    random.shuffle(c_indices)
    split = int(0.05 * len(c_indices))
    val_chunks.append(c_indices[:split] if split>0 else c_indices[:1])  # ensure at least 1

# ---------- Create initial global models ----------
global_models = {"taskA": SimpleCNN(10).to(device), "taskB": SimpleCNN(10).to(device)}
server = Server(global_models)

# ---------- Create clients ----------
clients = []
for cid in range(num_clients):
    c = Client(cid, chunks[cid], val_chunks[cid], global_models, initial_mapping_taskB=None)
    clients.append(c)

# ---------- Test loaders ----------
test_loader_taskA = DataLoader(testset, batch_size=batch_size, shuffle=False)
test_loader_taskB = DataLoader(testset, batch_size=batch_size, shuffle=False)  # will relabel after drift

# ---------- Training loop with label-drift + mitigation ----------
history = {"taskA": {"acc": [], "loss": []}, "taskB": {"acc": [], "loss": []}}
# track per-client last val loss for drift detection
for c in clients:
    c.last_val_loss = c.evaluate_taskB_val_loss()

current_mapping = None

for r in range(rounds):
    print(f"\n=== Round {r+1}/{rounds} ===")

    # ------------- Apply label drift at drift_round -------------
    if r == drift_round:
        mapping = [(i + label_shift) % 10 for i in range(10)]
        current_mapping = mapping
        for c in clients:
            c.set_taskB_mapping(mapping)
            # recompute validation loss AFTER applying mapping (this represents observed loss after drift)
            c.last_val_loss = c.evaluate_taskB_val_loss()
        # relabel test loader for taskB as well (so evaluation matches new labels)
        test_loader_taskB = DataLoader(RelabelledSubset(testset, list(range(len(testset))), mapping=current_mapping),
                                       batch_size=batch_size, shuffle=False)
        print(">>> LABEL-DRIFT applied to Task B: mapping =", current_mapping)

    # ------------- Drift detection step: compute current val losses using server model -------------
    # Evaluate per-client current val loss (on task B) using current global model
    client_current_losses = []
    for c in clients:
        loss = c.evaluate_taskB_val_loss()
        client_current_losses.append(loss)

    # ------------- Compute per-client drift score -------------
    # drift_score = max(0, (current_loss - last_loss)/last_loss)  (relative increase)
    drift_scores = []
    for idx, c in enumerate(clients):
        prev = c.last_val_loss
        curr = client_current_losses[idx]
        if prev is None or curr is None or prev == 0:
            score = 0.0
        else:
            score = max(0.0, (curr - prev) / prev)
        drift_scores.append(score)

    # Update client's stored last loss to current for next round comparison
    for idx, c in enumerate(clients):
        c.last_val_loss = client_current_losses[idx]

    # ------------- Decide which clients are "high drift" -------------
    high_drift_flags = [ (score >= drift_threshold) for score in drift_scores ]
    # show summary
    print("Drift scores (per client):", ["{:.3f}".format(s) for s in drift_scores])
    print("High-drift flags:", high_drift_flags)

    # ------------- Selection / scheduling: prioritize high-drift clients -------------
    # We build a selection pool where high-drift clients appear more times => higher selection prob.
    pool = []
    for idx, c in enumerate(clients):
        weight = 3 if high_drift_flags[idx] else 1   # simple multiplicative factor
        pool += [idx] * weight
    # choose clients_per_round unique clients (if pool smaller, fallback)
    selected_idx = set()
    attempts = 0
    while len(selected_idx) < min(clients_per_round, num_clients) and attempts < 1000:
        picked = random.choice(pool)
        selected_idx.add(picked)
        attempts += 1
    selected_idx = list(selected_idx)
    selected_clients = [clients[i] for i in selected_idx]
    print("Selected clients:", selected_idx)

    # ------------- For selected clients, decide local epochs and aggregation weights -------------
    client_updates_taskA = []
    client_updates_taskB = []
    client_weights_taskA = []
    client_weights_taskB = []

    for idx in selected_idx:
        c = clients[idx]
        # if high drift: give extra local epochs for TaskB and larger aggregation weight
        if high_drift_flags[idx]:
            epochs_B = initial_local_epochs + mitigation_extra_epochs
            agg_weight = mitigation_weight_factor
        else:
            epochs_B = initial_local_epochs
            agg_weight = 1.0

        # Train Task A (keep default resources)
        sdA = c.local_train("taskA", epochs=initial_local_epochs, lr=0.01)
        # Train Task B (possibly extra epochs to adapt to drift)
        sdB = c.local_train("taskB", epochs=epochs_B, lr=0.01)

        client_updates_taskA.append(sdA)
        client_updates_taskB.append(sdB)

        # For simplicity, use same weight for both tasks but could be task-specific
        # here we weight TaskB updates more if client had high-drift
        client_weights_taskA.append(1.0)  # keep TaskA unaffected
        client_weights_taskB.append(agg_weight)

    # ------------- Aggregate (weighted) and update global models -------------
    # Task A aggregation
    if len(client_updates_taskA) > 0:
        avgA = server.aggregate_weighted(client_updates_taskA, client_weights_taskA)
        # load into model
        if avgA is not None:
            server.global_models["taskA"].load_state_dict(avgA)

    # Task B aggregation
    if len(client_updates_taskB) > 0:
        avgB = server.aggregate_weighted(client_updates_taskB, client_weights_taskB)
        if avgB is not None:
            server.global_models["taskB"].load_state_dict(avgB)

    # ------------- Evaluate global models on test sets -------------
    res = server.evaluate({"taskA": test_loader_taskA, "taskB": test_loader_taskB})
    print(f"Global eval - TaskA: acc={res['taskA']['acc']:.4f}, loss={res['taskA']['loss']:.4f} | "
          f"TaskB: acc={res['taskB']['acc']:.4f}, loss={res['taskB']['loss']:.4f}")
    history["taskA"]["acc"].append(res["taskA"]["acc"])
    history["taskA"]["loss"].append(res["taskA"]["loss"])
    history["taskB"]["acc"].append(res["taskB"]["acc"])
    history["taskB"]["loss"].append(res["taskB"]["loss"])

# ---------- Plot ----------
plt.figure(figsize=(9,4))
plt.plot(history["taskA"]["acc"], label="TaskA acc")
plt.plot(history["taskB"]["acc"], label="TaskB acc")
plt.axvline(drift_round, color='red', linestyle='--', label='drift')
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy per round")
plt.show()

plt.figure(figsize=(9,4))
plt.plot(history["taskA"]["loss"], label="TaskA loss")
plt.plot(history["taskB"]["loss"], label="TaskB loss")
plt.axvline(drift_round, color='red', linestyle='--', label='drift')
plt.xlabel("Round"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss per round")
plt.show()
