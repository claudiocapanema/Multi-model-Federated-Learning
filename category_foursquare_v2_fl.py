import copy
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# =========================
# DATASETS
# =========================

class NextCategoryDataset(Dataset):
    def __init__(self, hf_dataset):
        self.seq = torch.tensor(hf_dataset["sequence"], dtype=torch.long)
        self.y = torch.tensor(hf_dataset["label"], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq[idx], self.y[idx]


class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# =========================
# DIRICHLET PARTITION
# =========================

def dirichlet_label_partition_min_samples(
    targets,
    num_clients,
    num_classes,
    alpha,
    min_samples_per_client,
    seed,
    max_retries=20,
):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    for _ in range(max_retries):

        class_indices = defaultdict(list)
        for idx, y in enumerate(targets):
            class_indices[int(y)].append(idx)

        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idxs = class_indices[c]
            rng.shuffle(idxs)

            proportions = rng.dirichlet([alpha] * num_clients)
            splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            splits = np.split(idxs, splits)

            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())

        if min(len(ci) for ci in client_indices) >= min_samples_per_client:
            return client_indices

    print("⚠️ Fallback IID acionado")
    all_indices = np.arange(len(targets))
    rng.shuffle(all_indices)

    return [
        all_indices[i::num_clients].tolist()
        for i in range(num_clients)
    ]


# =========================
# SPLIT LOCAL TRAIN / TEST
# =========================

def split_client_indices(indices, test_fraction=0.2, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.array(indices)
    rng.shuffle(indices)

    n_test = int(len(indices) * test_fraction)
    test_idx = indices[:n_test].tolist()
    train_idx = indices[n_test:].tolist()

    return train_idx, test_idx


# =========================
# MODEL
# =========================

class NextCategoryLSTMWithTime(nn.Module):
    def __init__(
        self,
        num_categories,
        cat_emb_dim=3,
        hour_emb_dim=3,
        day_emb_dim=2,
        delta_emb_dim=2,
        hidden_dim=8,
        num_layers=1,
        dropout=0.5
    ):
        super().__init__()

        self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)
        self.hour_emb = nn.Embedding(24, hour_emb_dim)
        self.day_emb = nn.Embedding(7, day_emb_dim)
        self.delta_emb = nn.Embedding(6, delta_emb_dim)

        input_dim = (
            cat_emb_dim +
            hour_emb_dim +
            day_emb_dim +
            delta_emb_dim
        )

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, num_categories)

    def forward(self, sequence):
        cat_e = self.cat_emb(sequence[:, :, 0])
        hour_e = self.hour_emb(sequence[:, :, 1])
        day_e = self.day_emb(sequence[:, :, 2])
        delta_e = self.delta_emb(sequence[:, :, 3])

        x = torch.cat([cat_e, hour_e, day_e, delta_e], dim=-1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# =========================
# TRAIN / EVAL
# =========================

def train_local(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for seq, y in loader:
        seq, y = seq.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(seq), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_global_from_clients(
    model, client_partitions, base_dataset, criterion, device
):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for cid, parts in client_partitions.items():
        test_idx = parts["test"]
        if len(test_idx) == 0:
            continue

        loader = DataLoader(
            SubsetDataset(base_dataset, test_idx),
            batch_size=256,
            shuffle=False
        )

        for seq, y in loader:
            seq, y = seq.to(device), y.to(device)
            logits = model(seq)

            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def fedavg_weighted(global_model, client_models, client_sizes):
    total_samples = sum(client_sizes)
    global_state = global_model.state_dict()

    for k in global_state.keys():
        global_state[k] = sum(
            client_models[i].state_dict()[k] * client_sizes[i]
            for i in range(len(client_models))
        ) / total_samples

    global_model.load_state_dict(global_state)


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    HF_REPO = "claudiogsc/foursquare-us-sequences-highlevel-200000-samples"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_CLIENTS = 20
    CLIENT_FRACTION = 0.3
    NUM_SELECTED = max(1, int(NUM_CLIENTS * CLIENT_FRACTION))

    ROUNDS = 200
    SHIFT_ROUND = ROUNDS // 2

    ALPHA_INITIAL = 0.1
    ALPHA_SHIFTED = 10.0

    ALPHA_INITIAL = 0.1
    ALPHA_SHIFTED = 1.0

    # ALPHA_INITIAL = 1.0
    # ALPHA_SHIFTED = 0.1

    # ALPHA_INITIAL = 10.0
    # ALPHA_SHIFTED = 0.1

    BATCH_SIZE = 128
    LR = 1e-3
    MIN_SAMPLES_PER_CLIENT = 128

    dataset = load_dataset(HF_REPO)
    train_base = NextCategoryDataset(dataset["train"])

    num_categories = int(max(dataset["train"]["label"]) + 1)

    print(f"🌍 FL iniciado | Clientes: {NUM_CLIENTS}")
    print(f"📊 Classes: {num_categories}")

    global_model = NextCategoryLSTMWithTime(num_categories).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # -------- Particionamentos --------

    client_indices_initial = dirichlet_label_partition_min_samples(
        dataset["train"]["label"],
        NUM_CLIENTS,
        num_categories,
        ALPHA_INITIAL,
        MIN_SAMPLES_PER_CLIENT,
        seed=1000,
    )

    client_indices_shifted = dirichlet_label_partition_min_samples(
        dataset["train"]["label"],
        NUM_CLIENTS,
        num_categories,
        ALPHA_SHIFTED,
        MIN_SAMPLES_PER_CLIENT,
        seed=2000,
    )

    def build_client_partitions(client_indices):
        parts = {}
        for cid in range(NUM_CLIENTS):
            tr, te = split_client_indices(
                client_indices[cid], test_fraction=0.2, seed=cid
            )
            parts[cid] = {"train": tr, "test": te}
        return parts

    client_parts_initial = build_client_partitions(client_indices_initial)
    client_parts_shifted = build_client_partitions(client_indices_shifted)

    # -------- Training --------

    for rnd in range(1, ROUNDS + 1):

        client_parts = (
            client_parts_initial
            if rnd < SHIFT_ROUND
            else client_parts_shifted
        )

        print(f"\n🔄 Round {rnd:03d}")

        rng = np.random.default_rng(3000 + rnd)
        selected_clients = rng.choice(
            NUM_CLIENTS, NUM_SELECTED, replace=False
        )

        client_models, client_sizes = [], []

        for cid in selected_clients:
            idx = client_parts[cid]["train"]
            if len(idx) == 0:
                continue

            model = copy.deepcopy(global_model).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            loader = DataLoader(
                SubsetDataset(train_base, idx),
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True
            )

            train_local(model, loader, optimizer, criterion, DEVICE)

            client_models.append(model)
            client_sizes.append(len(idx))

        fedavg_weighted(global_model, client_models, client_sizes)

        loss, acc = evaluate_global_from_clients(
            global_model,
            client_parts,
            train_base,
            criterion,
            DEVICE
        )

        if rnd < SHIFT_ROUND:
            alpha = ALPHA_INITIAL
            client_parts = client_parts_initial
        else:
            alpha = ALPHA_SHIFTED
            client_parts = client_parts_shifted

        print(f"\n🔄 Round {rnd:03d} | alpha = {alpha}")
        print(f"📉 Loss: {loss:.4f} | 🎯 Acc: {acc:.4f}")
