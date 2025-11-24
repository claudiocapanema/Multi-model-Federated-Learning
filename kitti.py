"""
Federated Multi-modal Driver Activity Recognition
(Modified so that after each round the updated global model is tested on each client's local test data.
 Each client returns loss, accuracy and number of local test samples; server computes weighted average.)
"""

import os
import random
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Try to import mediapipe; if not available, we'll fallback
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# =========================
# User-configurable section
# =========================
DATA_ROOT = "/home/gustavo/Downloads/state-farm-distracted-driver-detection/imgs"  # root folder containing train/ subfolder with class subfolders
NUM_CLIENTS = 10
FRACTION_CLIENTS = 0.3
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LR = 1e-4
MOMENTUM = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIRICHLET_ALPHA = 0.5 # smaller -> more heterogeneous
NUM_WORKERS = 4
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_CLASSES_MIN = 4  # ensure dataset has >3 classes
TEST_LOCAL_RATIO = 0.15  # fraction of each client's local samples used as local test

# =========================
# Utilities
# =========================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_SEED)


# Dataset loader that returns (image, keypoints, label)
class MultimodalImageKeypointsDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], classes: List[str], transform=None):
        """samples: list of (image_path, label_index)"""
        self.samples = samples
        self.classes = classes
        self.transform = transform

        # initialize mediapipe pose if available
        if MP_AVAILABLE:
            self.mp_pose = mp.solutions.pose.Pose(static_image_mode=True,
                                                  model_complexity=1,
                                                  enable_segmentation=False,
                                                  min_detection_confidence=0.5)
        else:
            self.mp_pose = None

    def __len__(self):
        return len(self.samples)

    def _extract_keypoints(self, pil_image: Image.Image) -> np.ndarray:
        # returns a fixed-length vector of keypoints (x,y,vis) for N landmarks
        if self.mp_pose is None:
            # fallback: return zeros + small noise
            return np.zeros(33 * 3, dtype=np.float32)

        # Convert PIL to RGB numpy
        image_rgb = np.asarray(pil_image.convert("RGB"))
        results = self.mp_pose.process(image_rgb)
        if not results.pose_landmarks:
            return np.zeros(33 * 3, dtype=np.float32)

        lm = results.pose_landmarks.landmark
        vec = []
        for i in range(min(33, len(lm))):
            vec.extend([lm[i].x, lm[i].y, lm[i].visibility])
        # pad if less than 33
        while len(vec) < 33 * 3:
            vec.extend([0.0, 0.0, 0.0])
        return np.array(vec, dtype=np.float32)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = Image.open(path).convert("RGB")
        img = pil
        if self.transform:
            img = self.transform(pil)
        keypoints = self._extract_keypoints(pil)
        keypoints = torch.from_numpy(keypoints).float()
        return img, keypoints, label


# Build dataset list from folder structure
def build_dataset(root: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    train_dir = Path(root) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}. Please put your dataset there.\n" \
                                "Expected structure: DATA_ROOT/train/<class_name>/*.jpg")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if len(classes) < NUM_CLASSES_MIN:
        raise ValueError(f"Need at least {NUM_CLASSES_MIN} classes; found {len(classes)}")
    samples = []
    for i, c in enumerate(classes):
        for img_path in (train_dir / c).glob("**/*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((str(img_path), i))
    return samples, classes


# Dirichlet partitioning
def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float) -> List[np.ndarray]:
    """Return list of arrays with the sample indices for each client (indices refer to the labels array)"""
    n_classes = np.max(labels) + 1
    label_indices = [np.where(labels == i)[0] for i in range(n_classes)]
    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        # draw proportions for each client from Dirichlet
        proportions = np.random.dirichlet(alpha=[alpha] * n_clients)
        # shuffle class indices
        idx_c = label_indices[c].copy()
        np.random.shuffle(idx_c)
        # split according to proportions
        counts = (proportions * len(idx_c)).astype(int)
        # fix rounding
        while counts.sum() < len(idx_c):
            counts[np.argmax(proportions)] += 1
        pos = 0
        for client_id in range(n_clients):
            cnt = counts[client_id]
            if cnt > 0:
                client_indices[client_id].extend(idx_c[pos:pos+cnt].tolist())
                pos += cnt
    # convert to numpy arrays
    return [np.array(sorted(ci), dtype=int) for ci in client_indices]


# Simple multimodal model: image encoder (ResNet18 w/o fc) + keypoint MLP + fusion head
class MultimodalNet(nn.Module):
    def __init__(self, num_classes: int, kp_dim: int = 33*3, embed_dim: int = 512):
        super().__init__()
        self.image_model = models.resnet18(pretrained=False)
        # remove fc
        self.image_model.fc = nn.Identity()
        img_feat_dim = 512

        # keypoint encoder
        self.kp_encoder = nn.Sequential(
            nn.Linear(kp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # fusion and classifier
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + 128, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x_img, x_kp):
        img_feat = self.image_model(x_img)
        kp_feat = self.kp_encoder(x_kp)
        fused = torch.cat([img_feat, kp_feat], dim=1)
        out = self.fusion(fused)
        return out


# Federated utilities

def get_model_params(model: nn.Module):
    return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}


def set_model_params(model: nn.Module, params: Dict[str, torch.Tensor]):
    model.load_state_dict(params)


def average_params(param_list: List[Dict[str, torch.Tensor]], weights: List[float] = None) -> Dict[str, torch.Tensor]:
    if len(param_list) == 0:
        raise ValueError("param_list is empty")
    if weights is None:
        weights = [1.0 / len(param_list)] * len(param_list)
    avg = {}
    for k in param_list[0].keys():
        acc = None
        for i in range(len(param_list)):
            term = param_list[i][k].float() * weights[i]
            if acc is None:
                acc = term.clone()
            else:
                acc += term
        avg[k] = acc
    return avg


# Training and evaluation functions

def train_local(model: nn.Module, dataloader: DataLoader, epochs: int, device: torch.device):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running = 0.0
        pbar = tqdm(dataloader, desc=f"Local train ep {epoch+1}/{epochs}", leave=False)
        for imgs, kps, labels in pbar:
            imgs = imgs.to(device)
            kps = kps.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, kps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        # end epoch
    return model


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, kps, labels in dataloader:
            imgs = imgs.to(device)
            kps = kps.to(device)
            labels = labels.to(device)
            outputs = model(imgs, kps)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


# Main federated simulation

def main():
    print("Device:", DEVICE)
    samples, classes = build_dataset(DATA_ROOT)
    labels = np.array([s[1] for s in samples])

    # train/test split centrally (we still keep a central held-out test if desired)
    train_idx, central_test_idx = train_test_split(np.arange(len(samples)), test_size=0.2, stratify=labels, random_state=RANDOM_SEED)
    train_samples = [samples[i] for i in train_idx]
    train_labels = labels[train_idx]

    # Dirichlet partition on train set -> returns indices referencing train_samples array
    client_splits = dirichlet_partition(train_labels, NUM_CLIENTS, DIRICHLET_ALPHA)

    # transforms
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # build per-client train and test loaders
    client_train_loaders = []
    client_test_loaders = []
    client_train_sizes = []
    client_test_sizes = []

    for c in range(NUM_CLIENTS):
        idx_in_train = client_splits[c]  # indices into train_samples
        # if client has no samples, create empty loaders
        if len(idx_in_train) == 0:
            client_train_loaders.append(None)
            client_test_loaders.append(None)
            client_train_sizes.append(0)
            client_test_sizes.append(0)
            continue

        # build the actual sample lists for this client
        client_all_samples = [train_samples[i] for i in idx_in_train]
        client_all_labels = np.array([s[1] for s in client_all_samples])

        # decide split sizes: if too small, give all to train and zero to test
        if len(client_all_samples) < 2:
            # too small to split
            train_subset = client_all_samples
            test_subset = []
        else:
            # try stratified split; fallback to random if stratify fails
            try:
                local_train_idx, local_test_idx = train_test_split(
                    np.arange(len(client_all_samples)),
                    test_size=TEST_LOCAL_RATIO,
                    stratify=client_all_labels,
                    random_state=RANDOM_SEED
                )
            except Exception:
                local_train_idx, local_test_idx = train_test_split(
                    np.arange(len(client_all_samples)),
                    test_size=TEST_LOCAL_RATIO,
                    random_state=RANDOM_SEED
                )
            train_subset = [client_all_samples[i] for i in local_train_idx]
            test_subset = [client_all_samples[i] for i in local_test_idx]

        # create datasets/loaders
        if len(train_subset) > 0:
            train_ds = MultimodalImageKeypointsDataset(train_subset, classes, transform=transform)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            client_train_loaders.append(train_loader)
            client_train_sizes.append(len(train_ds))
        else:
            client_train_loaders.append(None)
            client_train_sizes.append(0)

        if len(test_subset) > 0:
            test_ds = MultimodalImageKeypointsDataset(test_subset, classes, transform=test_transform)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            client_test_loaders.append(test_loader)
            client_test_sizes.append(len(test_ds))
        else:
            client_test_loaders.append(None)
            client_test_sizes.append(0)

    # optional central test loader (from held out central_test_idx)
    central_test_samples = [samples[i] for i in central_test_idx]
    central_test_ds = MultimodalImageKeypointsDataset(central_test_samples, classes, transform=test_transform)
    central_test_loader = DataLoader(central_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # initialize global model
    global_model = MultimodalNet(num_classes=len(classes))
    global_model.to(DEVICE)

    # server loop
    results = []
    for rnd in range(1, ROUNDS + 1):
        print(f"\n--- Round {rnd}/{ROUNDS} ---")
        # sample clients (here: FRACTION_CLIENTS each round)
        m = max(1, int(NUM_CLIENTS * FRACTION_CLIENTS))
        local_params = []
        local_sizes = []
        # randomly select m clients
        selected_clients = random.sample(range(NUM_CLIENTS), m)

        print("Selected clients:", selected_clients)
        for client_id in selected_clients:
            # clone global weights
            local_model = MultimodalNet(num_classes=len(classes))
            set_model_params(local_model, get_model_params(global_model))

            train_loader = client_train_loaders[client_id]
            if train_loader is None or len(train_loader.dataset) == 0:
                print(f"Client {client_id} has 0 local train samples, skipping local training.")
                # still append their params (same as global) with size 0 so weight will be zero
                local_params.append(get_model_params(local_model))
                local_sizes.append(0)
                continue

            # train locally
            local_model = train_local(local_model, train_loader, LOCAL_EPOCHS, DEVICE)
            local_params.append(get_model_params(local_model))
            local_sizes.append(len(train_loader.dataset))

        # aggregate (FedAvg) - handle case where sum local_sizes == 0
        total_size = sum(local_sizes)
        if total_size == 0:
            print("No clients performed training this round; global model unchanged.")
        else:
            weights = [s / total_size for s in local_sizes]
            avg_params = average_params(local_params, weights)
            set_model_params(global_model, avg_params)

        # --- NEW: evaluate global model on each client's local test set ---
        per_client_results = []
        sum_test_samples = 0
        weighted_loss_accum = 0.0
        weighted_acc_accum = 0.0

        for cid in range(NUM_CLIENTS):
            test_loader = client_test_loaders[cid]
            n_test = client_test_sizes[cid]
            if test_loader is None or n_test == 0:
                # client has no test data -> skip (contributes 0 to weighted avg)
                per_client_results.append((cid, 0.0, 0.0, 0))
                continue
            # evaluate global model on client's local test data
            # ensure model params are the current global ones
            eval_model = MultimodalNet(num_classes=len(classes))
            set_model_params(eval_model, get_model_params(global_model))
            loss_c, acc_c = evaluate(eval_model, test_loader, DEVICE)
            per_client_results.append((cid, loss_c, acc_c, n_test))

            sum_test_samples += n_test
            weighted_loss_accum += loss_c * n_test
            weighted_acc_accum += acc_c * n_test

        # compute weighted averages
        if sum_test_samples > 0:
            round_loss = weighted_loss_accum / sum_test_samples
            round_acc = weighted_acc_accum / sum_test_samples
        else:
            round_loss = 0.0
            round_acc = 0.0

        # optionally evaluate on central test set as well
        central_loss, central_acc = evaluate(global_model, central_test_loader, DEVICE)

        print(f"Round {rnd} weighted (by client local test size) -> Loss: {round_loss:.4f}, Acc: {round_acc*100:.2f}%")
        print(f"Central held-out test -> Loss: {central_loss:.4f}, Acc: {central_acc*100:.2f}%")

        # print per-client brief summary
        for cid, l_c, a_c, n_c in per_client_results:
            if n_c == 0:
                print(f" Client {cid}: no local test samples")
            else:
                print(f" Client {cid}: n_test={n_c}, loss={l_c:.4f}, acc={a_c*100:.2f}%")

        results.append((rnd, round_loss, round_acc, central_loss, central_acc))

    print("\nTraining finished. Summary per round (weighted client-local test):")
    for r, l, a, cl, ca in results:
        print(f"Round {r}: Loss {l:.4f}, Acc {a*100:.2f}%; Central Loss {cl:.4f}, Central Acc {ca*100:.2f}%")

if __name__ == '__main__':
    main()
