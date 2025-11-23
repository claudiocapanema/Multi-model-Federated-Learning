"""
Federated Multi-modal Driver Activity Recognition
Single-file Python script that simulates federated learning (FedAvg) for driver activity
recognition using a multimodal approach: RGB images + pose keypoints.

Design choices (easy-to-use dataset):
- This script is written to work with the popular "State Farm Distracted Driver" dataset
  (10 classes) or any dataset organized as: root/train/<class_name>/*.jpg
  (you can also use any custom dataset with >3 classes following that structure).

Multimodality:
- Modality A: RGB image fed to a CNN (ResNet18 backbone)
- Modality B: 33 pose landmarks (x,y,visibility) obtained with MediaPipe Holistic/BlazePose
  (if mediapipe isn't available or pose fails, the code falls back to a small placeholder vector)
- Fusion: late fusion by concatenating image feature vector and keypoint embedding.

Federated setup:
- Partition using Dirichlet distribution (alpha parameter configurable) to create non-IID splits
- Simulates NUM_CLIENTS clients on a single machine. Each client trains locally for LOCAL_EPOCHS
  then the server averages weights (FedAvg).
- Evaluates global model on a held-out test split each round.

Requirements:
- Python 3.8+
- pip install torch torchvision tqdm scikit-learn pillow mediapipe
  (mediapipe optional but recommended: pip install mediapipe)

Usage:
- Put dataset in DATA_ROOT with structure:
    DATA_ROOT/train/<class_name>/*.jpg
  or change variables below.
- Run: python federated_multimodal_driver_activity.py

Notes:
- The script is intended as an automated starting point. It focuses on clarity and
  reproducibility over extreme optimization. Adjust hyperparameters as needed.

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
DIRICHLET_ALPHA = 0.1 # smaller -> more heterogeneous
NUM_WORKERS = 4
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_CLASSES_MIN = 4  # ensure dataset has >3 classes

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
    """Return list of arrays with the sample indices for each client"""
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
    if weights is None:
        weights = [1.0 / len(param_list)] * len(param_list)
    avg = {}
    for k in param_list[0].keys():
        avg[k] = sum(param_list[i][k].float() * weights[i] for i in range(len(param_list)))
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
    return total_loss / total, correct / total


# Main federated simulation

def main():
    print("Device:", DEVICE)
    samples, classes = build_dataset(DATA_ROOT)
    labels = np.array([s[1] for s in samples])

    # train/test split centrally
    train_idx, test_idx = train_test_split(np.arange(len(samples)), test_size=0.15, stratify=labels, random_state=RANDOM_SEED)
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]
    train_labels = labels[train_idx]

    # Dirichlet partition on train set
    client_splits = dirichlet_partition(train_labels, NUM_CLIENTS, DIRICHLET_ALPHA)

    # transforms
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # build datasets and dataloaders per client
    client_loaders = []
    for c in range(NUM_CLIENTS):
        idx = client_splits[c]
        client_samples = [train_samples[i] for i in idx]
        ds = MultimodalImageKeypointsDataset(client_samples, classes, transform=transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        client_loaders.append(loader)

    # test loader
    test_ds = MultimodalImageKeypointsDataset(test_samples, classes, transform=T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # initialize global model
    global_model = MultimodalNet(num_classes=len(classes))
    global_model.to(DEVICE)

    # server loop
    results = []
    for rnd in range(1, ROUNDS + 1):
        print(f"\n--- Round {rnd}/{ROUNDS} ---")
        # sample clients (here: all clients participate)
        # sample 30% of clients each round
        m = max(1, int(NUM_CLIENTS * FRACTION_CLIENTS))
        local_params = []
        local_sizes = []
        # randomly select 30% of clients
        selected_clients = random.sample(range(NUM_CLIENTS), m)
        for client_id in selected_clients:
            # clone global weights
            local_model = MultimodalNet(num_classes=len(classes))
            set_model_params(local_model, get_model_params(global_model))
            # train locally
            if len(client_loaders[client_id].dataset) == 0:
                print(f"Client {client_id} has 0 samples, skipping")
                local_params.append(get_model_params(local_model))
                local_sizes.append(0)
                continue
            local_model = train_local(local_model, client_loaders[client_id], LOCAL_EPOCHS, DEVICE)
            local_params.append(get_model_params(local_model))
            local_sizes.append(len(client_loaders[client_id].dataset))
        # aggregate
        total_size = sum(local_sizes) if sum(local_sizes) > 0 else 1
        weights = [s / total_size for s in local_sizes]
        avg_params = average_params(local_params, weights)
        set_model_params(global_model, avg_params)

        # evaluate global
        loss, acc = evaluate(global_model, test_loader, DEVICE)
        print(f"Global eval -> Loss: {loss:.4f}, Acc: {acc*100:.2f}%")
        results.append((rnd, loss, acc))

    print("\nTraining finished. Summary per round:")
    for r, l, a in results:
        print(f"Round {r}: Loss {l:.4f}, Acc {a*100:.2f}%")


if __name__ == '__main__':
    main()
