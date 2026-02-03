#!/usr/bin/env python3
"""
federated_multimodal_autokp_statefarm.py

- Para dataset no formato:
    DATA_ROOT/
        train/
            c0/
            c1/
            ...
            c9/

- Detecta classes automaticamente (ordenadas) — normalmente c0..c9.
- Gera keypoints com MediaPipe Pose (33 landmarks x 3) e salva por imagem em KEYPOINTS_ROOT
  (mesma árvore de diretórios, .npy por imagem). Na próxima execução, carrega os .npy já salvos.
- Treina federado multimodal (RGB + keypoints flatten) com FedAvg e Dirichlet partition.
- Mostra loss e acurácia por cliente e média ponderada por rodada, além do teste central.
"""

import os
import sys
import random
from pathlib import Path
from glob import glob
from typing import List, Tuple, Dict

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split

# ---------------------------
# CONFIGURAÇÃO (edite aqui)
# ---------------------------
DATA_ROOT = "/home/gustavo/Downloads/state-farm-distracted-driver-detection/imgs"  # raiz contendo "train/"
KEYPOINTS_ROOT = "/home/gustavo/Downloads/state-farm-keypoints"  # onde salvar .npy de keypoints (mesma árvore)
NUM_CLIENTS = 10
FRACTION_CLIENTS = 0.3  # 30% clientes por rodada
ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIRICHLET_ALPHA = 0.5
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_WORKERS = 4
TEST_LOCAL_RATIO = 0.2  # divisão local treino/teste dentro de cada cliente
# ---------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------------------------
# UTIL: paths e classes
# ---------------------------
TRAIN_DIR = Path(DATA_ROOT) / "train"
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Diretório de treino não encontrado: {TRAIN_DIR}")

# detecta pastas de classe (ordenadas)
CLASS_FOLDERS = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])
if len(CLASS_FOLDERS) == 0:
    raise RuntimeError(f"Nenhuma pasta de classe encontrada em {TRAIN_DIR}")
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_FOLDERS)}
NUM_CLASSES = len(CLASS_FOLDERS)

print(f"[Info] Encontradas {NUM_CLASSES} classes: {CLASS_FOLDERS}")

# ---------------------------
# KEYPOINTS: geração + cache por arquivo
# ---------------------------
mp_pose = mp.solutions.pose

def kp_target_path_for_image(img_path: str) -> Path:
    """
    Mapear imagem -> caminho .npy em KEYPOINTS_ROOT mantendo subpastas.
    Ex: DATA_ROOT/train/c0/xxx.jpg -> KEYPOINTS_ROOT/train/c0/xxx.npy
    """
    p = Path(img_path)
    # relative path from DATA_ROOT
    rel = p.relative_to(Path(DATA_ROOT))
    kp_path = Path(KEYPOINTS_ROOT) / rel
    kp_path = kp_path.with_suffix(".npy")
    return kp_path

def detect_keypoints_mediapipe(img_bgr) -> np.ndarray:
    """
    Recebe imagem BGR (OpenCV) e retorna np.array shape (33,3) float32
    Se não detectar, retorna zeros.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return np.zeros((33, 3), dtype=np.float32)
        kps = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)
        return kps

def ensure_keypoints_for_all_images(force_regen=False):
    """
    Percorre todas as imagens em DATA_ROOT/train/** e garante .npy em KEYPOINTS_ROOT.
    Se já existir e force_regen==False, pula.
    """
    print("[Info] Verificando keypoints (MediaPipe). Isso pode demorar na primeira execução...")
    all_images = sorted(glob(str(TRAIN_DIR / "**" / "*.*"), recursive=True))
    imgs = [p for p in all_images if Path(p).suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if len(imgs) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada em {TRAIN_DIR}")

    os.makedirs(KEYPOINTS_ROOT, exist_ok=True)
    # Usar media pipe reusando o contexto para performance
    mp_sol = mp.solutions.pose
    pose = mp_sol.Pose(static_image_mode=True)
    try:
        for img_path in tqdm(imgs, desc="Gerando keypoints", unit="img"):
            kp_path = kp_target_path_for_image(img_path)
            if kp_path.exists() and (not force_regen):
                continue
            # criamos pastas necessárias
            kp_path.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(img_path)
            if img is None:
                # salvar zeros se falha leitura
                np.save(str(kp_path), np.zeros((33,3), dtype=np.float32))
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            if not res.pose_landmarks:
                kp = np.zeros((33,3), dtype=np.float32)
            else:
                kp = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)
            np.save(str(kp_path), kp)
    finally:
        pose.close()
    print("[Info] Keypoints garantidos em:", KEYPOINTS_ROOT)

# ---------------------------
# DATASET multimodal (carrega keypoints .npy)
# ---------------------------
class MultimodalDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], transform=None, kp_root=KEYPOINTS_ROOT):
        """
        samples: list of tuples (img_path, label_idx)
        kp_root: root where .npy saved (same rel structure)
        """
        self.samples = samples
        self.transform = transform
        self.kp_root = Path(kp_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl = self.samples[idx]
        # leitura imagem
        img = cv2.imread(img_path)
        if img is None:
            # fallback: black image
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            # torchvision transforms work on PIL or tensor; convert to PIL
            from PIL import Image
            img_t = Image.fromarray(img_rgb)
            img_t = self.transform(img_t)
        else:
            # fallback: convert to tensor manually
            img_t = T.ToTensor()(img_rgb)

        # keypoints load
        kp_path = kp_target_path_for_image(img_path)
        if kp_path.exists():
            kp = np.load(str(kp_path)).astype(np.float32).ravel()
        else:
            # fallback zeros (shouldn't happen if ensured)
            kp = np.zeros((33*3,), dtype=np.float32)

        kp_t = torch.from_numpy(kp)

        return img_t, kp_t, int(lbl)

# ---------------------------
# MODELO multimodal
# ---------------------------
class MultimodalNet(nn.Module):
    def __init__(self, kp_dim=33*3, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        # backbone imagem (resnet18 sem fc)
        try:
            # preferível: não forçar weights None; usar pretrained se disponível
            self.cnn = models.resnet18(weights=None)
        except Exception:
            self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()  # output 512-d

        # MLP keypoints
        self.kp_mlp = nn.Sequential(
            nn.Linear(kp_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        fusion_dim = 512 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_kp):
        img_feat = self.cnn(x_img)          # (B,512)
        kp_feat = self.kp_mlp(x_kp)         # (B,64)
        fused = torch.cat([img_feat, kp_feat], dim=1)
        out = self.classifier(fused)
        return out

# ---------------------------
# FUNÇÕES FEDERADAS
# ---------------------------
def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float):
    """
    labels: numpy array of integer labels (for the training set)
    retorna lista de arrays de índices (referentes ao array labels)
    """
    labels = np.array(labels)
    n_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]

    client_indices = [[] for _ in range(n_clients)]

    for c, idxs in enumerate(idx_by_class):
        if len(idxs) == 0:
            continue
        # proporções Dirichlet
        proportions = np.random.dirichlet([alpha] * n_clients)
        # quantos pegar por cliente nesta classe
        counts = (proportions * len(idxs)).astype(int)
        # corrigir arredondamento
        while counts.sum() < len(idxs):
            counts[np.argmax(proportions)] += 1
        # embaralha índices da classe
        np.random.shuffle(idxs)
        ptr = 0
        for i in range(n_clients):
            take = counts[i]
            if take > 0:
                client_indices[i].extend(idxs[ptr:ptr+take].tolist())
                ptr += take
    # converter para arrays ordenados
    return [np.array(sorted(lst), dtype=int) for lst in client_indices]

def get_state_dict_cpu(model: nn.Module):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_state_dict_to_model(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    model.load_state_dict(state_dict)

def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float]):
    avg = {}
    keys = state_dicts[0].keys()
    for k in keys:
        acc = None
        for s, w in zip(state_dicts, weights):
            term = s[k].float() * w
            if acc is None:
                acc = term.clone()
            else:
                acc += term
        avg[k] = acc
    return avg

def train_local(model: nn.Module, loader: DataLoader, epochs: int, device):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"Local ep {ep+1}/{epochs}", leave=False)
        for imgs, kps, labels in pbar:
            imgs = imgs.to(device)
            kps = kps.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs, kps)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model: nn.Module, loader: DataLoader, device):
    if loader is None:
        return 0.0, 0.0
    model.to(device)
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for imgs, kps, labels in loader:
            imgs = imgs.to(device)
            kps = kps.to(device)
            labels = labels.to(device)
            out = model(imgs, kps)
            total_loss += criterion(out, labels).item()
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total

# ---------------------------
# MAIN
# ---------------------------
def main():
    print(f"[Info] DEVICE: {DEVICE}")
    # 1) garantir keypoints (gera se necessário)
    ensure_keypoints_for_all_images()

    # 2) montar lista de amostras (apenas a árvore "train")
    all_img_paths = sorted([str(p) for p in TRAIN_DIR.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if len(all_img_paths) == 0:
        raise RuntimeError("Nenhuma imagem encontrada.")

    # label por caminho usando folder name diretamente (train/<folder>/<img>)
    samples = []
    for p in all_img_paths:
        folder = Path(p).parent.name
        if folder not in CLASS_TO_IDX:
            # se folder inesperado, pular
            print(f"[Aviso] pasta não mapeada (pulando): {folder} <- {p}")
            continue
        samples.append((p, CLASS_TO_IDX[folder]))

    # shuffle e split central holdout (sobre todas as amostras)
    indices = list(range(len(samples)))
    labels = np.array([s[1] for s in samples])
    train_idx, holdout_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=RANDOM_SEED)
    train_samples = [samples[i] for i in train_idx]
    holdout_samples = [samples[i] for i in holdout_idx]
    train_labels = np.array([s[1] for s in train_samples])

    # 3) Dirichlet partition sobre train_samples
    client_partitions = dirichlet_partition(train_labels, NUM_CLIENTS, DIRICHLET_ALPHA)

    # transforms
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Build loaders per client (robusto: split local stratify)
    client_train_loaders = []
    client_test_loaders = []
    client_train_sizes = []
    client_test_sizes = []

    for cid in range(NUM_CLIENTS):
        idxs = client_partitions[cid]
        if len(idxs) == 0:
            client_train_loaders.append(None)
            client_test_loaders.append(None)
            client_train_sizes.append(0)
            client_test_sizes.append(0)
            continue

        client_all = [train_samples[i] for i in idxs]
        labels_local = np.array([s[1] for s in client_all])

        try:
            tr_i, te_i = train_test_split(
                list(range(len(client_all))),
                test_size=TEST_LOCAL_RATIO,
                stratify=labels_local,
                random_state=RANDOM_SEED
            )
        except Exception:
            tr_i, te_i = train_test_split(
                list(range(len(client_all))),
                test_size=TEST_LOCAL_RATIO,
                random_state=RANDOM_SEED
            )

        train_subset = [client_all[i] for i in tr_i]
        test_subset = [client_all[i] for i in te_i]

        if len(train_subset) > 0:
            ds_tr = MultimodalDataset(train_subset, transform=transform)
            loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=(DEVICE.type=="cuda"))
            client_train_loaders.append(loader_tr)
            client_train_sizes.append(len(ds_tr))
        else:
            client_train_loaders.append(None)
            client_train_sizes.append(0)

        if len(test_subset) > 0:
            ds_te = MultimodalDataset(test_subset, transform=test_transform)
            loader_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS)
            client_test_loaders.append(loader_te)
            client_test_sizes.append(len(ds_te))
        else:
            client_test_loaders.append(None)
            client_test_sizes.append(0)

    # central holdout loader
    central_test_ds = MultimodalDataset(holdout_samples, transform=test_transform)
    central_test_loader = DataLoader(central_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 4) criar modelo global
    example_kp = np.load(str(kp_target_path_for_image(train_samples[0][0]))) if len(train_samples)>0 else np.zeros((33,3))
    kp_dim = int(example_kp.ravel().shape[0])
    print(f"[Info] Keypoint dim: {kp_dim}")
    global_model = MultimodalNet(kp_dim=kp_dim, num_classes=NUM_CLASSES) if 'kp_dim' in MultimodalNet.__init__.__code__.co_varnames else MultimodalNet()
    # NOTE: MultimodalNet defined above expects default kp_dim 99; we'll construct properly:
    global_model = MultimodalNet(kp_dim=kp_dim, num_classes=NUM_CLASSES)
    # inicializa com weights padrão
    global_state = get_state_dict_cpu(global_model)

    # 5) federated training
    for rnd in range(1, ROUNDS+1):
        print(f"\n--- Round {rnd}/{ROUNDS} ---")
        m = max(1, int(NUM_CLIENTS * FRACTION_CLIENTS))
        selected = random.sample(range(NUM_CLIENTS), m)
        print("Selected clients:", selected)

        local_states = []
        local_sizes = []

        for cid in selected:
            loader = client_train_loaders[cid]
            if loader is None or client_train_sizes[cid] == 0:
                print(f" - Client {cid}: sem dados para treinar, pulando.")
                continue

            print(f" - Client {cid}: treinamento local (N={client_train_sizes[cid]})")
            local_model = MultimodalNet(kp_dim=kp_dim, num_classes=NUM_CLASSES)
            local_model.load_state_dict(global_state)
            local_model = train_local(local_model, loader, LOCAL_EPOCHS, DEVICE)

            local_states.append(get_state_dict_cpu(local_model))
            local_sizes.append(client_train_sizes[cid])

        if len(local_states) > 0:
            total = sum(local_sizes)
            weights = [s/total for s in local_sizes]
            new_state = average_state_dicts(local_states, weights)
            # atualizar global_state e global_model
            global_state = new_state
            set_state_dict_to_model(global_model, global_state)
            print(f"[Info] Global model atualizado (agregados {len(local_states)} clientes, total amostras {total})")
        else:
            print("[Aviso] Nenhum cliente treinou nesta rodada; modelo global não alterado.")

        # avaliação por cliente (local tests)
        total_test_samples = sum(client_test_sizes)
        weighted_loss = 0.0
        weighted_acc = 0.0

        for cid in range(NUM_CLIENTS):
            loader_te = client_test_loaders[cid]
            n = client_test_sizes[cid]
            if loader_te is None or n == 0:
                continue
            eval_model = MultimodalNet(kp_dim=kp_dim, num_classes=NUM_CLASSES)
            eval_model.load_state_dict(global_state)
            loss_c, acc_c = evaluate(eval_model, loader_te, DEVICE)
            print(f" Client {cid} test local N={n}: Loss={loss_c:.4f} Acc={acc_c*100:.2f}%")
            weighted_loss += loss_c * n
            weighted_acc += acc_c * n

        if total_test_samples > 0:
            round_loss = weighted_loss / total_test_samples
            round_acc = weighted_acc / total_test_samples
        else:
            round_loss = round_acc = 0.0

        print(f" Weighted local test (média ponderada): Loss={round_loss:.4f} Acc={round_acc*100:.2f}%")

        # avaliação central holdout
        c_loss, c_acc = evaluate(global_model, central_test_loader, DEVICE)
        print(f" Central holdout: Loss={c_loss:.4f} Acc={c_acc*100:.2f}%")

    print("\nTreinamento federado finalizado.")

if __name__ == "__main__":
    main()
