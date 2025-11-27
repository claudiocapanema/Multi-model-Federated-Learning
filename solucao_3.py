# backbone atualizado depois do round robin + CSV logging
import os
import cv2
import glob
import pickle
import random
import copy
import csv
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset  # Hugging Face GTSRB

###########################################
# CONFIGURAÇÕES
###########################################
DATA_DIR = "/home/gustavo/Downloads/state-farm-distracted-driver-detection/imgs/train"
POSE_CACHE = "pose_cache.pkl"

NUM_CLIENTS = 40
CLIENTS_PER_TASK = 4          # 4 clientes por tarefa em cada rodada
ROUNDS = 100                  # total de rodadas

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hiperparâmetros de convergência
THRESHOLD_IMPROVEMENT = 0.02      # 2%
PLATEAU_T = 5                     # t_hat: nº de rodadas sem melhora
MIN_ROUND_FOR_CONV_CHECK = 20     # só verifica convergência após essa rodada

# Alphas para Dirichlet (StateFarm / GTSRB)
ALPHA_STATEFARM = 0.5
ALPHA_GTSRB = 0.5

# Arquivos de saída CSV (diretório atual)
CSV_STATEFARM_FINE = "results_statefarm_fine.csv"
CSV_STATEFARM_COARSE = "results_statefarm_coarse.csv"
CSV_GTSRB = "results_gtsrb.csv"

# Grupo 1 (clientes 0-19): tarefa fine (10 classes originais)
# Grupo 2 (clientes 20-39): tarefa coarse (3 classes agregadas)


###########################################
# CARREGAR ARQUIVOS STATE FARM
###########################################
def load_image_paths():
    classes = sorted(os.listdir(DATA_DIR))
    img_paths = []
    labels = []

    for idx, cls in enumerate(classes):
        paths = glob.glob(os.path.join(DATA_DIR, cls, "*.jpg"))
        for p in paths:
            img_paths.append(p)
            labels.append(idx)

    return img_paths, labels


###########################################
# EXTRAÇÃO DE POSE (MediaPipe)
###########################################
mp_pose = mp.solutions.pose

def extract_pose_single(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(img_rgb)
        if not result.pose_landmarks:
            return np.zeros((33, 3), dtype=np.float32)

        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)


def extract_pose_parallel(image_paths):
    print("\n⏳ Extraindo poses em paralelo...")
    with Pool(cpu_count()) as pool:
        poses = list(tqdm(pool.imap(extract_pose_single, image_paths), total=len(image_paths)))
    return poses


###########################################
# CACHE DE POSES
###########################################
def get_pose_features(img_paths):
    if os.path.exists(POSE_CACHE):
        print("✔️ Carregando poses do cache...")
        with open(POSE_CACHE, "rb") as f:
            poses = pickle.load(f)
    else:
        poses = extract_pose_parallel(img_paths)
        print("💾 Salvando poses no cache...")
        with open(POSE_CACHE, "wb") as f:
            pickle.dump(poses, f)

    for i in range(len(poses)):
        if poses[i] is None:
            poses[i] = np.zeros((33, 3), dtype=np.float32)

    return poses


###########################################
# DATASET MULTIMODAL (STATE FARM)
###########################################
class MultiModalDataset(Dataset):
    def __init__(self, img_paths, poses, labels):
        self.img_paths = img_paths
        self.poses = poses
        self.labels = labels

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),   # resolução reduzida
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tf(img)

        pose = torch.tensor(self.poses[idx].flatten(), dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, pose, label


###########################################
# DATASET GTSRB (Hugging Face claudiogsc/GTSRB)
###########################################
class GTSRBDataset(Dataset):
    def __init__(self, hf_dataset, indices):
        self.ds = hf_dataset
        self.indices = list(indices)

        self.tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.ds[int(real_idx)]
        img = item["image"]           # PIL Image
        label = item["label"]         # int

        img = self.tf(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


###########################################
# MODELO MULTIMODAL + MULTI-TASK (STATE FARM)
###########################################
class MultiTaskMultiModalNet(nn.Module):
    """
    Multi-modal: RGB + Pose
    Multi-task: fine (10 classes) e coarse (3 classes)
    - cnn_shared + pose_mlp_shared: parâmetros compartilhados inicialmente
    - cnn_fine / cnn_coarse, pose_fine / pose_coarse: backbones individuais após convergência
    - residual_fine / residual_coarse: específicos de cada tarefa
    - head_fine / head_coarse: específicos de cada tarefa

    Regras:
    - Antes da convergência, tasks usam apenas backbone compartilhado.
    - Quando uma task correlacionada converge, ela passa a usar backbone individual.
    - Quando >= 2 tasks correlacionadas convergem e completam um ciclo do round-robin,
      o backbone compartilhado é atualizado a partir dos individuais e, em seguida,
      propaga de volta para os individuais.
    """
    def __init__(self, num_classes_fine=10, num_classes_coarse=3):
        super().__init__()

        # backbone visual compartilhado
        self.cnn_shared = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # backbone de pose compartilhado
        self.pose_mlp_shared = nn.Sequential(
            nn.Linear(99, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )

        fused_dim = 64 + 32  # 64 (cnn) + 32 (pose)

        # resíduos por tarefa (personalização leve)
        self.residual_fine = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )
        self.residual_coarse = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

        # cabeças por tarefa
        self.head_fine = nn.Linear(fused_dim, num_classes_fine)
        self.head_coarse = nn.Linear(fused_dim, num_classes_coarse)

        # Backbones específicos por tarefa (serão criados on-demand)
        self.cnn_fine = None
        self.cnn_coarse = None
        self.pose_fine = None
        self.pose_coarse = None

        # Flags de uso de backbone individual
        self.use_private_fine = False
        self.use_private_coarse = False

    def ensure_private_backbone(self, task_type, device=DEVICE):
        """
        Cria backbone individual para a tarefa se ainda não existir.
        No momento da criação, ele é idêntico ao backbone compartilhado
        (mantendo a ideia de que antes da convergência eles eram os mesmos).
        """
        if task_type == "fine" and not self.use_private_fine:
            self.cnn_fine = copy.deepcopy(self.cnn_shared).to(device)
            self.pose_fine = copy.deepcopy(self.pose_mlp_shared).to(device)
            self.use_private_fine = True
            print("✅ Criado backbone individual para tarefa fine.")
        elif task_type == "coarse" and not self.use_private_coarse:
            self.cnn_coarse = copy.deepcopy(self.cnn_shared).to(device)
            self.pose_coarse = copy.deepcopy(self.pose_mlp_shared).to(device)
            self.use_private_coarse = True
            print("✅ Criado backbone individual para tarefa coarse.")

    def sync_shared_and_private(self, device=DEVICE):
        """
        Atualiza o backbone compartilhado a partir da média dos backbones individuais
        (fine/coarse) e, em seguida, sobrescreve os individuais com o compartilhado.
        Usado ao final de um ciclo do round-robin entre tarefas correlacionadas convergidas.
        """
        if not (self.use_private_fine and self.use_private_coarse):
            return

        state = self.state_dict()
        new_state = {}

        for k in list(state.keys()):
            if k.startswith("cnn_shared."):
                suffix = k[len("cnn_shared."):]
                k_fine = "cnn_fine." + suffix
                k_coarse = "cnn_coarse." + suffix
                if k_fine in state and k_coarse in state:
                    shared_val = 0.5 * (state[k_fine] + state[k_coarse])
                    new_state[k] = shared_val
                    new_state[k_fine] = shared_val.clone()
                    new_state[k_coarse] = shared_val.clone()
            elif k.startswith("pose_mlp_shared."):
                suffix = k[len("pose_mlp_shared."):]
                k_fine = "pose_fine." + suffix
                k_coarse = "pose_coarse." + suffix
                if k_fine in state and k_coarse in state:
                    shared_val = 0.5 * (state[k_fine] + state[k_coarse])
                    new_state[k] = shared_val
                    new_state[k_fine] = shared_val.clone()
                    new_state[k_coarse] = shared_val.clone()

        state.update(new_state)
        self.load_state_dict(state)
        print("🔁 Sincronização ciclo MT: shared backbone ↔ individuais (fine/coarse).")

    def forward(self, x_img, x_pose, task_type="fine"):
        # Escolhe backbone conforme estado (compartilhado vs. individual)
        if task_type == "fine" and self.use_private_fine:
            img_feat = self.cnn_fine(x_img).flatten(1)
            pose_feat = self.pose_fine(x_pose)
        elif task_type == "coarse" and self.use_private_coarse:
            img_feat = self.cnn_coarse(x_img).flatten(1)
            pose_feat = self.pose_coarse(x_pose)
        else:
            # ainda usando backbone compartilhado
            img_feat = self.cnn_shared(x_img).flatten(1)
            pose_feat = self.pose_mlp_shared(x_pose)

        fused = torch.cat([img_feat, pose_feat], dim=1)  # [B, fused_dim]

        if task_type == "fine":
            res = self.residual_fine(fused)
            z = fused + res
            return self.head_fine(z)
        else:
            res = self.residual_coarse(fused)
            z = fused + res
            return self.head_coarse(z)


###########################################
# MODELO GTSRB (tarefa independente)
###########################################
class GTSRBNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


###########################################
# SEPARAÇÃO VIA DIRICHLET
###########################################
def dirichlet_split(labels, num_clients, alpha=0.5):
    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions / proportions.sum()) * len(idx)
        proportions = proportions.astype(int)

        total = sum(proportions)
        diff = len(idx) - total
        if diff > 0:
            proportions[random.randint(0, num_clients - 1)] += diff

        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_indices[i].extend(idx[start:end])
            start = end

    return client_indices


###########################################
# MAPEAMENTO 10 → 3 CLASSES (coarse)
###########################################
COARSE_MAP_LIST = [0, 1, 2, 1, 1, 2, 1, 2, 2, 2]


def map_labels_to_coarse_torch(y, device):
    coarse_map = torch.tensor(COARSE_MAP_LIST, device=device)
    return coarse_map[y]


###########################################
# TREINAMENTO LOCAL (STATE FARM MULTI-TASK)
# → retorna também o número de amostras do cliente
###########################################
def local_train_statefarm(model, dataloader, task_type="fine", epochs=1):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x_img, x_pose, y in dataloader:
            x_img, x_pose, y = x_img.to(DEVICE), x_pose.to(DEVICE), y.to(DEVICE)

            if task_type == "coarse":
                y_task = map_labels_to_coarse_torch(y, DEVICE)
            else:
                y_task = y

            optimizer.zero_grad()
            out = model(x_img, x_pose, task_type=task_type)
            loss = loss_fn(out, y_task)
            loss.backward()
            optimizer.step()

    num_samples = len(dataloader.dataset)
    return model.state_dict(), num_samples


###########################################
# AVALIAÇÃO (STATE FARM MULTI-TASK)
###########################################
def evaluate_statefarm(model, dataloader, task_type="fine"):
    model = model.to(DEVICE)
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_img, x_pose, y in dataloader:
            x_img, x_pose, y = x_img.to(DEVICE), x_pose.to(DEVICE), y.to(DEVICE)

            if task_type == "coarse":
                y_task = map_labels_to_coarse_torch(y, DEVICE)
            else:
                y_task = y

            out = model(x_img, x_pose, task_type=task_type)
            loss = loss_fn(out, y_task)
            loss_sum += loss.item() * len(y_task)

            preds = out.argmax(1)
            correct += (preds == y_task).sum().item()
            total += len(y_task)

    return correct / total, loss_sum / total


###########################################
# TREINAMENTO LOCAL (GTSRB – tarefa independente)
# → retorna também o número de amostras do cliente
###########################################
def local_train_gtsrb(model, dataloader, epochs=1):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    num_samples = len(dataloader.dataset)
    return model.state_dict(), num_samples


def evaluate_gtsrb(model, dataloader):
    model = model.to(DEVICE)
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            loss_sum += loss.item() * len(y)

            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)

    return correct / total, loss_sum / total


###########################################
# FEDERATED AVERAGING STATE FARM (PARTILHA PARCIAL)
# Agora com média ponderada pelo nº de amostras
###########################################
def federated_aggregate_statefarm(global_model, client_states_fine, client_states_coarse):
    """
    Atualiza parâmetros do modelo StateFarm considerando:
    - Se a tarefa usa backbone compartilhado, seus updates afetam cnn_shared / pose_mlp_shared.
    - Se a tarefa usa backbone individual, seus updates afetam apenas cnn_fine/pose_fine ou
      cnn_coarse/pose_coarse (não o compartilhado).
    - Cabeças e resíduos sempre são atualizados por tarefa.

    client_states_fine:   lista de (state_dict, num_samples)
    client_states_coarse: lista de (state_dict, num_samples)
    """
    global_state = global_model.state_dict()

    all_states = client_states_fine + client_states_coarse

    use_private_fine = global_model.use_private_fine
    use_private_coarse = global_model.use_private_coarse

    def weighted_avg_param(param_name, states_list):
        """
        Faz média ponderada do parâmetro 'param_name' pelos tamanhos dos datasets.
        states_list: lista de (state_dict, num_samples)
        """
        weighted_sum = None
        total_weight = 0

        for sd, n in states_list:
            if param_name not in sd:
                continue
            tensor = sd[param_name].to(global_state[param_name].dtype)
            if weighted_sum is None:
                weighted_sum = tensor * n
            else:
                weighted_sum += tensor * n
            total_weight += n

        if weighted_sum is None or total_weight == 0:
            return

        global_state[param_name] = (weighted_sum / total_weight).type_as(global_state[param_name])

    for k in list(global_state.keys()):

        # Backbones compartilhados
        if k.startswith("cnn_shared.") or k.startswith("pose_mlp_shared."):
            # apenas tarefas que ainda usam backbone compartilhado contribuem
            sources = []
            if not use_private_fine:
                sources.extend(client_states_fine)
            if not use_private_coarse:
                sources.extend(client_states_coarse)
            weighted_avg_param(k, sources)

        # Backbones individuais
        elif k.startswith("cnn_fine.") or k.startswith("pose_fine."):
            weighted_avg_param(k, client_states_fine)

        elif k.startswith("cnn_coarse.") or k.startswith("pose_coarse."):
            weighted_avg_param(k, client_states_coarse)

        # Cabeças e resíduos de fine
        elif k.startswith("residual_fine.") or k.startswith("head_fine."):
            weighted_avg_param(k, client_states_fine)

        # Cabeças e resíduos de coarse
        elif k.startswith("residual_coarse.") or k.startswith("head_coarse."):
            weighted_avg_param(k, client_states_coarse)

        else:
            # Outros parâmetros (se existirem) são atualizados com base em todos
            weighted_avg_param(k, all_states)

    global_model.load_state_dict(global_state)


###########################################
# FEDERATED AVERAGING SIMPLES (GTSRB)
# Agora com média ponderada pelo nº de amostras
###########################################
def federated_aggregate_simple(global_model, client_states):
    """
    client_states: lista de (state_dict, num_samples)
    """
    if len(client_states) == 0:
        return

    global_state = global_model.state_dict()

    for k in global_state.keys():
        weighted_sum = None
        total_weight = 0

        for sd, n in client_states:
            if k not in sd:
                continue
            tensor = sd[k].to(global_state[k].dtype)
            if weighted_sum is None:
                weighted_sum = tensor * n
            else:
                weighted_sum += tensor * n
            total_weight += n

        if weighted_sum is not None and total_weight > 0:
            global_state[k] = (weighted_sum / total_weight).type_as(global_state[k])

    global_model.load_state_dict(global_state)


###########################################
# FUNÇÃO AUXILIAR: ATUALIZA CONVERGÊNCIA
###########################################
def update_convergence_state(task_state, acc, round_idx, task_key, converged_order):
    """
    Atualiza best_acc, since_improve e converged para uma tarefa.
    Convergência: após MIN_ROUND_FOR_CONV_CHECK, se acurácia não melhora mais que
    THRESHOLD_IMPROVEMENT por PLATEAU_T rodadas consecutivas.
    """
    if task_state["converged"]:
        return

    if acc > task_state["best_acc"]:
        task_state["best_acc"] = acc

    if round_idx < MIN_ROUND_FOR_CONV_CHECK:
        return

    if acc > task_state["best_acc"] + THRESHOLD_IMPROVEMENT:
        task_state["best_acc"] = acc
        task_state["since_improve"] = 0
    else:
        task_state["since_improve"] += 1

    if task_state["since_improve"] >= PLATEAU_T:
        task_state["converged"] = True
        converged_order.append(task_key)
        print(f"✅ Tarefa {task_key} convergiu na rodada {round_idx}.")


###########################################
# INICIALIZA CSVs (sobrescreve, opção 1)
###########################################
def init_csv_files():
    header = [
        "Round (t)",
        "Accuracy",
        "Loss",
        "Selected clients for training",
        "Alpha",
        "Dataset name",
        "Trained or not in current round"
    ]

    for csv_path in [CSV_STATEFARM_FINE, CSV_STATEFARM_COARSE, CSV_GTSRB]:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


###########################################
# MAIN
###########################################
def main():
    # Inicializa (sobrescreve) os CSVs
    init_csv_files()

    # ===================== STATE FARM =====================
    print("🔍 Carregando imagens State Farm...")
    img_paths, labels = load_image_paths()

    print("🤖 Obtendo poses State Farm...")
    poses = get_pose_features(img_paths)

    print("📌 Distribuindo dados State Farm via Dirichlet para", NUM_CLIENTS, "clientes...")
    client_splits_sf = dirichlet_split(labels, NUM_CLIENTS, alpha=ALPHA_STATEFARM)

    # cria datasets por cliente (State Farm)
    clients_statefarm = []
    for idxs in client_splits_sf:
        ds = MultiModalDataset(
            [img_paths[i] for i in idxs],
            [poses[i] for i in idxs],
            [labels[i] for i in idxs]
        )
        clients_statefarm.append(ds)

    num_classes_fine = len(set(labels))  # deve ser 10
    num_classes_coarse = 3

    # Modelo global multi-modal + multi-task (State Farm)
    global_model_statefarm = MultiTaskMultiModalNet(
        num_classes_fine=num_classes_fine,
        num_classes_coarse=num_classes_coarse
    ).to(DEVICE)

    # Grupos de clientes para State Farm:
    client_groups_sf = []
    for cid in range(NUM_CLIENTS):
        if cid < NUM_CLIENTS // 2:
            client_groups_sf.append("fine")   # 0-19
        else:
            client_groups_sf.append("coarse") # 20-39

    fine_clients = list(range(0, NUM_CLIENTS // 2))              # 0–19
    coarse_clients = list(range(NUM_CLIENTS // 2, NUM_CLIENTS))  # 20–39

    # ===================== GTSRB =====================
    print("\n🔍 Carregando GTSRB do Hugging Face (claudiogsc/GTSRB)...")
    gtsrb_train = load_dataset("claudiogsc/GTSRB", split="train")

    gtsrb_labels = np.array(gtsrb_train["label"])
    num_classes_gtsrb = len(set(gtsrb_labels))

    print("📌 Distribuindo dados GTSRB via Dirichlet para", NUM_CLIENTS, "clientes...")
    client_splits_gtsrb = dirichlet_split(gtsrb_labels, NUM_CLIENTS, alpha=ALPHA_GTSRB)

    gtsrb_clients = []
    for idxs in client_splits_gtsrb:
        ds = GTSRBDataset(gtsrb_train, idxs)
        gtsrb_clients.append(ds)

    global_model_gtsrb = GTSRBNet(num_classes=num_classes_gtsrb).to(DEVICE)

    # ===================== ESTADO DAS TAREFAS =====================
    tasks_state = {
        "sf_fine":   {"converged": False, "best_acc": 0.0, "since_improve": 0},
        "sf_coarse": {"converged": False, "best_acc": 0.0, "since_improve": 0},
        "gtsrb":     {"converged": False, "best_acc": 0.0, "since_improve": 0},
    }
    converged_order = []   # ordem em que tarefas convergiram (para round-robin)
    conv_rr_index = 0      # índice atual para round-robin entre convergidas

    # Controle de ciclo para tarefas correlacionadas (fine/coarse)
    correlated_tasks = ["sf_fine", "sf_coarse"]
    cycle_progress_sf = {tk: False for tk in correlated_tasks}

    # ===================== LOGS EM MEMÓRIA (por rodada) =====================
    round_accuracies_sf = []
    round_losses_sf = []

    round_accuracies_gtsrb = []
    round_losses_gtsrb = []

    print("\n🚀 Iniciando Federated Learning:")
    print("   - State Farm: multi-modal + multi-task (fine/coarse)")
    print("   - GTSRB: tarefa independente\n")

    for r in range(ROUNDS):
        print(f"\n===== ROUND {r+1} / {ROUNDS} =====")

        # ========= PLANEJAMENTO DAS TAREFAS NO ROUND =========
        tasks_to_train = {}

        # tarefas não convergidas treinam normalmente
        if not tasks_state["sf_fine"]["converged"]:
            tasks_to_train["sf_fine"] = CLIENTS_PER_TASK

        if not tasks_state["sf_coarse"]["converged"]:
            tasks_to_train["sf_coarse"] = CLIENTS_PER_TASK

        if not tasks_state["gtsrb"]["converged"]:
            tasks_to_train["gtsrb"] = CLIENTS_PER_TASK

        # tarefas convergidas: round-robin com soma de clientes
        converged_tasks = [k for k, v in tasks_state.items() if v["converged"]]
        if len(converged_tasks) >= 2 and len(converged_order) > 0:
            round_task = converged_order[conv_rr_index % len(converged_order)]
            conv_rr_index = (conv_rr_index + 1) % len(converged_order)
            extra_clients = CLIENTS_PER_TASK * len(converged_tasks)
            tasks_to_train[round_task] = tasks_to_train.get(round_task, 0) + extra_clients
            print(f"🔁 Round-robin convergidas: {converged_tasks}, treinando hoje: {round_task} com {extra_clients} clientes extras.")

        print("Plano de treinamento neste round:")
        for tk, nc in tasks_to_train.items():
            print(f"  - {tk}: {nc} clientes")

        # flags e listas de clientes treinados neste round
        sf_fine_trained_this_round = False
        sf_coarse_trained_this_round = False
        gtsrb_trained_this_round = False

        selected_fine_clients = []
        selected_coarse_clients = []
        selected_gtsrb_clients = []

        # ========= STATE FARM: TREINAMENTO LOCAL =========
        client_states_fine = []    # lista de (state_dict, num_samples)
        client_states_coarse = []  # lista de (state_dict, num_samples)

        # Fine
        if "sf_fine" in tasks_to_train and tasks_to_train["sf_fine"] > 0:
            n_clients_fine = min(tasks_to_train["sf_fine"], len(fine_clients))
            selected_fine = random.sample(fine_clients, n_clients_fine)
            selected_fine_clients = selected_fine
            sf_fine_trained_this_round = True if n_clients_fine > 0 else False
            print(f"State Farm - Fine   → clientes: {selected_fine}")

            for cid in selected_fine:
                dl = DataLoader(
                    clients_statefarm[cid],
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                local_state, n_samples = local_train_statefarm(global_model_statefarm, dl, task_type="fine")
                client_states_fine.append((local_state, n_samples))
        else:
            print("State Farm - Fine   → não treinada neste round.")

        # Coarse
        if "sf_coarse" in tasks_to_train and tasks_to_train["sf_coarse"] > 0:
            n_clients_coarse = min(tasks_to_train["sf_coarse"], len(coarse_clients))
            selected_coarse = random.sample(coarse_clients, n_clients_coarse)
            selected_coarse_clients = selected_coarse
            sf_coarse_trained_this_round = True if n_clients_coarse > 0 else False
            print(f"State Farm - Coarse → clientes: {selected_coarse}")

            for cid in selected_coarse:
                dl = DataLoader(
                    clients_statefarm[cid],
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                local_state, n_samples = local_train_statefarm(global_model_statefarm, dl, task_type="coarse")
                client_states_coarse.append((local_state, n_samples))
        else:
            print("State Farm - Coarse → não treinada neste round.")

        # Agregação federada State Farm (FedAvg ponderado)
        federated_aggregate_statefarm(global_model_statefarm, client_states_fine, client_states_coarse)

        # ========= GTSRB: TREINAMENTO LOCAL =========
        client_states_gtsrb = []   # lista de (state_dict, num_samples)
        all_clients = list(range(NUM_CLIENTS))

        if "gtsrb" in tasks_to_train and tasks_to_train["gtsrb"] > 0:
            n_clients_g = min(tasks_to_train["gtsrb"], len(all_clients))
            selected_gtsrb = random.sample(all_clients, n_clients_g)
            selected_gtsrb_clients = selected_gtsrb
            gtsrb_trained_this_round = True if n_clients_g > 0 else False
            print(f"GTSRB → clientes: {selected_gtsrb}")

            for cid in selected_gtsrb:
                dl_g = DataLoader(
                    gtsrb_clients[cid],
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                local_state_g, n_samples_g = local_train_gtsrb(global_model_gtsrb, dl_g)
                client_states_gtsrb.append((local_state_g, n_samples_g))
        else:
            print("GTSRB → não treinado neste round.")

        # Agregação federada GTSRB (FedAvg ponderado)
        federated_aggregate_simple(global_model_gtsrb, client_states_gtsrb)

        # ========= AVALIAÇÃO STATE FARM (MÉDIA PONDERADA) =========
        acc_sum_all_sf = 0.0
        loss_sum_all_sf = 0.0
        total_all_sf = 0

        acc_sum_fine = 0.0
        loss_sum_fine = 0.0
        total_fine = 0

        acc_sum_coarse = 0.0
        loss_sum_coarse = 0.0
        total_coarse = 0

        for cid in range(NUM_CLIENTS):
            dl_test = DataLoader(
                clients_statefarm[cid],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            task_type = client_groups_sf[cid]
            acc, loss = evaluate_statefarm(global_model_statefarm, dl_test, task_type=task_type)

            n_samples_client = len(clients_statefarm[cid])

            # global (fine + coarse)
            acc_sum_all_sf += acc * n_samples_client
            loss_sum_all_sf += loss * n_samples_client
            total_all_sf += n_samples_client

            if task_type == "fine":
                acc_sum_fine += acc * n_samples_client
                loss_sum_fine += loss * n_samples_client
                total_fine += n_samples_client
            else:
                acc_sum_coarse += acc * n_samples_client
                loss_sum_coarse += loss * n_samples_client
                total_coarse += n_samples_client

        mean_acc_sf = acc_sum_all_sf / total_all_sf if total_all_sf > 0 else 0.0
        mean_loss_sf = loss_sum_all_sf / total_all_sf if total_all_sf > 0 else 0.0
        round_accuracies_sf.append(mean_acc_sf)
        round_losses_sf.append(mean_loss_sf)

        mean_acc_fine = acc_sum_fine / total_fine if total_fine > 0 else 0.0
        mean_loss_fine = loss_sum_fine / total_fine if total_fine > 0 else 0.0

        mean_acc_coarse = acc_sum_coarse / total_coarse if total_coarse > 0 else 0.0
        mean_loss_coarse = loss_sum_coarse / total_coarse if total_coarse > 0 else 0.0

        print(f"\n📊 STATE FARM - Acurácia global (fine + coarse, ponderada): {mean_acc_sf:.4f} | Loss global: {mean_loss_sf:.4f}")
        print(f"  ▶ Grupo 1 (10 classes - fine)  - acc (pond): {mean_acc_fine:.4f} | loss (pond): {mean_loss_fine:.4f}")
        print(f"  ▶ Grupo 2 (3 classes  - coarse) - acc (pond): {mean_acc_coarse:.4f} | loss (pond): {mean_loss_coarse:.4f}")

        # ========= AVALIAÇÃO GTSRB (MÉDIA PONDERADA) =========
        acc_sum_gtsrb = 0.0
        loss_sum_gtsrb = 0.0
        total_gtsrb = 0

        for cid in range(NUM_CLIENTS):
            dl_test_g = DataLoader(
                gtsrb_clients[cid],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            acc_g, loss_g = evaluate_gtsrb(global_model_gtsrb, dl_test_g)
            n_samples_client_g = len(gtsrb_clients[cid])

            acc_sum_gtsrb += acc_g * n_samples_client_g
            loss_sum_gtsrb += loss_g * n_samples_client_g
            total_gtsrb += n_samples_client_g

        mean_acc_gtsrb = acc_sum_gtsrb / total_gtsrb if total_gtsrb > 0 else 0.0
        mean_loss_gtsrb = loss_sum_gtsrb / total_gtsrb if total_gtsrb > 0 else 0.0
        round_accuracies_gtsrb.append(mean_acc_gtsrb)
        round_losses_gtsrb.append(mean_loss_gtsrb)

        print(f"\n🚦 GTSRB - Acurácia média ponderada entre clientes: {mean_acc_gtsrb:.4f} | Loss média ponderada: {mean_loss_gtsrb:.4f}")

        # ========= LOG EM CSV (por tarefa) =========
        round_t = r + 1

        # StateFarm Fine
        with open(CSV_STATEFARM_FINE, mode="a", newline="") as f_fine:
            writer_fine = csv.writer(f_fine)
            writer_fine.writerow([
                round_t,
                mean_acc_fine,
                mean_loss_fine,
                str(selected_fine_clients),
                ALPHA_STATEFARM,
                "StateFarm-fine",
                1 if sf_fine_trained_this_round else 0
            ])

        # StateFarm Coarse
        with open(CSV_STATEFARM_COARSE, mode="a", newline="") as f_coarse:
            writer_coarse = csv.writer(f_coarse)
            writer_coarse.writerow([
                round_t,
                mean_acc_coarse,
                mean_loss_coarse,
                str(selected_coarse_clients),
                ALPHA_STATEFARM,
                "StateFarm-coarse",
                1 if sf_coarse_trained_this_round else 0
            ])

        # GTSRB
        with open(CSV_GTSRB, mode="a", newline="") as f_gtsrb:
            writer_gtsrb = csv.writer(f_gtsrb)
            writer_gtsrb.writerow([
                round_t,
                mean_acc_gtsrb,
                mean_loss_gtsrb,
                str(selected_gtsrb_clients),
                ALPHA_GTSRB,
                "GTSRB",
                1 if gtsrb_trained_this_round else 0
            ])

        # ========= ATUALIZA CONVERGÊNCIA DAS 3 TAREFAS =========
        update_convergence_state(tasks_state["sf_fine"],   mean_acc_fine,   round_t, "sf_fine",   converged_order)
        update_convergence_state(tasks_state["sf_coarse"], mean_acc_coarse, round_t, "sf_coarse", converged_order)
        update_convergence_state(tasks_state["gtsrb"],     mean_acc_gtsrb,  round_t, "gtsrb",     converged_order)

        # Ao convergir uma tarefa correlacionada, ela passa a ter backbone individual
        if tasks_state["sf_fine"]["converged"] and not global_model_statefarm.use_private_fine:
            global_model_statefarm.ensure_private_backbone("fine", device=DEVICE)

        if tasks_state["sf_coarse"]["converged"] and not global_model_statefarm.use_private_coarse:
            global_model_statefarm.ensure_private_backbone("coarse", device=DEVICE)

        # ========= CONTROLE DE CICLO MULTI-TASK (fine/coarse) =========
        # Marcamos quais tarefas correlacionadas convergidas foram treinadas neste round
        if tasks_state["sf_fine"]["converged"] and sf_fine_trained_this_round:
            cycle_progress_sf["sf_fine"] = True
        if tasks_state["sf_coarse"]["converged"] and sf_coarse_trained_this_round:
            cycle_progress_sf["sf_coarse"] = True

        # Se ambas correlacionadas estão convergidas e já treinaram pelo menos uma vez
        # desde a última sincronização, fechamos um ciclo e sincronizamos backbone.
        if all(tasks_state[tk]["converged"] for tk in correlated_tasks) and \
           all(cycle_progress_sf[tk] for tk in correlated_tasks):

            print("\n🔄 Ciclo completo de round-robin entre tarefas correlacionadas (fine/coarse).")
            print("   → Atualizando backbone compartilhado e individuais.")
            global_model_statefarm.sync_shared_and_private(device=DEVICE)

            # reset para próximo ciclo
            for tk in correlated_tasks:
                cycle_progress_sf[tk] = False

    # ===================== RESULTADOS FINAIS =====================
    print("\n===============================")
    print("📌 RESULTADOS FINAIS")
    print("STATE FARM (multi-modal + multi-task):")
    print(f"  🎯 Acurácia média global (entre rounds): {np.mean(round_accuracies_sf):.4f}")
    print(f"  📉 Loss média global:                  {np.mean(round_losses_sf):.4f}")

    print("\nGTSRB (tarefa independente):")
    print(f"  🎯 Acurácia média global (entre rounds): {np.mean(round_accuracies_gtsrb):.4f}")
    print(f"  📉 Loss média global:                   {np.mean(round_losses_gtsrb):.4f}")
    print("===============================")


if __name__ == "__main__":
    main()
