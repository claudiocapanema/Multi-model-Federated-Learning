import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import pairwise_distances
import random

# ==========================
# Configurações
# ==========================
num_clients = 10
num_clusters = 5  # CIFAR-10 → 10 clusters (usamos 10 por conveniência)
fuzziness = 2.0
fcm_max_iter = 30
accept_delta = 0.01  # limiar para detecção de drift
batch_size = 128
rounds = 10  # conforme solicitado pelo usuário
participation_rate = 0.3  # Apenas 30% dos clientes treinam por rodada
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Carregar CIFAR-10 (pequeno pipeline)
# ==========================
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=base_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=base_transform)

# Dividir CIFAR-10 em clientes (distribuição horizontal simples)
per_client = len(trainset_full) // num_clients
client_subsets = []
indices = list(range(len(trainset_full)))
for i in range(num_clients):
    sub_idx = indices[i*per_client:(i+1)*per_client]
    client_subsets.append(torch.utils.data.Subset(trainset_full, sub_idx))

# Para acelerar: reduzimos cada cliente a um subconjunto menor por padrão no clustering
cluster_sample_per_client = 1000


def sample_for_clustering(dataset, n=cluster_sample_per_client):
    n = min(n, len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
    images, _ = next(iter(loader))
    return images.view(len(images), -1).numpy()


# ==========================
# Modelo simples para avaliar acurácia
# ==========================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss/total, 100*correct/total


def evaluate_global(model, testloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss/total, 100*correct/total


# ==========================
# Fuzzy C-Means (Local)
# ==========================
def fuzzy_c_means(X, c, m=2, max_iter=100, eps=1e-5):
    n = X.shape[0]
    # Inicialização: Dirichlet garante somatório 1
    U = np.random.dirichlet(np.ones(c), size=n)
    for _ in range(max_iter):
        U_old = U.copy()
        centers = (U**m).T @ X / np.sum(U**m, axis=0)[:, None]
        dist = pairwise_distances(X, centers, metric='euclidean') + 1e-8
        U = 1.0 / (dist ** (2/(m-1)))
        U = U / U.sum(axis=1, keepdims=True)
        if np.linalg.norm(U - U_old) < eps:
            break
    return centers, U


# ==========================
# Federated Fuzzy C-Means (servidor faz média simples de centros locais)
# ==========================
def federated_fcm(clients, c, m=2, rounds=5):
    # inicializa centros com amostra do cliente 0
    X0 = sample_for_clustering(clients[0])
    centers, _ = fuzzy_c_means(X0, c, m, max_iter=10)

    for _ in range(rounds):
        local_centers = []
        for client in clients:
            Xc = sample_for_clustering(client)
            _, Uc = fuzzy_c_means(Xc, c, m, max_iter=10)
            Cc = (Uc**m).T @ Xc / np.sum(Uc**m, axis=0)[:, None]
            local_centers.append(Cc)
        centers = np.mean(np.stack(local_centers, axis=0), axis=0)
    return centers


# ==========================
# Federated Fuzzy Davies-Bouldin Index
# ==========================
def fuzzy_davies_bouldin(X, centers, U, m=2):
    K = centers.shape[0]
    S = np.zeros(K)
    for j in range(K):
        denom = np.sum(U[:, j]**m)
        if denom <= 0:
            S[j] = 0.0
        else:
            S[j] = np.sum((U[:, j]**m) * np.linalg.norm(X - centers[j], axis=1)) / denom
    M = pairwise_distances(centers) + 1e-8
    R = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                R[i, j] = (S[i] + S[j]) / M[i, j]
    Ri = np.max(R, axis=1)
    D = np.mean(Ri)
    return D


# ==========================
# Geradores de drift (simulam Xt para cada cliente)
# ==========================
# Scenarios:
# A.1 - No drift at all (Xt ~ X0)
# A.2 - Local only drift (each client replaces its new data by another client's initial distribution)
# B.1 - Global drift: unseen distribution (strong color jitter + noise applied to all clients)
# B.2 - Global drift: disappearing distributions (some classes removed from some clients)

base_augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

strong_augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3),
    transforms.GaussianBlur(kernel_size=(5,5)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


def build_new_batch(client_dataset, scenario='A1', rho=1.0, disappeared_fraction=0.0, apply_drift=False):
    """Retorna um objecto Dataset representando o novo lote Xt para um cliente.
    rho controla fração de pontos 'novos' (apenas em B.1/B.2 faz sentido)
    apply_drift: se False, retorna dados normais (sem drift) — usado para garantir que o drift só começa na rodada 5
    """
    # Pegamos um subset dos dados locais (pequeno para velocidade)
    n = min(500, len(client_dataset))
    loader = torch.utils.data.DataLoader(client_dataset, batch_size=n, shuffle=True)
    imgs, labels = next(iter(loader))

    if scenario == 'A1' or not apply_drift:
        # Sem drift: retorna os mesmos dados
        return torch.utils.data.TensorDataset(imgs, labels)

    if scenario == 'A2' and apply_drift:
        # Local-only drift: permutar para dados de outro cliente será feito externamente
        return torch.utils.data.TensorDataset(imgs, labels)

    if scenario == 'B1' and apply_drift:
        # Global unseen: aplicar forte transformação a uma fração rho das imagens
        n_new = int(rho * n)
        imgs_new = imgs.clone()
        for i in range(n_new):
            pil = transforms.ToPILImage()(imgs_new[i].cpu())
            aug = strong_augment(pil)
            imgs_new[i] = aug
        # manter parte antiga
        return torch.utils.data.TensorDataset(imgs_new, labels)

    if scenario == 'B2' and apply_drift:
        # Disappearing distributions: remove aleatoriamente algumas classes (por disappeared_fraction)
        labels_np = labels.numpy()
        unique = np.unique(labels_np)
        k = max(1, int(len(unique) * disappeared_fraction))
        remove_classes = set(np.random.choice(unique, size=k, replace=False))
        keep_mask = [l not in remove_classes for l in labels_np]
        if sum(keep_mask) < 10:
            # se removemos demais, volta para original
            return torch.utils.data.TensorDataset(imgs, labels)
        imgs_keep = imgs[keep_mask]
        labels_keep = labels[keep_mask]
        # preencher até n com imagens antigas (simula rho)
        n_keep = len(imgs_keep)
        n_fill = max(0, n - n_keep)
        if n_fill > 0:
            imgs_fill = imgs[:n_fill]
            labels_fill = labels[:n_fill]
            imgs_out = torch.cat([imgs_keep, imgs_fill], dim=0)
            labels_out = torch.cat([labels_keep, labels_fill], dim=0)
        else:
            imgs_out = imgs_keep[:n]
            labels_out = labels_keep[:n]
        return torch.utils.data.TensorDataset(imgs_out, labels_out)

    return torch.utils.data.TensorDataset(imgs, labels)


# ==========================
# Função principal: simula rounds federados em cada cenário
# ==========================

def run_simulation(scenario='A1'):
    print(f"=== Executando cenário {scenario} ===")

    # inicializa modelo global
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Inicial: aprenda centros iniciais M0 (fuzzy c-means) usando amostras iniciais de cada cliente
    centers = federated_fcm(client_subsets, num_clusters, m=fuzziness, rounds=5)
    # construir X0 (amostras concatenadas) para índice inicial
    X0_list = [sample_for_clustering(c) for c in client_subsets]
    X0 = np.vstack(X0_list)
    _, U0 = fuzzy_c_means(X0, num_clusters, fuzziness, max_iter=fcm_max_iter)
    D0 = fuzzy_davies_bouldin(X0, centers, U0, fuzziness)
    print(f"Índice inicial DB: {D0:.4f}")

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Configuração: o drift deve começar na rodada 5 (inclusive) e permanecer a partir daí.
    drift_start_round = 5

    for rnd in range(1, rounds+1):
        # selecionar clientes que irão treinar nesta rodada
        n_part = max(1, int(participation_rate * len(client_subsets)))
        selected_idx = random.sample(range(len(client_subsets)), n_part)
        selected = [client_subsets[i] for i in selected_idx]

        # cada cliente treina localmente por um epoch e envia pesos
        local_states = []
        for idx, cdata in zip(selected_idx, selected):
            # determinar se aplicamos drift neste cliente (aplica-se a todos os clientes quando for cenário global)
            apply_drift = (rnd >= drift_start_round)

            # para cenários que requerem B1/B2/A2 ajustamos o batch de treino local
            if scenario == 'A1':
                train_dataset = build_new_batch(cdata, scenario='A1', apply_drift=False)
            elif scenario == 'A2':
                # local-only drift: cada cliente recebe dados de outro cliente APENAS quando apply_drift=True
                if apply_drift:
                    other_idx = (idx + rnd) % len(client_subsets)
                    train_dataset = build_new_batch(client_subsets[other_idx], scenario='A1', apply_drift=False)
                else:
                    train_dataset = build_new_batch(cdata, scenario='A1', apply_drift=False)
            elif scenario == 'B1':
                # todos os clientes obtêm Xt com forte transformação a partir da rodada de drift
                train_dataset = build_new_batch(cdata, scenario='B1', rho=0.7, apply_drift=apply_drift)
            elif scenario == 'B2':
                train_dataset = build_new_batch(cdata, scenario='B2', disappeared_fraction=0.5, apply_drift=apply_drift)
            else:
                train_dataset = build_new_batch(cdata, scenario='A1', apply_drift=False)

            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # cópia local do modelo
            local_model = SimpleNN().to(device)
            local_model.load_state_dict(model.state_dict())
            local_opt = optim.SGD(local_model.parameters(), lr=0.01)

            loss, acc = train_one_epoch(local_model, loader, local_opt, nn.CrossEntropyLoss())
            # coletar estado
            local_states.append(local_model.state_dict())
            print(f"Rodada {rnd} - Cliente {idx} treina: loss={loss:.4f}, acc={acc:.2f}%")

        # Agregação simples: média aritmética dos parâmetros dos clientes participantes
        global_state = model.state_dict()
        for k in global_state.keys():
            global_state[k] = torch.zeros_like(global_state[k])
        for state in local_states:
            for k in global_state.keys():
                global_state[k] += state[k]
        for k in global_state.keys():
            global_state[k] = global_state[k] / len(local_states)
        model.load_state_dict(global_state)

        # avaliação global 
        g_loss, g_acc = evaluate_global(model, testloader, nn.CrossEntropyLoss())
        print(f"-- Após rodada {rnd}: Global Loss={g_loss:.4f}, Global Acurácia={g_acc:.2f}%")

        # Construir novo lote global X_t concatenando os Xt de cada cliente conforme cenário
        Xt_list = []
        for i, cdata in enumerate(client_subsets):
            # decidir se apply_drift para este cliente (global scenarios: all clients apply when rnd>=drift_start_round)
            apply_drift = (rnd >= drift_start_round)
            if scenario == 'A1':
                ds = build_new_batch(cdata, scenario='A1', apply_drift=False)
            elif scenario == 'A2':
                if apply_drift:
                    other_idx = (i + rnd) % len(client_subsets)
                    ds = build_new_batch(client_subsets[other_idx], scenario='A1', apply_drift=True)
                else:
                    ds = build_new_batch(cdata, scenario='A1', apply_drift=False)
            elif scenario == 'B1':
                ds = build_new_batch(cdata, scenario='B1', rho=0.7, apply_drift=apply_drift)
            elif scenario == 'B2':
                ds = build_new_batch(cdata, scenario='B2', disappeared_fraction=0.5, apply_drift=apply_drift)
            else:
                ds = build_new_batch(cdata, scenario='A1', apply_drift=False)
            # extrair features
            imgs, _ = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds))))
            Xt_list.append(imgs.view(len(imgs), -1).numpy())

        Xt = np.vstack(Xt_list)
        _, U_t = fuzzy_c_means(Xt, num_clusters, fuzziness, max_iter=fcm_max_iter)
        D_t = fuzzy_davies_bouldin(Xt, centers, U_t, fuzziness)

        if not ((1-accept_delta)*D0 <= D_t <= (1+accept_delta)*D0):
            print(f"  -> Drift detectado na rodada {rnd} (Δt={D_t:.4f}, Δ0={D0:.4f})")
        else:
            print(f"  -> Sem drift na rodada {rnd} (Δt={D_t:.4f})")

    print(f"=== Fim do cenário {scenario} ===\n")


# ==========================
# Executar os 4 cenários do artigo
# ==========================
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    run_simulation('A1')  # A.1 - No drift
    run_simulation('A2')  # A.2 - Local-only drift (drift começa na rodada 5)
    run_simulation('B1')  # B.1 - Global unseen drift (drift começa na rodada 5)
    run_simulation('B2')  # B.2 - Global disappearing distributions (drift começa na rodada 5)

    print('Simulações concluídas.')