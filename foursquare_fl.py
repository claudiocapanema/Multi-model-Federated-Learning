import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import Counter
import random
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/foursquare/"
FILENAME = "fdata.txt"

BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/dataset_TIST2015/"
FILENAME = "dataset_TIST2015_Checkins.txt"

SEQ_LEN = 5
BATCH_SIZE = 64
GLOBAL_ROUNDS = 200
LOCAL_EPOCHS = 2
LR = 1e-5
EMBED_DIM = 64
HIDDEN_DIM = 128
MIN_VENUE_FREQ = 700

NUM_CLIENTS = 20
CLIENT_FRAC = 0.3   # 30% por rodada
DIRICHLET_ALPHA = 0.1

#########################################
# 1. LOAD DATA
#########################################

df = pd.read_csv(
    f"{BASE}{FILENAME}",
    sep="\t",
    header=None,
    names=["userid", "venueid", "datetime", "lat", "lng"]
)[["userid", "venueid", "datetime"]].head(6000000)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"])

df["weekday"] = df["datetime"].dt.weekday.astype(int)
df = df.sort_values(["userid", "datetime"])

df["delta_t"] = df.groupby("userid")["datetime"].diff().dt.total_seconds() / 3600.0
df["delta_t"] = df["delta_t"].fillna(0.0)

df["delta_t_bin"] = pd.cut(
    df["delta_t"],
    bins=[-1, 0.5, 2, 6, 24, 72, 1e9],
    labels=[0,1,2,3,4,5]
).astype(int)

df["hour"] = df["datetime"].dt.hour.astype(int)

#########################################
# 2. FILTER RARE VENUES
#########################################

df = df.groupby("venueid").filter(lambda x: len(x) >= MIN_VENUE_FREQ)

#########################################
# 3. SPLIT BY USER
#########################################

users = df["userid"].unique()
np.random.shuffle(users)

train_users = set(users[:int(0.8 * len(users))])
test_users  = set(users[int(0.8 * len(users)):])

train_df = df[df["userid"].isin(train_users)]
test_df  = df[df["userid"].isin(test_users)]

#########################################
# 4. ENCODE VENUES
#########################################

le_venue = LabelEncoder()
train_df["venue_id_enc"] = le_venue.fit_transform(train_df["venueid"])

test_df = test_df[test_df["venueid"].isin(le_venue.classes_)]
test_df["venue_id_enc"] = le_venue.transform(test_df["venueid"])

num_classes = len(le_venue.classes_)
print("Num classes:", num_classes)

def build_sequences(df):
    sequences = []
    labels = []

    for user_id, user_data in df.groupby("userid"):
        venues = user_data["venue_id_enc"].values.astype(np.int64)
        hours = user_data["hour"].values.astype(np.int64)
        weekdays = user_data["weekday"].values.astype(np.int64)
        deltas = user_data["delta_t_bin"].values.astype(np.int64)

        for i in range(len(venues) - SEQ_LEN):
            seq = np.stack([
                venues[i:i+SEQ_LEN],
                hours[i:i+SEQ_LEN],
                weekdays[i:i+SEQ_LEN],
                deltas[i:i+SEQ_LEN]
            ], axis=1)

            sequences.append(seq)
            labels.append(venues[i+SEQ_LEN])

    return np.array(sequences), np.array(labels)

train_X, train_y = build_sequences(train_df)
test_X, test_y = build_sequences(test_df)

print(f"Num samples train: ", len(train_X))
print(f"Num samples test: ", len(test_X))

class FoursquareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FoursquareDataset(train_X, train_y)
test_dataset  = FoursquareDataset(test_X, test_y)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
def dirichlet_partition(dataset, labels, num_clients, alpha):
    client_indices = [[] for _ in range(num_clients)]
    labels = np.array(labels)
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]

        split = np.split(idx, proportions)

        for i in range(num_clients):
            client_indices[i].extend(split[i])

    return client_indices

client_indices = dirichlet_partition(train_dataset, train_y, NUM_CLIENTS, DIRICHLET_ALPHA)

client_loaders = []
for i in range(NUM_CLIENTS):
    subset = Subset(train_dataset, client_indices[i])
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    client_loaders.append(loader)
class NextPlaceModel(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim):
        super().__init__()

        self.venue_embedding = nn.Embedding(num_classes, embed_dim)
        self.hour_embedding = nn.Embedding(24, embed_dim)
        self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.delta_embedding = nn.Embedding(6, embed_dim)

        self.lstm = nn.LSTM(embed_dim * 4, hidden_dim, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        venues = x[:,:,0]
        hours = x[:,:,1]
        weekdays = x[:,:,2]
        deltas = x[:,:,3]

        x = torch.cat([
            self.venue_embedding(venues),
            self.hour_embedding(hours),
            self.weekday_embedding(weekdays),
            self.delta_embedding(deltas)
        ], dim=2)

        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)
def local_train(model, loader, epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    return model.state_dict(), len(loader.dataset)


def fedavg(global_model, client_states, client_sizes):
    new_state = copy.deepcopy(global_model.state_dict())
    total = sum(client_sizes)

    for k in new_state.keys():
        new_state[k] = sum(client_states[i][k] * (client_sizes[i]/total)
                           for i in range(len(client_states)))

    global_model.load_state_dict(new_state)


def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            loss_sum += loss.item()
            preds = torch.argmax(out,1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return loss_sum/len(loader), correct/total
global_model = NextPlaceModel(num_classes, EMBED_DIM, HIDDEN_DIM).to(DEVICE)

for round in range(GLOBAL_ROUNDS):

    selected_clients = random.sample(range(NUM_CLIENTS), int(NUM_CLIENTS * CLIENT_FRAC))

    client_states = []
    client_sizes = []

    print(f"\n🌍 Round {round+1}/{GLOBAL_ROUNDS}")

    for cid in selected_clients:
        local_model = copy.deepcopy(global_model).to(DEVICE)
        state, size = local_train(local_model, client_loaders[cid], LOCAL_EPOCHS)

        client_states.append(state)
        client_sizes.append(size)

    fedavg(global_model, client_states, client_sizes)

    test_loss, test_acc = evaluate(global_model, test_loader)

    print(f"📊 Global Test Loss: {test_loss:.4f} | Global Test Acc: {test_acc:.4f}")
    print("="*60)
