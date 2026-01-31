BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/foursquare/"
FILENAME = "fdata.txt"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 5
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-5          # menor para evitar overfitting
EMBED_DIM = 64
HIDDEN_DIM = 128
MIN_VENUE_FREQ = 150   # filtrar venues raros
#########################################
# 1. LOAD DATA
#########################################

df = pd.read_csv(
    f"{BASE}{FILENAME}",
    sep="\t",
    header=None,
    names=["userid", "venueid", "datetime", "lat", "lng"]
).head(6000000)

print(df.head())
print(df.columns)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"])

# weekday: 0=Monday ... 6=Sunday
df["weekday"] = df["datetime"].dt.weekday.astype(int)

# delta time em horas entre check-ins do mesmo usuário
df = df.sort_values(["userid", "datetime"])
df["delta_t"] = df.groupby("userid")["datetime"].diff().dt.total_seconds() / 3600.0

# preencher NaN do primeiro check-in de cada usuário
df["delta_t"] = df["delta_t"].fillna(0.0)

# discretizar delta_t (bins)
df["delta_t_bin"] = pd.cut(
    df["delta_t"],
    bins=[-1, 0.5, 2, 6, 24, 72, 1e9],
    labels=[0,1,2,3,4,5]
).astype(int)

print("Weekday range:", df["weekday"].min(), df["weekday"].max())
print("Delta_t_bin range:", df["delta_t_bin"].min(), df["delta_t_bin"].max())


df["hour"] = df["datetime"].dt.hour.astype(int)

df = df.sort_values(["userid", "datetime"])

print("Hour range:", df["hour"].min(), df["hour"].max())
#########################################
# 2. FILTER RARE VENUES
#########################################

df = df.groupby("venueid").filter(lambda x: len(x) >= MIN_VENUE_FREQ)

#########################################
# 3. SPLIT BY USER (NO LEAKAGE)
#########################################

users = df["userid"].unique()
np.random.shuffle(users)

train_users = set(users[:int(0.8 * len(users))])
test_users  = set(users[int(0.8 * len(users)):])

train_df = df[df["userid"].isin(train_users)]
test_df  = df[df["userid"].isin(test_users)]

print("Train users:", len(train_users))
print("Test users:", len(test_users))
#########################################
# 4. ENCODE VENUE IDS
#########################################

le_venue = LabelEncoder()
train_df["venue_id_enc"] = le_venue.fit_transform(train_df["venueid"])

test_df = test_df[test_df["venueid"].isin(le_venue.classes_)]
test_df["venue_id_enc"] = le_venue.transform(test_df["venueid"])

num_classes = len(le_venue.classes_)

print("Num classes:", num_classes)
#########################################
# 5. BUILD SEQUENCES
#########################################

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

    return np.array(sequences, dtype=np.int64), np.array(labels, dtype=np.int64)

train_X, train_y = build_sequences(train_df)
test_X, test_y = build_sequences(test_df)

print("Train samples:", len(train_X))
print("Test samples:", len(test_X))
#########################################
# 6. DATASET
#########################################

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)
#########################################
# 7. CLASS WEIGHTS
#########################################

label_counts = Counter(train_y)

class_weights = np.zeros(num_classes, dtype=np.float32)
for cls in range(num_classes):
    class_weights[cls] = 1.0 / label_counts.get(cls, 1)

class_weights = class_weights / class_weights.sum() * num_classes
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
#########################################
# 8. MODEL
#########################################

class NextPlaceModel(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim):
        super().__init__()

        self.venue_embedding = nn.Embedding(num_classes, embed_dim)
        self.hour_embedding = nn.Embedding(24, embed_dim)
        self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.delta_embedding = nn.Embedding(6, embed_dim)  # 6 bins

        self.lstm = nn.LSTM(
            embed_dim * 4,
            hidden_dim,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        venues = x[:, :, 0]
        hours = x[:, :, 1]
        weekdays = x[:, :, 2]
        deltas = x[:, :, 3]

        v_emb = self.venue_embedding(venues)
        h_emb = self.hour_embedding(hours)
        w_emb = self.weekday_embedding(weekdays)
        d_emb = self.delta_embedding(deltas)

        x = torch.cat([v_emb, h_emb, w_emb, d_emb], dim=2)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

model = NextPlaceModel(num_classes, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#########################################
# 9. METRIC: TOP-K ACCURACY
#########################################

def top_k_accuracy(outputs, targets, k=5):
    _, topk = outputs.topk(k, dim=1)
    correct = topk.eq(targets.view(-1, 1))
    return correct.any(dim=1).float().mean().item()
#########################################
# 10. TRAINING + TEST PER EPOCH
#########################################

for epoch in range(EPOCHS):

    ###################
    # TRAIN
    ###################
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]"):
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total


    ###################
    # TEST
    ###################
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    test_top5 = 0

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TEST]"):
            X, y = X.to(DEVICE), y.to(DEVICE)

            outputs = model(X)
            loss = criterion(outputs, y)

            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

            test_top5 += top_k_accuracy(outputs, y, k=5) * y.size(0)

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    test_top5 = test_top5 / test_total


    ###################
    # PRINT METRICS
    ###################
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f} | Top-5 Acc: {test_top5:.4f}")
    print("-" * 50)
