BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Documentos/gowalla/"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 10
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 4
HIDDEN_DIM = 16

#########################################
# 1. LOAD DATA
#########################################

#########################################
# 1. LOAD DATA
#########################################

df = pd.read_csv(f"{BASE}gowalla_checkins_texas.csv")

df = df.sort_values(["userid", "datetime"])

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"])

df["hour"] = df["datetime"].dt.hour.astype(int)

le_cat = LabelEncoder()
df["cat_id"] = le_cat.fit_transform(df["category"])
num_classes = len(le_cat.classes_)

print("Hour range:", df["hour"].min(), df["hour"].max())
print("Cat range:", df["cat_id"].min(), df["cat_id"].max())



le_cat = LabelEncoder()
df["cat_id"] = le_cat.fit_transform(df["category"])
num_classes = len(le_cat.classes_)

#########################################
# 2. BUILD SEQUENCES
#########################################

#########################################
# 2. BUILD SEQUENCES (cat + hour)
#########################################

sequences = []
labels = []

for user_id, user_data in df.groupby("userid"):
    cats = user_data["cat_id"].values
    hours = user_data["hour"].values

    for i in range(len(cats) - SEQ_LEN):
        seq = np.stack([cats[i:i+SEQ_LEN], hours[i:i+SEQ_LEN]], axis=1)
        sequences.append(seq)
        labels.append(cats[i+SEQ_LEN])

sequences = np.array(sequences, dtype=np.int64)
labels = np.array(labels, dtype=np.int64)

print("Total samples:", len(sequences))
print("Sequence shape:", sequences.shape)  # (N, SEQ_LEN, 2)

#########################################
# CLASS WEIGHTS
#########################################

from collections import Counter

label_counts = Counter(labels)

class_weights = np.zeros(num_classes, dtype=np.float32)

for cls in range(num_classes):
    class_weights[cls] = 1.0 / label_counts.get(cls, 1)

# normalizar (opcional, mas recomendado)
class_weights = class_weights / class_weights.sum() * num_classes

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print("Class weights shape:", class_weights.shape, class_weights)




#########################################
# 3. DATASET
#########################################

class GowallaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = GowallaDataset(sequences, labels)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


#########################################
# 4. MODEL
#########################################

class NextPlaceModel(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim):
        super().__init__()

        self.cat_embedding = nn.Embedding(num_classes, embed_dim)
        self.hour_embedding = nn.Embedding(24, embed_dim)

        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, 2)
        cats = x[:, :, 0]
        hours = x[:, :, 1]

        cat_emb = self.cat_embedding(cats)
        hour_emb = self.hour_embedding(hours)

        x = torch.cat([cat_emb, hour_emb], dim=2)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

model = NextPlaceModel(num_classes, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#########################################
# 5. TRAINING
#########################################

#########################################
# 5. TRAINING + EVALUATION PER EPOCH
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

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TEST]"):
            X, y = X.to(DEVICE), y.to(DEVICE)

            outputs = model(X)
            loss = criterion(outputs, y)

            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total


    ###################
    # PRINT METRICS
    ###################
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")
    print("-" * 50)

