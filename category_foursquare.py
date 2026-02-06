import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

FOURSQUARE_HIGH_LEVEL = {
    "Food": [
        "Restaurant", "Fast Food Restaurant", "Sushi Restaurant",
        "Italian Restaurant", "Brazilian Restaurant", "Mexican Restaurant",
        "Pizza Place", "Burger Joint", "Coffee Shop", "Bakery",
        "Sandwich Place", "Steakhouse", "Diner", "Café"
    ],
    "Nightlife Spot": [
        "Bar", "Pub", "Nightclub", "Lounge"
    ],
    "Travel & Transport": [
        "Gas Station", "Airport", "Train Station",
        "Bus Station", "Subway", "Hotel"
    ],
    "Shop & Service": [
        "Grocery Store", "Supermarket", "Mall",
        "Pharmacy", "Convenience Store", "Clothing Store"
    ],
    "Outdoors & Recreation": [
        "Park", "Playground", "Gym",
        "Trail", "Plaza", "Stadium"
    ],
    "College & University": [
        "University", "College", "High School"
    ],
    "Arts & Entertainment": [
        "Movie Theater", "Museum", "Music Venue"
    ],
    "Professional & Other Places": [
        "Office", "Coworking Space"
    ],
}

def map_to_high_level_category(cat):
    for high_level, fine_list in FOURSQUARE_HIGH_LEVEL.items():
        if cat in fine_list:
            return high_level
    return "Other"


def load_foursquare_us(csv_path):
    """
    Loader específico para o dataset Foursquare (TIST2015)
    já filtrado para United States.
    """
    df = pd.read_csv(csv_path)

    # usa utc_time (já timezone-aware)
    df["utc_time"] = pd.to_datetime(df["utc_time"])

    # ordena por usuário e tempo (obrigatório para sequências)
    df = df.sort_values(by=["user_id", "utc_time"])

    return df

def print_class_distribution(df, top_k=10):
    """
    Exibe proporção das classes (categorias).
    """
    counts = df["category_id"].value_counts()
    total = counts.sum()

    print("\n📊 Distribuição de categorias")
    print(f"Total de amostras: {total}")
    print(f"Total de categorias: {len(counts)}\n")

    print("Top categorias mais frequentes:")
    for cat, cnt in counts.head(top_k).items():
        print(f"  Categoria {cat:4d}: {cnt:8d} ({cnt/total:.2%})")

    print("\nCategorias menos frequentes:")
    for cat, cnt in counts.tail(top_k).items():
        print(f"  Categoria {cat:4d}: {cnt:8d} ({cnt/total:.4%})")

def split_users(user_sequences, train_ratio=0.8, seed=42):
    """
    Faz split por USUÁRIO (sem vazamento).
    user_sequences: dict[user_id] -> list of sequences
    """
    rng = np.random.default_rng(seed)

    users = list(user_sequences.keys())
    rng.shuffle(users)

    split = int(train_ratio * len(users))
    train_users = set(users[:split])
    val_users = set(users[split:])

    train_seqs = []
    val_seqs = []

    for u in train_users:
        train_seqs.extend(user_sequences[u])

    for u in val_users:
        val_seqs.extend(user_sequences[u])

    return train_seqs, val_seqs


def build_sequences_per_user_with_time(
    df,
    seq_len=3,
    min_checkins=10
):
    """
    Retorna:
    user_id -> lista de (cat_seq, hour_seq, day_seq, target_cat)
    """
    user_sequences = {}

    for user_id, group in df.groupby("user_id"):
        cats = group["category_id"].values
        hours = group["local_hour"].values
        days = group["local_dayofweek"].values

        if len(cats) < min_checkins:
            continue

        seqs = []
        for i in range(len(cats) - seq_len):
            seqs.append((
                cats[i : i + seq_len],
                hours[i : i + seq_len],
                days[i : i + seq_len],
                cats[i + seq_len]
            ))

        if seqs:
            user_sequences[user_id] = seqs

    return user_sequences

def encode_categories(df):
    """
    Converte categorias de ALTO NÍVEL em IDs inteiros.
    """
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["category_high"])

    num_categories = len(le.classes_)
    print(f"📊 Total de categorias (alto nível): {num_categories}")
    print("Categorias:", list(le.classes_))

    return df, le, num_categories

def filter_top_k_categories(df, k=10):
    """
    Mantém apenas as top-k categorias mais frequentes.
    """
    counts = df["category"].value_counts()
    top_k_cats = counts.head(k).index

    df = df[df["category"].isin(top_k_cats)].copy()

    print(f"🔥 Treinando apenas TOP-{k} categorias")
    print("Categorias mantidas:")
    for cat, cnt in counts.head(k).items():
        print(f"  {cat:35s} {cnt}")

    return df



class NextCategoryWithTimeDataset(Dataset):
    def __init__(self, sequences):
        self.cat = torch.tensor([s[0] for s in sequences], dtype=torch.long)
        self.hour = torch.tensor([s[1] for s in sequences], dtype=torch.long)
        self.day = torch.tensor([s[2] for s in sequences], dtype=torch.long)
        self.y = torch.tensor([s[3] for s in sequences], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.cat[idx], self.hour[idx], self.day[idx], self.y[idx]

def build_dataloaders_us_with_time(
    csv_path,
    seq_len=3,
    batch_size=128,
    min_checkins=10
):
    df = load_foursquare_us(csv_path)

    # 🔥 MAPEAMENTO HIERÁRQUICO OFICIAL
    df["category_high"] = df["category"].apply(map_to_high_level_category)

    # (opcional, mas recomendado)
    df = df[df["category_high"] != "Other"]

    df, label_encoder, num_categories = encode_categories(df)

    print_class_distribution(df)

    df, label_encoder, num_categories = encode_categories(df)

    print_class_distribution(df)

    user_sequences = build_sequences_per_user_with_time(
        df,
        seq_len=seq_len,
        min_checkins=min_checkins
    )

    print(f"\n👤 Usuários válidos: {len(user_sequences)}")

    train_seq, val_seq = split_users(user_sequences)

    print(f"🧩 Treino: {len(train_seq)} | Val: {len(val_seq)}")

    train_ds = NextCategoryWithTimeDataset(train_seq)
    val_ds = NextCategoryWithTimeDataset(val_seq)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, num_categories

import torch.nn as nn

class NextCategoryLSTMWithTime(nn.Module):
    def __init__(
        self,
        num_categories,
        cat_emb_dim=3,
        hour_emb_dim=3,
        day_emb_dim=2,
        hidden_dim=64,
        num_layers=1,
        dropout=0.5
    ):
        super().__init__()

        self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)
        self.hour_emb = nn.Embedding(24, hour_emb_dim)
        self.day_emb = nn.Embedding(7, day_emb_dim)

        input_dim = cat_emb_dim + hour_emb_dim + day_emb_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, num_categories)

    def forward(self, cat_seq, hour_seq, day_seq):
        cat_e = self.cat_emb(cat_seq)
        hour_e = self.hour_emb(hour_seq)
        day_e = self.day_emb(day_seq)

        x = torch.cat([cat_e, hour_e, day_e], dim=-1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for cat, hour, day, y in loader:
        cat = cat.to(device)
        hour = hour.to(device)
        day = day.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(cat, hour, day)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for cat, hour, day, y in loader:
        cat = cat.to(device)
        hour = hour.to(device)
        day = day.to(device)
        y = y.to(device)

        logits = model(cat, hour, day)
        loss = criterion(logits, y)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    return total_loss / len(loader), acc

if __name__ == "__main__":
    CSV_US = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/dataset_TIST2015/dataset_tist2015_United_States.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, num_categories = build_dataloaders_us_with_time(
        CSV_US,
        seq_len=20,
        batch_size=256,
        min_checkins=60
    )

    model = NextCategoryLSTMWithTime(num_categories).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 50

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
