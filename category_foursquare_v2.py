import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class NextCategoryWithTimeDataset(Dataset):
    def __init__(self, hf_dataset):
        self.cat = torch.tensor(hf_dataset["cat_seq"], dtype=torch.long)
        self.hour = torch.tensor(hf_dataset["hour_seq"], dtype=torch.long)
        self.day = torch.tensor(hf_dataset["day_seq"], dtype=torch.long)
        self.y = torch.tensor(hf_dataset["target"], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.cat[idx], self.hour[idx], self.day[idx], self.y[idx]

def build_dataloaders_from_huggingface(
    repo_id,
    batch_size=128
):
    print(f"📥 Carregando dataset do Hugging Face: {repo_id}")

    dataset = load_dataset(repo_id)

    train_ds = NextCategoryWithTimeDataset(dataset["train"])
    val_ds = NextCategoryWithTimeDataset(dataset["validation"])

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

    # 🔑 inferência segura do número de classes
    num_categories = int(
        max(
            dataset["train"]["target"] +
            dataset["validation"]["target"]
        ) + 1
    )

    print(f"📊 Número de categorias: {num_categories}")

    return train_loader, val_loader, num_categories

import torch.nn as nn

class NextCategoryLSTMWithTime(nn.Module):
    def __init__(
        self,
        num_categories,
        cat_emb_dim=3,
        hour_emb_dim=3,
        day_emb_dim=2,
        hidden_dim=8,
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

    HF_REPO = "claudiogsc/foursquare-us-sequences-highlevel"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, num_categories = (
        build_dataloaders_from_huggingface(
            HF_REPO,
            batch_size=256
        )
    )

    print("tamanho ", len(train_loader.dataset))
    exit()

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

