import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split

# ======================
# Configurações
# ======================
DATA_DIR = "/home/gustavo/Downloads/v1.0-mini"
CAM_CHANNEL = "CAM_FRONT"
LIDAR_CHANNEL = "LIDAR_TOP"
CLASS_LABELS = ['car', 'pedestrian', 'truck', 'animal', 'motorcycle', 'emergency']  # exemplo

# ======================
# Dataset personalizado
# ======================

def filter_dict(data, value, key):
    for k in data.keys():
        if data[k][key] == value:
            return data[k]

class NuScenesMiniDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_points=10000):
        self.root_dir = root_dir
        self.transform = transform
        self.max_points = max_points

        with open(os.path.join(root_dir, 'v1.0-mini/sample.json')) as f:
            self.samples = json.load(f)

        with open(os.path.join(root_dir, 'v1.0-mini/sample_data.json')) as f:
            self.sample_data = json.load(f)

        with open(os.path.join(root_dir, 'v1.0-mini/calibrated_sensor.json')) as f:
            self.calibrated_sensor = json.load(f)
            new = {}
            for i in self.calibrated_sensor:
                new[i["token"]] = i
            self.calibrated_sensor = new

        with open(os.path.join(root_dir, 'v1.0-mini/sensor.json')) as f:
            self.sensor = json.load(f)
            new = {}
            for i in self.sensor:
                new[i["token"]] = i
            self.sensor = new

        with open(os.path.join(root_dir, 'v1.0-mini/sample_annotation.json')) as f:
            self.sample_annotations = json.load(f)
            new = {}
            for i in self.sample_annotations:
                new[i["token"]] = i
            self.sample_annotations = new

        with open(os.path.join(root_dir, 'v1.0-mini/category.json')) as f:
            self.category = json.load(f)
            new = {}
            for i in self.category:
                new[i["token"]] = i
            self.category = new

        with open(os.path.join(root_dir, 'v1.0-mini/instance.json')) as f:
            self.instance = json.load(f)
            new = {}
            for i in self.instance:
                new[i["token"]] = i
            self.instance = new

        self.sample_data_by_sample_token = {}
        for s in self.sample_data:
            if s['sample_token'] not in self.sample_data_by_sample_token:
                self.sample_data_by_sample_token[s['sample_token']] = {}
            # channel = self.sensor['token'] == self.calibrated_sensor['sensor_token'] == s['calibrated_sensor_token']
            # print(self.calibrated_sensor[0])
            # f = filter_dict(self.calibrated_sensor, s['calibrated_sensor_token'], "token")
            f = self.calibrated_sensor[s['calibrated_sensor_token']]
            # f = filter_dict(self.sensor, f['sensor_token'], 'token')
            f = self.sensor[f['sensor_token']]
            f = f | s
            channel = f['channel']
            print(f)
            self.sample_data_by_sample_token[s['sample_token']][channel] = f
        self.annotations_by_sample_token = {}
        for key in self.sample_annotations:
            ann = self.sample_annotations[key]
            token = ann['sample_token']
            for channel in [LIDAR_CHANNEL, CAM_CHANNEL]:
                f =  self.sample_data_by_sample_token[token][channel]
            if token not in self.annotations_by_sample_token:
                self.annotations_by_sample_token[token] = {}
            if channel in [LIDAR_CHANNEL, CAM_CHANNEL]:
                self.annotations_by_sample_token[token] = ann

        self.data_pairs = []
        for s in self.samples:
            sample_token = s['token']
            channel = self.sample_data_by_sample_token[sample_token]
            # print(self.annotations_by_sample_token[sample_token])
            lidar_token = self.sample_data_by_sample_token[sample_token][LIDAR_CHANNEL]
            cam_token = self.sample_data_by_sample_token[sample_token][CAM_CHANNEL]
            print(cam_token)
            if sample_token in self.annotations_by_sample_token:
                anns = self.annotations_by_sample_token[sample_token]
                # labels = [ann['instance_token'].split('.')[0] for ann in anns]
                # label = next((CLASS_LABELS.index(l) for l in labels if l in CLASS_LABELS), None)
                instance_token = anns['instance_token']
                category_token = self.instance[instance_token]["category_token"]
                label = self.category[category_token]["name"]
                print("rotulo: ", label)
                for i, c in enumerate(CLASS_LABELS):
                    if c in CLASS_LABELS:
                        label = i
                    else:
                        label = None
                    if label is not None:
                        self.data_pairs.append((sample_token, label))
        print("aaa: ", len(self.data_pairs))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        sample_token, label = self.data_pairs[idx]

        # Carrega imagem
        cam_data = self.sample_data_by_sample_token[sample_token][CAM_CHANNEL]
        cam_path = os.path.join(self.root_dir, cam_data['filename'])
        image = Image.open(cam_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Carrega point cloud
        lidar_data = self.sample_data_by_sample_token[sample_token][LIDAR_CHANNEL]
        lidar_path = os.path.join(self.root_dir, lidar_data['filename'])
        pointcloud = np.fromfile(os.path.join(self.root_dir, lidar_path), dtype=np.float32).reshape(-1, 5)[:, :3]
        if pointcloud.shape[0] > self.max_points:
            indices = np.random.choice(pointcloud.shape[0], self.max_points, replace=False)
            pointcloud = pointcloud[indices]
        else:
            pad = self.max_points - pointcloud.shape[0]
            pointcloud = np.pad(pointcloud, ((0, pad), (0, 0)))

        return image, torch.tensor(pointcloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ======================
# Modelo multimodal
# ======================
class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.ReLU(),
            nn.Dropout3d(p=0.3),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.lidar_branch = nn.Sequential(
            nn.Linear(30000, 256),
            nn.ReLU(),
            nn.Dropout3d(p=0.3),
            nn.Linear(256, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ReLU(),
            nn.Dropout3d(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, lidar):
        img_feat = self.image_branch(image)
        lidar_feat = self.lidar_branch(lidar.view(lidar.size(0), -1))
        combined = torch.cat((img_feat, lidar_feat), dim=1)
        return self.classifier(combined)

# ======================
# Treinamento
# ======================
def train():
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    dataset = NuScenesMiniDataset(DATA_DIR, transform=transform)

    # 2. Dividindo em treino e teste (70% / 30%)
    print("Tamanho dataset: ", len(dataset))
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MultiModalClassifier(num_classes=len(CLASS_LABELS))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(40):
        model.train()
        total_loss = 0
        correct = 0
        y_true = []
        y_prob = []
        for img, pc, label in train_loader:
            img, pc, label = img.to(device), pc.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img, pc)
            loss = criterion(output, label)
            correct += (torch.max(output.data, 1)[1] == label).sum().item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        accuracy = correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
        print(f"Accuracy: {accuracy*100:.2f}%")

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for img, pc, label in test_loader:
                img, pc, label = img.to(device), pc.to(device), label.to(device)
                outputs = model(img, pc)
                preds = outputs.argmax(1)
                correct += (preds == label).sum().item()
                total += label.size(0)

        print(f"Test Accuracy = {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
