import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.model import Resnet_classifier
from utils.utils import mixup_criterion

# ------------------------ ハイパーパラメータ ------------------------
target_attr = "Male"
batch_size = 64
alpha = 0.4
epochs = 50
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4

# ------------------------ データ変換 ------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(178, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ------------------------ データセット読み込み ------------------------
train_data = datasets.CelebA(root="/data/furuya", split="train", target_type="attr", transform=transform_train, download=True)
valid_data = datasets.CelebA(root="/data/furuya", split="valid", target_type="attr", transform=transform_test)
test_data  = datasets.CelebA(root="/data/furuya", split="test",  target_type="attr", transform=transform_test)

attr_idx = train_data.attr_names.index(target_attr)

def attr_to_label(attr_tensor):
    return ((attr_tensor[attr_idx] + 1) // 2).long()

class CelebALabelDataset(torch.utils.data.Dataset):
    """CelebA の attr Tensor から 0/1 クラスに変換するラッパー"""
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, attr = self.base[i]
        label = attr_to_label(attr)
        return img, label

train_dataset = CelebALabelDataset(train_data)
valid_dataset = CelebALabelDataset(valid_data)
test_dataset  = CelebALabelDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

# ------------------------ モデル・最適化設定 ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Resnet_classifier().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

# ------------------------ ログ記録用 ------------------------
train_losses, val_losses, val_accuracies = [], [], []
best_acc = 0

# ------------------------ 学習ループ関数 ------------------------
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Mixupの混合係数をベータ分布から取得
        lam = float(np.random.beta(alpha, alpha))

        logits, t_a, t_b = model(imgs, labels, lam, mode='train')

        # Mixup損失の計算
        loss = mixup_criterion(criterion, logits, t_a, t_b, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
    sys.stdout.flush()
    return avg_loss

def validate(epoch):
    global best_acc
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, None, None, mode='test')
            val_loss += criterion(logits, labels).item() * imgs.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()

    val_loss /= len(valid_loader.dataset)
    val_acc = correct / len(valid_loader.dataset) * 100
    print(f"         Val   Loss = {val_loss:.4f}, Acc = {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best.pth")
        print(f"Saved new best model with acc = {best_acc:.2f}%")
    sys.stdout.flush()

    return val_loss, val_acc

def test():
    model.load_state_dict(torch.load("best.pth"))
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs, labels, 1, mode='test')
            loss = criterion(outputs, labels)
            test_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100
    print(f"------> Test loss = {test_loss:.4f}, Test Accuracy = {acc:.2f}%")
    sys.stdout.flush()

# ------------------------ メイントレーニングループ ------------------------
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(epoch)
    val_loss, val_acc = validate(epoch)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step()

# ------------------------ テストと学習曲線出力 ------------------------
test()

# 学習曲線のプロット
epochs_range = range(1, epochs + 1)
plt.figure()
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(epochs_range, val_accuracies, label="Validation Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig("accuracy_curve.png")