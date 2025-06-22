import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib
from tqdm import tqdm

from models.recon import Mixup_VAE

# ------------------------ ハイパーパラメータ ------------------------
batch_size = 64
latent_dim = 128

# ------------------------ データ変換 ------------------------
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# ------------------------ データセット読み込み ------------------------
train_ds = datasets.CelebA(root="/data/furuya", split="train", target_type="attr",
                           transform=transform, download=False)
valid_ds = datasets.CelebA(root="/data/furuya", split="valid", target_type="attr",
                           transform=transform, download=False)
test_ds  = datasets.CelebA(root="/data/furuya", split="test",  target_type="attr",
                           transform=transform, download=False)
attr_idx = train_ds.attr_names.index("Male")

class LatentDataset(torch.utils.data.Dataset):
    """VAEエンコーダを使って潜在表現を事前抽出"""
    def __init__(self, base_ds, vae, device):
        self.base = base_ds
        self.vae = vae
        self.device = device

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, attr = self.base[idx]
        x = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            z, mu, logvar = self.vae.encoder(x)
        # flatten to 1D
        z = z.view(z.size(0), -1).squeeze(0).cpu().numpy()
        label = ((attr[attr_idx] + 1)//2).item()
        return z, label

# ──────────────── モデル読み込み ────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = Mixup_VAE(z_dim=latent_dim).to(device)
vae.load_state_dict(torch.load("best_model.pth", map_location=device))
vae.eval()

# ──────────────── 潜在データ生成 ────────────────
def extract_latents(dataset, name):
    print(f"Encoding {name} set...")
    X, y = [], []
    for z_vec, label in tqdm(LatentDataset(dataset, vae, device)):
        X.append(z_vec)
        y.append(label)
    return np.stack(X), np.array(y)

X_train, y_train = extract_latents(train_ds, "train")
X_valid, y_valid = extract_latents(valid_ds, "valid")
X_test,  y_test  = extract_latents(test_ds,  "test")

# ------------------------ 特徴量スケーリング ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test)

# ------------------------ SVM学習 ------------------------
svm = SVC(kernel='linear', probability=True, verbose=True)
svm.fit(X_train_scaled, y_train)

# 保存
joblib.dump(svm, 'svm_latent_male_female.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ------------------------ 評価 ------------------------
for split, X, y in [('Train', X_train_scaled, y_train),
                    ('Valid', X_valid_scaled, y_valid),
                    ('Test',  X_test_scaled,  y_test)]:
    y_pred = svm.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"{split} Accuracy: {acc*100:.2f}%")
    print(classification_report(y, y_pred, target_names=['Female','Male']))
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix ({split}):\n", cm)

print("SVM training and evaluation completed.")
