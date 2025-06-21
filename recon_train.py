import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

from models.recon import Mixup_VAE   
from tqdm import tqdm

# ------------------------ ハイパーパラメータ ------------------------
batch_size   = 64
alpha        = 0.4     # Mixup β分布パラメータ
epochs       = 50
lr           = 1e-3
momentum     = 0.9
weight_decay = 5e-4
beta         = 1.0     # 再構成損失の重み
gamma        = 1e-3    # KL 散逸損失の重み
sample_dir = "val_outputs"
os.makedirs(sample_dir, exist_ok=True)
target_attr = "Male"

# ------------------------ データ変換 ------------------------
transform = transforms.Compose([
    transforms.RandomCrop(178, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# CelebA のラベルは使わないので、target_type=None でもOK
train_data = datasets.CelebA(root="/data/furuya", split="train",
                            target_type="attr", transform=transform, download=True)
valid_data = datasets.CelebA(root="/data/furuya", split="valid",
                            target_type="attr", transform=transform)
test_data  = datasets.CelebA(root="/data/furuya", split="test",
                            target_type="attr", transform=transform)

attr_idx = train_data.attr_names.index(target_attr)

# Dataset から画像だけ取り出す小ワークアラウンド
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
model  = Mixup_VAE(z_dim=128).to(device)

# オプティマイザは encoder + decoder のみを更新
opt_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
optimizer = optim.SGD(opt_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

# 損失ログ用
train_recon_losses = []
train_kl_losses    = []
val_orig_losses    = []
val_mix_losses     = []
val_kl_losses      = []
best_val_loss = float('inf')
best_model_path = "best_model.pth"

# ------------------------ 学習ループ ------------------------
for epoch in range(1, epochs+1):
    model.train()
    tot_recon, tot_kl, tot_samples = 0.0, 0.0, 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        B = imgs.size(0)

        lam = float(np.random.beta(alpha, alpha))

        # forward
        recon_orig, recon_mix, x, x_mix, mu, logvar = model(
            imgs, labels, lam, mode='train'
        )

        # 1) オリジナル再構成損失
        loss_orig  = F.mse_loss(recon_orig, imgs)

        # 2) semantic Mixup 再構成損失
        loss_mix   = F.mse_loss(recon_mix, x_mix)

        # 3) KL 散逸損失
        loss_kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 合成損失
        loss = beta * (loss_orig + loss_mix) + gamma * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_recon += (loss_orig.item() + loss_mix.item()) * B
        tot_kl += loss_kl.item() * B
        tot_samples += B
        pbar.set_postfix({
            "loss_recon": loss.item(),
            "loss_kl": loss_kl.item()
        })

    train_recon_losses.append(tot_recon / tot_samples)
    train_kl_losses.append(   tot_kl    / tot_samples)

    # ------------------------ バリデーション ------------------------
    model.eval()
    v_orig, v_mix, v_kl, v_n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for imgs, _ in valid_loader:
            B = imgs.size(0)
            imgs = imgs.to(device)

            # ラップ：train でも lam=1 ならオリジナル再構成と Mixup 再構成が得られる
            recon_orig, recon_mix, x, x_mix, mu, logvar = model(
                imgs,
                torch.zeros(B, dtype=torch.long, device=device),
                lam=1.0,
                mode='train'
            )

            loss_orig = F.mse_loss(recon_orig, imgs)
            loss_mix  = F.mse_loss(recon_mix, x_mix)
            loss_kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            v_orig += loss_orig.item() * B
            v_mix  += loss_mix.item()  * B
            v_kl   += loss_kl.item()   * B
            v_n   += B

    # 平均を記録
    val_orig_losses.append(v_orig / v_n)
    val_mix_losses.append( v_mix  / v_n)
    val_kl_losses.append(  v_kl   / v_n)

    print(f"[Val] Recon(orig) {val_orig_losses[-1]:.4f}  "
          f"Recon(mix)  {val_mix_losses[-1]:.4f}  "
          f"KL          {val_kl_losses[-1]:.4f}")

    # ------------------------ サンプル画像の生成・保存 ------------------------
    imgs_sample, _ = next(iter(valid_loader))
    imgs_sample = imgs_sample[:5].to(device)

    # オリジナル再構成
    with torch.no_grad():
        recon_o = model(imgs_sample, mode='test')   # test モードでオリジナル再構成だけ返すようモデルを調整
    for i in range(5):
        save_image(imgs_sample[i],
                   f"{sample_dir}/orig_input_{i}.png", normalize=True)
        save_image(recon_o[i],
                   f"{sample_dir}/orig_recon_{i}.png", normalize=True)

    # semantic Mixup 再構成
    lam_mix = 0.5
    with torch.no_grad():
        _, recon_m, x, x_mix, _, _ = model(
            imgs_sample,
            torch.zeros(5, dtype=torch.long, device=device),
            lam_mix,
            mode='train'
        )
    for i in range(5):
        save_image(x_mix[i],
                   f"{sample_dir}/mix_target_{i}.png", normalize=True)
        save_image(recon_m[i],
                   f"{sample_dir}/mix_recon_{i}.png", normalize=True)

    if epoch == 1:
        best_val_loss = float('inf')
        best_model_path = "best_model.pth"

    current_val_loss = val_orig_losses[-1] + val_mix_losses[-1]
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with val_loss = {best_val_loss:.4f}")

# ------------------------ テスト ------------------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
t_orig, t_mix, t_kl, t_n = 0.0, 0.0, 0.0, 0

with torch.no_grad():
    for imgs, _ in test_loader:
        B = imgs.size(0)
        imgs = imgs.to(device)

        # 同じく lam=1 の train モードで両方取得
        recon_orig, recon_mix, x, x_mix, mu, logvar = model(
            imgs,
            torch.zeros(B, dtype=torch.long, device=device),
            lam=1.0,
            mode='train'
        )

        t_orig += F.mse_loss(recon_orig, imgs).item() * B
        t_mix  += F.mse_loss(recon_mix, x_mix).item() * B
        t_kl   += (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())).item() * B
        t_n   += B

test_orig_loss = t_orig / t_n
test_mix_loss  = t_mix  / t_n
test_kl_loss   = t_kl   / t_n
print(f"[Test] Recon(orig) {test_orig_loss:.4f}, "
      f"Recon(mix) {test_mix_loss:.4f}, KL {test_kl_loss:.4f}")

# テストサンプル保存（バリデーションと同様）
imgs_sample, _ = next(iter(test_loader))
imgs_sample = imgs_sample[:5].to(device)

with torch.no_grad():
    recon_o = model(imgs_sample, mode='test')
for i in range(5):
    save_image(imgs_sample[i],
               f"{sample_dir}/test_input_{i}.png", normalize=True)
    save_image(recon_o[i],
               f"{sample_dir}/test_orig_recon_{i}.png", normalize=True)

with torch.no_grad():
    _, recon_m, x, x_mix, _, _ = model(
        imgs_sample,
        torch.zeros(5, dtype=torch.long, device=device),
        lam_mix,
        mode='train'
    )
for i in range(5):
    save_image(x_mix[i],
               f"{sample_dir}/test_mix_target_{i}.png", normalize=True)
    save_image(recon_m[i],
               f"{sample_dir}/test_mix_recon_{i}.png", normalize=True)

# ------------------------ 学習曲線プロット ------------------------
epochs_range = range(1, epochs+1)
plt.figure()
plt.plot(epochs_range, train_recon_losses, label="Train Recon")
plt.plot(epochs_range, val_orig_losses,   label="Val Recon(orig)")
plt.plot(epochs_range, val_mix_losses,    label="Val Recon(mix)")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Reconstruction Loss")
plt.legend(); plt.grid()
plt.savefig("recon_loss.png")

plt.figure()
plt.plot(epochs_range, train_kl_losses, label="Train KL")
plt.plot(epochs_range, val_kl_losses,   label="Val KL")
plt.xlabel("Epoch"); plt.ylabel("KL Loss")
plt.title("KL Divergence Loss")
plt.legend(); plt.grid()
plt.savefig("kl_loss.png")