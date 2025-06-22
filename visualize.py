import torch
from torchvision import transforms, datasets
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from models.recon import Mixup_VAE

# ──────────────── 設定 ────────────────
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim         = 128
n_steps       = 11            # λ_user のステップ数（0.0,0.1,…,1.0）
out_dir       = "mixup_interp"
num_patterns  = 10            # 試すパターン数
os.makedirs(out_dir, exist_ok=True)

# ──────────────── モデル読み込み ────────────────
model = Mixup_VAE(z_dim=z_dim).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ──────────────── データセット準備 ────────────────
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])
val_ds = datasets.CelebA(root="/data/furuya",
                         split="valid",
                         target_type="attr",
                         transform=transform,
                         download=False)
attr_idx = val_ds.attr_names.index("Male")

# ──────────────── 画像ペアの取得関数 ────────────────
def sample_pair(dataset):
    img_f = img_m = None
    indices = torch.randperm(len(dataset))
    for idx in indices:
        img, attr = dataset[idx]
        label = ((attr[attr_idx] + 1)//2).item()  # 0: Female, 1: Male
        if label == 0 and img_f is None:
            img_f = img
        if label == 1 and img_m is None:
            img_m = img
        if img_f is not None and img_m is not None:
            break
    return img_f, img_m

# ──────────────── メイン処理 ────────────────
for p in range(num_patterns):
    pattern_dir = os.path.join(out_dir, f"pattern_{p:02d}")
    os.makedirs(pattern_dir, exist_ok=True)

    # サンプリング
    img_f, img_m = sample_pair(val_ds)
    x1 = img_f.unsqueeze(0).to(device)  # Female
    x2 = img_m.unsqueeze(0).to(device)  # Male

    # 入力画像の保存
    save_image(x1, os.path.join(pattern_dir, "input_female.png"), normalize=True, value_range=(-1,1))
    save_image(x2, os.path.join(pattern_dir, "input_male.png"),   normalize=True, value_range=(-1,1))

    # ミックスアップ推論
    lambdas_user = torch.linspace(0, 1, steps=n_steps, device=device)
    semantic_outs = []
    pixel_outs    = []
    for i, lam_u in enumerate(lambdas_user):
        # Semantic Mixup (VAE)
        lam_model = 1.0 - lam_u.item()
        with torch.no_grad():
            recon = model.generate_mixup(x1, x2, lam_model)
        semantic_outs.append(recon.squeeze(0))
        sem_path = os.path.join(pattern_dir, f"sem_step_{i:02d}_lam{lam_u.item():.2f}.png")
        save_image(recon, sem_path, normalize=True, value_range=(-1,1))

        # Pixel-level Mixup (修正: 左→右が女性→男性になるように重みを逆転)
        pixel = (1.0 - lam_u) * x1 + lam_u * x2
        pixel_outs.append(pixel.squeeze(0))
        pix_path = os.path.join(pattern_dir, f"pix_step_{i:02d}_lam{lam_u.item():.2f}.png")
        save_image(pixel, pix_path, normalize=True, value_range=(-1,1))

    # モンタージュ作成
    semantic_grid = make_grid(semantic_outs, nrow=n_steps, normalize=True, value_range=(-1,1))
    pixel_grid    = make_grid(pixel_outs,    nrow=n_steps, normalize=True, value_range=(-1,1))
    # 上段: Pixel, 下段: Semantic
    comparison = torch.cat([pixel_grid, semantic_grid], dim=1)
    save_image(comparison, os.path.join(pattern_dir, "comparison_montage.png"))

    # Matplotlib での保存
    plt.figure(figsize=(n_steps, 4))  # 高さ2倍
    plt.imshow(comparison.permute(1,2,0).cpu())
    plt.axis('off')
    plt.title(f"Pattern {p:02d}: Pixel (上) vs Semantic (下)")
    plt.savefig(os.path.join(pattern_dir, "comparison_output.png"))
    plt.close()

    print(f"Finished pattern {p:02d}, saved to {pattern_dir}")