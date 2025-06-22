import torch
from torchvision import transforms, datasets
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from models.recon import Mixup_VAE

# ──────────────── 設定 ────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim    = 128
n_steps  = 11           # λ_user のステップ数（0.0,0.1,…,1.0）
out_dir  = "mixup_interp"
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

# ──────────────── 画像ペアの取得 ────────────────
def sample_pair(dataset):
    img_f = img_m = None
    for img, attr in dataset:
        label = ((attr[attr_idx] + 1)//2).item()  # 0: Female, 1: Male
        if label == 0 and img_f is None:
            img_f = img
        if label == 1 and img_m is None:
            img_m = img
        if img_f is not None and img_m is not None:
            break
    return img_f, img_m

img_f, img_m = sample_pair(val_ds)
x1 = img_f.unsqueeze(0).to(device)   # Female
x2 = img_m.unsqueeze(0).to(device)   # Male

# ──────────────── 入力画像の保存＆表示 ────────────────
female_path = os.path.join(out_dir, "input_female.png")
male_path   = os.path.join(out_dir, "input_male.png")
save_image(x1, female_path, normalize=True, value_range=(-1,1))
save_image(x2, male_path,   normalize=True, value_range=(-1,1))
print(f"Mixup inputs → Female: {female_path}, Male: {male_path}")

# ──────────────── ミックスアップ推論 ────────────────
lambdas_user = torch.linspace(0, 1, steps=n_steps, device=device)
outs = []

for i, lam_u in enumerate(lambdas_user):
    lam_model = 1.0 - lam_u.item()   # generate_mixup の λ
    with torch.no_grad():
        recon = model.generate_mixup(x1, x2, lam_model)  # lam_model=1→female, 0→male
    outs.append(recon.squeeze(0))

    # デバッグ用に各ステップを別ファイルにも出力
    step_path = os.path.join(out_dir, f"step_{i:02d}_lam{lam_u.item():.2f}.png")
    save_image(recon, step_path, normalize=True, value_range=(-1,1))
    print(f" Step {i:02d}: lam_user={lam_u.item():.2f} → saved {step_path}")

# ──────────────── モンタージュ作成・保存 ────────────────
grid = make_grid(outs, nrow=n_steps, normalize=True, value_range=(-1,1))
montage_path = os.path.join(out_dir, "female_to_male_montage.png")
save_image(grid, montage_path)
print(f" Montage saved → {montage_path}")

# ──────────────── （任意）Matplotlib での保存 ────────────────
plt.figure(figsize=(n_steps, 2))
plt.imshow(grid.permute(1,2,0).cpu())
plt.axis("off")
plt.title("Female → Male Mixup (stepwise λ_user)")
plt.savefig(os.path.join(out_dir, "output.png"))