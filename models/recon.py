import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models.utils import mixup_aligned, align_features, mixup_aligned_recon

# --- シンプルなCNNエンコーダ ---
class VAEEncoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        # 入力 3x128x128 -> 64x64x64 -> 128x32x32 -> 256x16x16
        self.conv1 = nn.Conv2d(3, 64,  kernel_size=4, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128,  kernel_size=4, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        # z_dim 次元の特徴マップに変換
        self.conv_mu     = nn.Conv2d(256, z_dim,      kernel_size=1)
        self.conv_logvar = nn.Conv2d(256, z_dim,      kernel_size=1)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))   # -> 64x64x64
        h = F.relu(self.bn2(self.conv2(h)))   # -> 128x32x32
        h = F.relu(self.bn3(self.conv3(h)))   # -> 256x16x16
        mu     = self.conv_mu(h)              # -> z_dimx16x16
        logvar = self.conv_logvar(h)          # -> z_dimx16x16
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                     # reparameterization trick
        return z, mu, logvar  # z: (B, z_dim, 16, 16)

# --- シンプルなCNNデコーダ ---
class VAEDecoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        # z_dimx16x16 -> 256x32x32 -> 128x64x64 -> 64x128x128 -> 3x128x128
        self.deconv1 = nn.ConvTranspose2d(z_dim,256,kernel_size=4,stride=2,padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128,64, kernel_size=4,stride=2,padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        # 空間解像度を維持して最終RGBに
        self.conv4   = nn.Conv2d(64,3,kernel_size=3,padding=1)

    def forward(self, z):
        h = F.relu(self.bn1(self.deconv1(z)))  # -> 256x32x32
        h = F.relu(self.bn2(self.deconv2(h)))  # -> 128x64x64
        h = F.relu(self.bn3(self.deconv3(h)))  # -> 64x128x128
        x_recon = torch.tanh(self.conv4(h))    # -> 3x128x128
        return x_recon

# --- VAE Mixup 分類 + 再構成モデル ---
class Mixup_VAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.encoder    = VAEEncoder(z_dim)
        self.decoder    = VAEDecoder(z_dim)

    def forward(self, x, targets=None, lam=None, mode='train'):
        """
        x:       入力画像 (B,3,128,128)
        targets: ラベル (B,), train時必須
        lam:     mixup係数 (float), train時必須
        mode:    'train' or 'test'
        """
        # 1) VAE encoding
        z, mu, logvar = self.encoder(x)  # -> (B, z_dim, 16, 16)

        if mode == 'test':
            return self.decoder(z)

        # 2) 元画像再構成 (オリジナル再構成)
        recon_orig = self.decoder(z)

        # 3) Mixup-aligned
        z_mix, t_a, t_b, indices = mixup_aligned_recon(z, targets, lam, return_index=True)
        z_shuf, _, _ = self.encoder(x[indices])

        # 4) Mixup 再構成
        recon_mix = self.decoder(z_mix)

        # 5) 意味的合成画像をターゲットとして作成
        x_a = recon_orig
        x_b = self.decoder(z_shuf)  
        x_semantic_mix = lam * x_a + (1 - lam) * x_b


        # 7) 必要なものを返す
        #    logits: 分類ログit
        #    t_a, t_b: Mixup用ラベルペア
        #    recon: セマンティックMixup潜在からの再構成画像
        #    x_semantic_mix: 再構成ターゲット画像
        #    mu, logvar: VAEのKL項計算用
        return (recon_orig, recon_mix,
                    x, x_semantic_mix,
                    mu, logvar)

    def generate_mixup(self, x1, x2, lam):
        # 1) encode
        z1, _, _ = self.encoder(x1)   # female latent
        z2, _, _ = self.encoder(x2)   # male latent

        # 端点はアライメントなしで純粋に出力
        if lam >= 0.999:
            return self.decoder(z1)   # 100% female
        if lam <= 0.001:
            return self.decoder(z2)   # 100% male

        # # 2) align (Sinkhorn ベース)
        # z2_aligned = align_features(z1, z2)

        # 3) semantic mix
        z_mix = lam * z1 + (1 - lam) * z2

        # 4) decode
        return self.decoder(z_mix)