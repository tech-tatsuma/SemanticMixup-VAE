import torch

import numpy as np
import random

from utils.utils import SinkhornDistance

# 通常のMixup処理：特徴を線形結合
def mixup_process(out, y, lam):
    indices = np.random.permutation(out.size(0)) # バッチ内のランダムな順列を生成
    out = out*lam + out[indices]*(1-lam) # 特徴を線形混合
    y_a, y_b = y, y[indices] # 対応するラベルも取得
    return out,  y_a, y_b

def mixup_aligned_recon(out, y, lam, return_index=False):
    B = out.size(0)
    # --- ここを変更 ---
    # 各サンプル i に対して y[i] != y[j] を満たす j をランダムに選ぶ
    indices = []
    for i in range(B):
        # 異なるラベルを持つサンプルの一覧
        opp = torch.nonzero(y != y[i], as_tuple=False).view(-1)
        # もし全員同じラベルなら普通に perm でfallback
        if len(opp)==0:
            j = random.randrange(B)
        else:
            j = opp[random.randrange(len(opp))].item()
        indices.append(j)
    indices = torch.tensor(indices, device=out.device)

    # 以下は従来どおりSinkhorn＋アライメントMix
    feat1 = out.view(B, out.size(1), -1)
    feat2 = out[indices].view(B, out.size(1), -1)
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    P = sinkhorn(feat1.permute(0,2,1), feat2.permute(0,2,1)).detach()
    P = P * (out.size(2)*out.size(3))

    # ランダムに align_mix 方向を決定
    if random.randint(0,1)==0:
        f1 = torch.matmul(feat2, P.permute(0,2,1)).view_as(out)
        final = out*lam + f1*(1-lam)
    else:
        f2 = torch.matmul(feat1, P).view_as(out)
        final = f2*lam + out[indices]*(1-lam)

    y_a, y_b = y, y[indices]
    return (final, y_a, y_b, indices) if return_index else (final, y_a, y_b)

# AlignMix処理（特徴のアライメントをSinkhorn距離で計算）
def mixup_aligned(out, y, lam, return_index=False):
    """
    引数:
        out: 特徴マップ (Tensor) - shape = (B, C, H, W)
        y:   正解ラベル (Tensor) - shape = (B,)
        lam: Mixup係数 (float) - 0〜1の間の値

    処理内容:
        - バッチ内の他のサンプルとランダムにペアを組む
        - Sinkhorn距離を用いて空間的特徴の対応を取得
        - 対応関係に基づいて特徴を位置合わせ（Align）
        - 対応後の特徴に対してMixup（線形補間）を実施

    戻り値:
        final: アライメント後にMixupされた特徴 (Tensor)
        y_a, y_b: Mixup対象のラベルペア (Tensor, Tensor)
    """

    # ランダムにバッチ内のインデックスを並べ替え（対応ペアを作成）
    indices = np.random.permutation(out.size(0))

    # 特徴マップを3次元（B, C, H*W）に変形
    feat1 = out.view(out.shape[0], out.shape[1], -1) # shape: (B, C, HW)
    feat2 = out[indices].view(out.shape[0], out.shape[1], -1) # shape: (B, C, HW)
    
    # Sinkhorn距離を計算し、対応行列 P を得る（shape: B x HW x HW）
    # 転置して (B, HW, C) にし、特徴空間に対してSinkhorn距離を計算
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    P = sinkhorn(feat1.permute(0,2,1), feat2.permute(0,2,1)).detach() 
    
    # 正規化係数を元の空間にスケーリング
    P = P*(out.size(2)*out.size(3))

    # アライメントMixの方向をランダムに決定（対称性の考慮）
    align_mix = random.randint(0,1)
   
    if (align_mix == 0):
        # feat2をfeat1にアライメントしてMixup
        f1 = torch.matmul(feat2, P.permute(0,2,1).cuda()).view(out.shape) 
        final = feat1.view(out.shape)*lam + f1*(1-lam)

    elif (align_mix == 1):
        # feat1をfeat2にアライメントしてMixup
        f2 = torch.matmul(feat1, P.cuda()).view(out.shape).cuda()
        final = f2*lam + feat2.view(out.shape)*(1-lam)

    # Mixup対象のラベルペアを返す
    y_a, y_b = y,y[indices]

    if return_index:
        return final, y_a, y_b, indices
    else:
        return final, y_a, y_b

def align_features(feat1: torch.Tensor,
                   feat2: torch.Tensor) -> torch.Tensor:
    """
    セマンティック特徴マップ feat2 を feat1 に空間的に位置合わせします。

    引数:
        feat1: Tensor, shape = (B, C, H, W)
        feat2: Tensor, shape = (B, C, H, W)
    戻り値:
        aligned_feat2: Tensor, shape = (B, C, H, W)
            feat2 を feat1 に対応づけた後の特徴マップ
    """
    B, C, H, W = feat1.shape
    HW = H * W
    device = feat1.device

    # (B, C, HW) に変形
    f1 = feat1.view(B, C, HW)
    f2 = feat2.view(B, C, HW)

    # SinkhornDistance で対応行列 P を計算
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    # 入力は (B, HW, C) になるように転置
    P = sinkhorn(f1.permute(0, 2, 1), f2.permute(0, 2, 1)).detach()  # (B, HW, HW)
    # 正規化スケールを戻す
    P = P * float(HW)

    # feat2 を feat1 に合わせてアライメント
    # f2: (B, C, HW), P.permute: (B, HW, HW)
    aligned = torch.matmul(f2, P.permute(0, 2, 1).to(device))
    aligned_feat2 = aligned.view(B, C, H, W)

    return aligned_feat2