import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    mixupの損失関数: 2つのラベルと出力の重み付き平均を計算
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
 
def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """ 学習率を指定スケジュールに基づいて調整する関数
    指定したエポックで学習率をgamma倍ずつ減衰させる

    Args:
        optimizer: 最適化オブジェクト
        epoch: 現在のエポック数
        gammas: 減衰係数のリスト（例：[0.1, 0.1]）
        schedule: 減衰を適用するエポック数（例：[30, 40]）
    """
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr # 新しい学習率を設定

    return lr


# Sinkhorn距離を用いた最適輸送の計算モジュール
class SinkhornDistance(nn.Module):
    r"""
    Sinkhornアルゴリズムにより、2つの確率分布の間の正則化されたWasserstein距離を計算するクラス

    Args:
        eps: 正則化項の重み（ε）
        max_iter: Sinkhorn反復の最大回数
        reduction: 出力の集約方法（'none', 'mean', 'sum'のいずれか）
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # x, y: (batch_size, num_points, dim) のテンソル
        C = self._cost_matrix(x, y)  # コスト行列（Wasserstein距離の基礎）
        x_points = x.shape[-2]
        y_points = y.shape[-2]

        # バッチサイズの取得
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # 重み（μとν）は一様分布（各点に等しい重み）
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        # 双対変数の初期化（u, v）
        u = torch.zeros_like(mu).cuda()
        v = torch.zeros_like(nu).cuda()
        actual_nits = 0
        # 収束判定用の誤差しきい値
        thresh = 1e-1

        # Sinkhorn反復
        for i in range(self.max_iter):
            u1 = u  # 更新前のuを保存（収束判定用）

            # 更新式（双対変数の対数スケール更新）
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean() # uの変化量の平均

            actual_nits += 1
            if err.item() < thresh: # 収束したら反復終了
                break

        U, V = u, v
        # 最適輸送計画 π = diag(exp(u)) * exp(-C/ε) * diag(exp(v))
        pi = torch.exp(self.M(C, U, V))
       
        return pi # πは対応行列（アラインメントとして使える）

    def M(self, C, u, v):
        """
        対数スケールのコスト関数
        M_{ij} = (-c_{ij} + u_i + v_j) / ε
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        """
        x_iとy_jのp乗ノルム距離行列を計算

        Args:
            x: (B, N, D)
            y: (B, M, D)
            return: (B, N, M) の距離行列
        """
        x_col = x.unsqueeze(-2).cuda()
        y_lin = y.unsqueeze(-3).cuda()
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        """
        加速Sinkhorn用の補間関数（未使用）
        u, u1: 双対変数
        tau: 補間係数
        """
        return tau * u + (1 - tau) * u1