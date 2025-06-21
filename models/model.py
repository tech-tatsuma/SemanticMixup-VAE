import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from models.components import conv3x3, BasicBlock, PreActBlock, Bottleneck, PreActBottleneck

# ResNet全体（4層構造）を定義するクラス
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=4):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        return out

# ResNet18（PreActBlockを使用）
def ResNet18():
    return ResNet(PreActBlock, [2,2,2,2])

# 全体の分類モデル（ResNet + Mixup）
class Resnet_classifier(nn.Module):
    def __init__(self, num_classes=2, z_dim=512):
        super(Resnet_classifier, self).__init__()
        
        self.encoder = ResNet18()
        self.classifier = nn.Linear(z_dim, num_classes)


    def forward(self, x, targets, lam, mode):
        
        if (mode == 'train'):
            
            layer_mix = random.randint(0,1) # 入力レベルが特徴レベルかをランダムに選択
            
            if layer_mix == 0:
                # 入力画像xとラベルtargetsに対してMixup処理を実行
                # 引数:
                #   x: 入力画像テンソル (B, C, H, W)
                #   targets: 対応するラベル (B,)
                #   lam: Mixup係数（0〜1の間の実数）
                #
                # 戻り値:
                #   x: Mixupされた画像テンソル
                #   t_a: 元のラベル
                #   t_b: 混合されたもう一方のラベル（ランダムにペアリング）
                x,t_a,t_b = mixup_process(x, targets, lam)

            out = self.encoder(x, lin=0, lout=0)
            out = self.encoder.layer1(out)
            out = self.encoder.layer2(out)
            out = self.encoder.layer3(out)
            out = self.encoder.layer4(out)

            if layer_mix == 1:
                # out: Sinkhorn距離に基づき空間的に位置合わせ（アライメント）された特徴マップ同士を、係数 lam で Mixup した新しい特徴マップ
                # t_a: 元の入力画像 out に対応するラベル（オリジナルのラベル）
                # t_b: シャッフルされた特徴 out[indices] に対応するラベル（Mixup相手のラベル）
                out,t_a,t_b = mixup_aligned(out, targets, lam)            

            out = F.adaptive_avg_pool2d(out, (1,1))
            out = out.view(out.size(0), -1)  
            cls_output = self.classifier(out)  

            return cls_output, t_a, t_b

            
        elif (mode == 'test'):
            out = self.encoder(x)
            out = F.adaptive_avg_pool2d(out, (1,1))
            out = out.view(out.size(0), -1) 
            cls_output = self.classifier(out)

            return  cls_output