# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = ConvBNAct(in_ch, out_ch, use_bn=use_bn)
        self.conv2 = ConvBNAct(out_ch, out_ch, use_bn=use_bn)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, use_bn=use_bn)
        self.conv2 = ConvBNAct(out_ch, out_ch, use_bn=use_bn)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class IllumUNet(nn.Module):
    """
    轻量 U-Net，用于估计光照图 L（3 通道，归一化到 (0,1)）
    输入：I ∈ [0,1] (B,3,H,W)
    输出：L ∈ (0,1] (B,3,H,W)
    """
    def __init__(self, in_ch=3, base=32, use_bn=True):
        super().__init__()
        # encoder
        self.e1 = nn.Sequential(
            ConvBNAct(in_ch, base, use_bn=use_bn),
            ConvBNAct(base, base, use_bn=use_bn),
        )
        self.e2 = DownBlock(base, base * 2, use_bn=use_bn)
        self.e3 = DownBlock(base * 2, base * 4, use_bn=use_bn)
        self.e4 = DownBlock(base * 4, base * 8, use_bn=use_bn)

        # bottleneck
        self.bott = nn.Sequential(
            DownBlock(base * 8, base * 16, use_bn=use_bn),
            ConvBNAct(base * 16, base * 16, use_bn=use_bn),
        )

        # decoder
        self.u4 = UpBlock(base * 16 + base * 8, base * 8, use_bn=use_bn)
        self.u3 = UpBlock(base * 8 + base * 4, base * 4, use_bn=use_bn)
        self.u2 = UpBlock(base * 4 + base * 2, base * 2, use_bn=use_bn)
        self.u1 = UpBlock(base * 2 + base, base, use_bn=use_bn)

        # 输出 3 通道光照
        self.outc = nn.Conv2d(base, 3, kernel_size=1)

        # 初始化：避免 L 初始为纯 0.5 灰
        nn.init.zeros_(self.outc.weight)
        nn.init.constant_(self.outc.bias, -0.5)  # sigmoid(-0.5)~0.38，偏暗一点更贴近低照

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b  = self.bott(e4)

        d4 = self.u4(b,  e4)
        d3 = self.u3(d4, e3)
        d2 = self.u2(d3, e2)
        d1 = self.u1(d2, e1)

        L = torch.sigmoid(self.outc(d1)) + 1e-3  # 避免后面除零
        return L


class ReflectRefine(nn.Module):
    """
    对物理反射图 R_physical = I / L 做一个极轻的残差细化：
      输入： concat(R_physical, I) ∈ (B,6,H,W)
      输出：ΔR ∈ (B,3,H,W)，最后 R = R_physical + ΔR
    """
    def __init__(self, in_ch=6, base=32, use_bn=True):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(in_ch, base, use_bn=use_bn),
            ConvBNAct(base, base, use_bn=use_bn),
            nn.Conv2d(base, 3, kernel_size=1)
        )

    def forward(self, I, L, R_physical):
        # 输入 concat(R_physical, I)
        rin = torch.cat([R_physical.clamp(0., 1.), I], dim=1)
        rdelta = self.refine(rin)
        R = (R_physical + rdelta).clamp(0., 1.)
        return R


class RetinexDecomposer(nn.Module):
    """
    Retinex 分解器：
      输入:  I ∈ [0,1], B×3×H×W
      输出:  L, R ∈ [0,1]
        I ≈ L * R

    说明：
      - 结构上是 “IllumUNet + 物理除法 + 轻量 R refine”
      - 因果约束（gamma 干预、不变性等）不在这里写，
        而是在训练脚本的 loss 中实现（Causal-Retinex Loss）。
    """
    def __init__(self, base=32, use_refine=True, use_bn=True):
        super().__init__()
        self.illum = IllumUNet(in_ch=3, base=base, use_bn=use_bn)
        self.use_refine = use_refine
        if use_refine:
            self.refine = ReflectRefine(in_ch=6, base=base, use_bn=use_bn)

    def forward(self, I):
        # === 1. 光照估计 ===
        L = self.illum(I)  # (B,3,H,W)

        # === 2. 光照后处理：平滑 + 限幅 ===
        # 灰度平滑：抑制纹理，把 L 更偏向大尺度亮度
        L_gray   = L.mean(1, keepdim=True)  # (B,1,H,W)
        L_smooth = F.avg_pool2d(L_gray, kernel_size=9, stride=1, padding=4)
        # 混合原 L 与平滑 L
        L = 0.6 * L + 0.4 * L_smooth

        # 限制动态范围，避免 L 过小/过大导致 R 爆掉
        L = L.clamp(min=0.15, max=1.0)

        # === 3. 计算物理反射分量 R_physical = I / L ===
        R_physical = I / L.clamp(min=1e-3)

        # === 4. 可选：对 R 做轻量残差细化 ===
        if self.use_refine:
            R = self.refine(I, L, R_physical)
        else:
            R = R_physical

        # === 5. 约束范围，输出 L, R ∈ [0,1] ===
        return L.clamp(0., 1.), R.clamp(0., 1.)


if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256)
    net = RetinexDecomposer(base=32, use_refine=True)
    L, R = net(x)
    print(L.shape, R.shape)
