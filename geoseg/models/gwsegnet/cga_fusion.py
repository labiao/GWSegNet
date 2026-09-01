# --------------------------------------------------------
# 论文：DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention
# GitHub地址：https://github.com/cecret3350/DEA-Net/tree/main
# --------------------------------------------------------

import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

        # -------------------------- 分支1：原有注意力融合分支（核心） --------------------------
        # self.branch1_conv = nn.Conv2d(dim, dim, 1, bias=True)

        # -------------------------- 分支2：直接特征交互分支（加+乘融合） --------------------------
        # self.branch2_conv = nn.Sequential(
        #     nn.Conv2d(dim * 2, dim, 1, bias=False),  # 拼接x/y后降维
        #     nn.BatchNorm2d(dim),  # 批量归一化
        #     nn.ReLU(inplace=True),  # 激活
        #     nn.Conv2d(dim, dim, 3, padding=1, bias=False)  # 3x3卷积增强局部特征
        # )

        # -------------------------- 分支3：跨维度增强分支（升维->交互->降维） --------------------------
        # self.branch3_up = nn.Conv2d(dim, dim * 2, 1, bias=False)  # 升维
        # self.branch3_interact = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False)  # 深度卷积（轻量化交互）
        # self.branch3_down = nn.Conv2d(dim * 2, dim, 1, bias=False)  # 降维回原维度

        # -------------------------- 分支融合：自适应加权机制 --------------------------
        # self.fusion_attn = nn.Sequential(
        #     nn.Conv2d(dim * 3, dim, 1, bias=False),  # 拼接3个分支后降维
        #     nn.BatchNorm2d(dim),
        #     nn.Sigmoid()  # 生成0-1的融合权重
        # )

        # 输出卷积（最终特征整合）
        self.out_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        branch1_out = initial + pattn2 * x + (1 - pattn2) * y
        # result = initial + pattn2 * x

        # 分支2：直接特征交互（捕捉x/y的原始交互信息）
        # xy_concat = torch.cat([x, y], dim=1)  # 拼接x和y的特征
        # branch2_out = self.branch2_conv(xy_concat)

        # 分支3：跨维度增强（升维后深度卷积，提升特征表达）
        # x_up = self.branch3_up(x)
        # y_up = self.branch3_up(y)
        # interact = self.branch3_interact(x_up + y_up)  # 升维后特征交互
        # branch3_out = self.branch3_down(interact)

        # 自适应分支融合（让模型自动学习各分支的重要性）
        # all_branches = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)  # 拼接3个分支
        # fusion_weights = self.fusion_attn(all_branches)  # 生成融合权重

        # 加权融合：每个分支乘以权重后相加
        # fused = branch1_out * fusion_weights + branch2_out * fusion_weights + branch3_out * fusion_weights

        # 最终输出
        result = self.out_conv(branch1_out)

        return result


# 特征融合
if __name__ == '__main__':
    block = CGAFusion(32)
    input1 = torch.rand(3, 32, 64, 64) # 输入 N C H W
    input2 = torch.rand(3, 32, 64, 64)
    output = block(input1, input2)
    print(output.size())