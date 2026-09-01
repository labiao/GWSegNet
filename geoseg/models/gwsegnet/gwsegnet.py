import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .cga_fusion import CGAFusion
from .dinov2 import DINOv2
from .gated_cnn_ssm import Mamba_Block
from .lsk import LSKblock
from .transformer import TransformerEncoder, TransformerEncoderLayer


class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            # nn.ReLU6()
            nn.SiLU(inplace=True)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class Residual_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, if_downsample=False, **kwargs):
        super(Residual_Block, self).__init__()
        self.if_downsample = if_downsample
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel))


    def forward(self, x):
        identity = x
        if self.if_downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    

class SpatialAttnLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.avgpool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.maxpool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.avgpool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxpool_w = nn.AdaptiveMaxPool2d((1, None))

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(inplace=True),
        )
        self.norm = nn.BatchNorm2d(d_model)
        self.norm1 = nn.GroupNorm(16, d_model)
        self.norm2 = nn.GroupNorm(16, d_model)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(d_model, d_model, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x, v):
        B, C, H, W = v.shape

        x_h = (self.avgpool_h(x) + self.maxpool_h(x)).squeeze(3).permute(0, 2, 1)
        x_w = (self.avgpool_w(x) + self.maxpool_w(x)).squeeze(2).permute(0, 2, 1)
        q = self.mlp(x_h)
        k = self.mlp(x_w).transpose(-1, -2)

        weight_score = torch.matmul(q, k)
        weight_probs = nn.ReLU(inplace=True)(weight_score).unsqueeze(1)

        src2 = weight_probs * v

        src = v + self.norm(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + src2
        src = self.norm2(src)

        return src.reshape(B, C, H, W)
    

class SpaceAttn(nn.Module):
    def __init__(self, d_model, num_layers, norm=None):
        super().__init__()
        SAL = SpatialAttnLayer(d_model)
        self.layers = nn.ModuleList([copy.deepcopy(SAL) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src1, src2=None) -> torch.Tensor:
        if src2 is None:
            src2= src1
        output = src1
        for layer in self.layers:
            output = layer(output, src2)

        if self.norm is not None:
            output = self.norm(output)

        return output

class MF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, depth=4,
                 bimamba_type='v2'):
        super().__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=1)
        self.fusion = CGAFusion(decode_channels)
        self.mamba = Mamba_Block(depth=depth, embed_dim=decode_channels, out_dim=decode_channels)
        self.post_conv = Residual_Block(decode_channels, decode_channels)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # x 为深层特征，res为上一层浅层特征
        res = self.pre_conv(res)  # (8，336，64，64) --> (8，128，64，64)
        x = self.fusion(x, res)
        x = self.mamba(x)
        x = self.post_conv(x)  # 再经过一个残差块 得到(8，128，64，64)
        
        return x
    
# ──────────────── Cross-Scale Fusion ──────────────── #
class ScaleFusion(nn.Module):
    """
    根据 Selective Kernel 思想，用 Softmax 门控自适应融合多尺度特征。
    传入一个 list(tensor)，元素尺寸一致，通道数=C。
    """
    def __init__(self, ch, n_branch, r=8):
        super().__init__()
        self.n = n_branch
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(ch // r, n_branch, 1, bias=False)            # 输出 n 个 gate
        )
        self.init_weight()

    def forward(self, xs):                      # xs = list of tensors
        assert len(xs) == self.n
        u = sum(xs)
        # 先做 element-wise sum
        gates = self.fc(u).softmax(1)           # B,n,1,1
        out = 0
        for i, x in enumerate(xs):
            out += x * gates[:, i:i+1]          # broadcast 乘权
        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def _resize_feature(x, size):
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class DINOv2FeatureAdapter(nn.Module):
    def __init__(self, encoder_size='small', pretrained=True,
                 weight_path='./model_weights/dinov2_small.pth'):
        super().__init__()
        self.encoder_size = encoder_size
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }
        self.embed_dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        out_channels = [48, 96, 192, 384]
        embed_dim = self.embed_dims[encoder_size]

        self.backbone = DINOv2(model_name=encoder_size)
        if pretrained and weight_path:
            weight_path = Path(weight_path)
            if not weight_path.is_file():
                raise FileNotFoundError(
                    f'DINOv2 weights not found: {weight_path}. '
                    'Set GWSegNet(pretrained=False) or provide dinov2_weight_path.'
                )
            self.backbone.load_state_dict(
                torch.load(weight_path, map_location='cpu', weights_only=True)
            )

        self.scale_fuser = ScaleFusion(embed_dim, n_branch=4)
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, out_channel, kernel_size=1, stride=1, padding=0)
            for out_channel in out_channels
        ])

    def forward(self, x):
        _, _, h, w = x.shape
        features = self.backbone.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder_size],
            reshape=True
        )
        features = list(features)
        features[-1] = self.scale_fuser(features)

        patch_h, patch_w = features[0].shape[-2:]
        target_sizes = [
            (patch_h * 4, patch_w * 4),
            (patch_h * 2, patch_w * 2),
            (patch_h, patch_w),
            (max(patch_h // 2, 1), max(patch_w // 2, 1)),
        ]
        return [
            _resize_feature(project(feature), target_sizes[i])
            for i, (project, feature) in enumerate(zip(self.projects, features))
        ]


class TimmFeatureAdapter(nn.Module):
    def __init__(self, model_name, pretrained=True, out_channels=(48, 96, 192, 384), img_size=512):
        super().__init__()
        self.img_size = img_size
        create_kwargs = dict(
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained
        )
        if img_size is not None:
            create_kwargs['img_size'] = img_size

        for optional_kwargs in (
            dict(output_stride=32),
            dict(dynamic_img_size=True),
            {}
        ):
            try:
                self.backbone = timm.create_model(model_name, **create_kwargs, **optional_kwargs)
                break
            except TypeError:
                self.backbone = None
        if self.backbone is None:
            self.backbone = timm.create_model(model_name, **create_kwargs)

        encoder_channels = self.backbone.feature_info.channels()
        if len(encoder_channels) < 4:
            raise ValueError(f'{model_name} only returns {len(encoder_channels)} feature maps; need 4.')
        encoder_channels = encoder_channels[:4]
        self.encoder_channels = encoder_channels
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
            for in_channel, out_channel in zip(encoder_channels, out_channels)
        ])

    def forward(self, x):
        _, _, h, w = x.shape
        backbone_x = x
        if self.img_size is not None and x.shape[-2:] != (self.img_size, self.img_size):
            backbone_x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        features = self.backbone(backbone_x)[:4]
        target_sizes = [
            (max(h // 4, 1), max(w // 4, 1)),
            (max(h // 8, 1), max(w // 8, 1)),
            (max(h // 16, 1), max(w // 16, 1)),
            (max(h // 32, 1), max(w // 32, 1)),
        ]

        outs = []
        for i, feature in enumerate(features):
            if feature.ndim == 3:
                b, n, c = feature.shape
                side = int(n ** 0.5)
                feature = feature[:, :side * side].transpose(1, 2).reshape(b, c, side, side)
            elif feature.ndim == 4 and feature.shape[1] != self.encoder_channels[i] and feature.shape[-1] == self.encoder_channels[i]:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            feature = self.projects[i](feature)
            outs.append(_resize_feature(feature, target_sizes[i]))
        return outs


def build_backbone_adapter(
    backbone_type='dinov2',
    backbone_name=None,
    pretrained=True,
    dinov2_size='small',
    dinov2_weight_path='./model_weights/dinov2_small.pth',
    backbone_img_size=512,
):
    timm_backbones = {
        'sam': backbone_name or 'samvit_base_patch16.sa1b',
        'clip': backbone_name or 'vit_base_patch16_clip_224.openai',
        'swin': backbone_name or 'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
    }
    if backbone_type == 'dinov2':
        return DINOv2FeatureAdapter(dinov2_size, pretrained, dinov2_weight_path)
    if backbone_type in timm_backbones:
        return TimmFeatureAdapter(
            timm_backbones[backbone_type],
            pretrained=pretrained,
            img_size=backbone_img_size
        )
    raise ValueError(f'Unsupported backbone_type: {backbone_type}')


class FeatureDetailDistiller(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64, dropout=0.1):
        super().__init__()
        self.channel = decode_channels
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=1)

        # self.attn = SpaceAttn(decode_channels, 1)
        self.attn = LSKblock(decode_channels)

        self.conv_f = ConvBN(decode_channels * 2, decode_channels, kernel_size=1)
        # self.lsk = LSKblock(128)
        # self.lwga = LWGA_Block(128,2)
        # self.conv1x1 = ConvBNSiLU(decode_channels, decode_channels, kernel_size=1)
        # self.conv3x3 = ConvBNSiLU(decode_channels, decode_channels, kernel_size=3)
        # self.conv5x5 = ConvBNSiLU(decode_channels, decode_channels, kernel_size=3, dilation=2)
        # self.conv7x7 = ConvBNSiLU(decode_channels, decode_channels, kernel_size=3, dilation=3)
        # ===== 2) 新增 ScaleFusion 取代后面的 torch.cat =====
        # self.scale_fuser = ScaleFusion(decode_channels, n_branch=4)
        #
        # self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        # self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2)
        # self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重
        #
        # self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.eps = 1e-8
        self.post_conv = Residual_Block(decode_channels, decode_channels, if_downsample=True)

    def forward(self, x, res):
        # ---------- 上半部分保持不变 ----------
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # 这里的res(8,48,256,256)是最浅层特征了
        # x 从 (8,128,128,128) --> (8,128,256,256)
        res = self.pre_conv(res)  # (8,48,256,256) --> (8,128,256,256)
        x = torch.cat([x, res], dim=1)  # 得到 (8,256,256,256)
        x = self.conv_f(x)  # 得到 (8,128,256,256)

         # ---------- 四路卷积 ----------
        # x1 = self.conv1x1(x[:,:self.channel // 4, :, :])  # 得到 (8,32,256,256)
        # x2 = self.conv3x3(x[:,self.channel // 4: self.channel // 2, :, :])  # 得到 (8,32,256,256)
        # x3 = self.conv5x5(x[:,self.channel // 2:self.channel // 4 * 3, :, :])  # 得到 (8,32,256,256)
        # x4 = self.conv7x7(x[:,self.channel // 4 * 3:, :, :])  # 得到 (8,32,256,256)
        # x1 = self.conv1x1(x)
        # x2 = self.conv3x3(x)
        # x3 = self.conv5x5(x)
        # x4 = self.conv7x7(x)
        # x = self.lwga(x)
        # x = torch.cat([x1, x2, x3, x4], dim=1)  # 得到 (8,128,256,256)
        # 替换 torch.cat → ScaleFusion
        # x = self.scale_fuser([x1, x2, x3, x4])      # B,C/4,H,W → B,C/4,H,W

        identity = x

        # ---------- 原通道注意力 (1D Conv) ----------
        # y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        # y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        # y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        # y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        # y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        # x = x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法
        
        # ---------- 两路权重融合 ----------
        # weights = nn.ReLU()(self.weights)  # [1.0, 1.0]
        # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)  # [0.5, 0.5]
        # x = fuse_weights[0] * identity + fuse_weights[1] * x

        # ---------- 残差块 ----------
        x = self.post_conv(x)  # 残差块

        return x
    

class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNSiLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class Decoder(nn.Module):
    def __init__(self, 
                 encoder_channels,
                 decode_channels,
                 num_classes,
                 depth=4,
                 bimamba_type='v2',
                 dropout=0.1,
                 ):
        super().__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)

        encoder_layer = TransformerEncoderLayer(
            decode_channels,
            nhead=8,
            dim_feedforward=decode_channels * 2,
            dropout=dropout,
            activation='silu',
        )
        self.global_attention = TransformerEncoder(copy.deepcopy(encoder_layer), 3)

        self.p3 = MF(
            encoder_channels[-2], decode_channels, depth=depth,
            bimamba_type=bimamba_type,
        )
        self.p2 = MF(
            encoder_channels[-3], decode_channels, depth=depth,
            bimamba_type=bimamba_type,
        )
        self.p1 = FeatureDetailDistiller(encoder_channels[-4], decode_channels)

        self.h = AuxHead(decode_channels, num_classes)

        self.segmentation_head = nn.Sequential(
            ConvBNSiLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1)
            )
        self.init_weight()
        
    def forward(self, res1, res2, res3, res4, h, w):
        # (8,48,256,256) (8,120,128,128) (8,336,64,64) (8,888,32,32)
        x = self.pre_conv(res4)  # (888-->128)
        B, C, H, W = x.shape  # (8,128,32,32)
        x = self.global_attention(x.flatten(2).permute(0, 2, 1))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        if self.training:
            # 这个地方会增加参数量，res4通道更多
            ah = self.h(x)  # 辅助分类器
        
        x = self.p3(x, res3)  # 8，128，64，64
        x = self.p2(x, res2)  # 8，128，128，128
        x = self.p1(x, res1)  # 8，128，256，256

        x = self.segmentation_head(x)  # 8，6，256，256
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)  # 8，6，1024，1024

        if self.training:
            return x, ah
        else:
            return x
    
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class GWSegNet(nn.Module):
    def __init__(self, 
                 decode_channels=128,
                 backbone_name='regnety_016.tv2_in1k',
                 # backbone_name='repvgg_b1g4,rvgg_in1k',
                 pretrained=True,
                 num_classes=2,
                 depth=4,
                 bimamba_type='v2',
                 backbone_type='dinov2',
                 dinov2_size='small',
                 dinov2_weight_path='./model_weights/dinov2_small.pth',
                 backbone_img_size=512,
                 ):
        super().__init__()
        encoder_channels = [48, 96, 192, 384]
        self.backbone = build_backbone_adapter(
            backbone_type=backbone_type,
            backbone_name=backbone_name,
            pretrained=pretrained,
            dinov2_size=dinov2_size,
            dinov2_weight_path=dinov2_weight_path,
            backbone_img_size=backbone_img_size,
        )
        self.decoder = Decoder(
            encoder_channels,
            decode_channels,
            num_classes,
            depth,
            bimamba_type,
        )

    def forward(self, x):
        _, _, H, W = x.shape
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
             x, ah = self.decoder(res1, res2, res3, res4, H, W)

             return [x, ah]
        else:
            x = self.decoder(res1, res2, res3, res4, H, W)

            return x
        
    

if __name__ == '__main__':
    data = torch.rand(1, 3, 504, 504)
    net = GWSegNet(num_classes=2, pretrained=False)
    net.eval()
    out = net(data)
    print(out.shape)
