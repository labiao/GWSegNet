import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from .gated_cnn_block import GatedCNNBlock



def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Mamba_Block(nn.Module):
    def __init__(self,
                 depth=24,
                 embed_dim=192,
                 out_dim=192 * 2,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 fused_add_norm=False,
                 if_rope=False,
                 flip_img_sequences_ratio=-1.) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dims = embed_dim
        self.out_dims = out_dim
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_rope = if_rope

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                GatedCNNBlock(
                    dim=embed_dim,
                    expansion_ratio=8/3,
                    kernel_size=7,
                    conv_ratio=1.0,
                    drop_path=0.1,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        self.apply(segm_init_weights)

    def forward(self, x, residual=None):
        B, C, H, W = x.shape  # c的维度应该都是128

        hidden_states = x.permute(0, 2, 3, 1)
        if residual is not None:
            residual = residual.permute(0, 2, 3, 1)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)  # 运行了4次

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            residual = residual.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states.transpose(1, 2).view(B, C, H, W)


if __name__ == "__main__":
    # 设置测试参数
    B = 1  # 批次大小
    C = 32  # 通道数
    H = 224  # 高度
    W = 224  # 宽度
    d_state = 16  # 状态维度
    drop_path = 0.1  # Drop路径的概率
    attn_drop_rate = 0  # 注意力丢弃率

    # 创建一个随机输入张量，形状为 (B, H, W, C)
    x = torch.randn(B, H, W, C).cuda()
    # 初始化
    model = Mamba_Block().cuda()
    print(model)
    # 运行模型前向传播
    output = model(x)
    print("\n微信公众号: AI缝合术!\n")
    # 打印输出的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
