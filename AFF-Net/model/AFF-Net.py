import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from model.swintransformerv2 import SwinTransformerV2
from model.FTUNetFormer import Decoder
from pytorch_wavelets import DWTForward, IDWT


class WaveletExtractor(nn.Module):
    """
    对输入的RGB图像执行4级DWT，以提取1/4、1/8和1/16尺度的高频细节。
    """
    def __init__(self, in_chans=3, wave='haar'):
        super().__init__()
        # J=4 表示进行4级分解，这样我们才能得到最深到1/16尺度的高频信息
        self.dwt = DWTForward(J=4, wave=wave, mode='reflect')
        self.in_chans = in_chans

    def forward(self, x):
        # x 是输入图像 (B, C, H, W)
        yl, yh = self.dwt(x)  # yl是低频，yh是包含各级高频信息的列表
        
        # yh[0] -> 第1级高频 (H/2, W/2)
        # yh[1] -> 第2级高频 (H/4, W/4)
        # yh[2] -> 第3级高频 (H/8, W/8)
        # yh[3] -> 第4级高频 (H/16, W/16)
        
        # 将每个高频张量的LH, HL, HH三个方向的细节展平到一个维度
        details_h4 = yh[1].view(x.shape[0], self.in_chans * 3, x.shape[2] // 4, x.shape[3] // 4)
        details_h8 = yh[2].view(x.shape[0], self.in_chans * 3, x.shape[2] // 8, x.shape[3] // 8)
        details_h16 = yh[3].view(x.shape[0], self.in_chans * 3, x.shape[2] // 16, x.shape[3] // 16)
        
        details = {
            '1_4': details_h4,
            '1_8': details_h8,
            '1_16': details_h16
        }
        return yl, yh, details


class GatedWaveletFusion(nn.Module):
    """
    使用门控机制将解码器特征与小波细节进行融合。
    解码器特征会生成一个注意力门，来控制小波特征的引入程度。
    """
    def __init__(self, decoder_ch, wavelet_ch):
        super().__init__()
        # 门控卷积，从解码器特征生成注意力图
        self.gate_conv = nn.Sequential(
            nn.Conv2d(decoder_ch, decoder_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_ch),
            nn.Sigmoid()  # Sigmoid输出0-1之间的值作为门控权重
        )
        
        # 用于将小波特征通道数调整为与解码器特征一致
        self.align_conv = nn.Conv2d(wavelet_ch, decoder_ch, kernel_size=1, padding=0, bias=False)

        # 最终的融合和特征提炼卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(decoder_ch, decoder_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, decoder_feat, wavelet_feat):
        # 确保空间维度匹配
        if wavelet_feat.shape[-2:] != decoder_feat.shape[-2:]:
            wavelet_feat = F.interpolate(wavelet_feat, size=decoder_feat.shape[-2:], mode='bilinear', align_corners=False)

        # 1. 从解码器特征生成门控信号
        gate = self.gate_conv(decoder_feat)
        
        # 2. 调整小波特征通道，并应用门控
        aligned_wavelet_feat = self.align_conv(wavelet_feat)
        gated_wavelet_feat = aligned_wavelet_feat * gate
        
        # 3. 将加权后的小波特征与原始解码器特征相加融合
        fused = decoder_feat + gated_wavelet_feat
        
        # 4. 通过最后的卷积进行特征提炼
        fused = self.fusion_conv(fused)
        
        return fused


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DSMPyramid(nn.Module):
    """Build a simple DSM feature pyramid aligned to Swin stages.
    Outputs 4 feature maps with channels [embed_dim, 2*embed_dim, 4*embed_dim, 8*embed_dim].
    Spatial sizes approximately match Swin stages (1/4, 1/8, 1/16, 1/32 of input).
    """
    def __init__(self, in_ch=1, embed_dim=128):
        super().__init__()
        c1, c2, c3, c4 = embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8
        # stride 2 twice -> 1/4
        self.l1 = nn.Sequential(
            ConvBNReLU(in_ch, c1 // 2, 3, stride=2),  # 1/2
            ConvBNReLU(c1 // 2, c1, 3, stride=2),      # 1/4
        )
        # 1/8
        self.l2 = nn.Sequential(
            ConvBNReLU(c1, c2, 3, stride=2),
        )
        # 1/16
        self.l3 = nn.Sequential(
            ConvBNReLU(c2, c3, 3, stride=2),
        )
        # 1/32
        self.l4 = nn.Sequential(
            ConvBNReLU(c3, c4, 3, stride=2),
        )

    def forward(self, x):
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return f1, f2, f3, f4


class FuseConv(nn.Module):
    """Fuse RGB and DSM features by concatenation + 1x1 conv to target channels."""
    def __init__(self, in_rgb, in_dsm, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_rgb + in_dsm, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, frgb, fdsm):
        return self.conv(torch.cat([frgb, fdsm], dim=1))


class FTUNetFormer_v2(nn.Module):
    """
    Two-stage deep fusion model:
    - Stage 1: RGB backbone (SwinTransformerV2) + Decoder, DSM ignored.
    - Stage 2: Add DSM pyramid and fuse at each encoder scale before decoding.
    """
    def __init__(self,
                 num_classes=6,
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 window_size=16,
                 drop_path_rate=0.3,
                 decode_channels=256,
                 dropout=0.1):
        super().__init__()

        self.backbone = SwinTransformerV2(img_size=256, in_chans=3, embed_dim=embed_dim,
                                          depths=list(depths), num_heads=list(num_heads),
                                          window_size=window_size, drop_path_rate=drop_path_rate)

        encoder_channels = [int(embed_dim * 2 ** i) for i in range(4)]
        self.decoder = Decoder(encoder_channels=encoder_channels,
                               decode_channels=decode_channels,
                               dropout=dropout,
                               window_size=8,  # keep small window in decoder blocks
                               num_classes=num_classes)

        # DSM path for stage 2
        self.dsm_pyramid = DSMPyramid(in_ch=1, embed_dim=embed_dim)

        # Fusion blocks (keep channels unchanged after fusion)
        self.fuse1 = FuseConv(encoder_channels[0], encoder_channels[0], encoder_channels[0])
        self.fuse2 = FuseConv(encoder_channels[1], encoder_channels[1], encoder_channels[1])
        self.fuse3 = FuseConv(encoder_channels[2], encoder_channels[2], encoder_channels[2])
        self.fuse4 = FuseConv(encoder_channels[3], encoder_channels[3], encoder_channels[3])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_rgb, x_dsm, stage=1):
        h, w = x_rgb.size()[-2:]

        # RGB backbone
        r1, r2, r3, r4 = self.backbone(x_rgb)  # tuple of 4 feature maps

        if stage == 2:
            d1, d2, d3, d4 = self.dsm_pyramid(x_dsm)
            # Ensure spatial alignment via interpolation if needed
            if d1.shape[-2:] != r1.shape[-2:]: d1 = F.interpolate(d1, size=r1.shape[-2:], mode='bilinear', align_corners=False)
            if d2.shape[-2:] != r2.shape[-2:]: d2 = F.interpolate(d2, size=r2.shape[-2:], mode='bilinear', align_corners=False)
            if d3.shape[-2:] != r3.shape[-2:]: d3 = F.interpolate(d3, size=r3.shape[-2:], mode='bilinear', align_corners=False)
            if d4.shape[-2:] != r4.shape[-2:]: d4 = F.interpolate(d4, size=r4.shape[-2:], mode='bilinear', align_corners=False)

            r1 = self.fuse1(r1, d1)
            r2 = self.fuse2(r2, d2)
            r3 = self.fuse3(r3, d3)
            r4 = self.fuse4(r4, d4)

        x = self.decoder(r1, r2, r3, r4, h, w)
        return x


def ft_unetformer_v2(pretrained=True,
                      num_classes=6,
                      weight_path='pretrain/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'):
    model = FTUNetFormer_v2(num_classes=num_classes,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            decode_channels=256)

    if pretrained and weight_path is not None:
        old = torch.load(weight_path)
        # timm ckpt might store in ['model'] or root
        state_old = old['model'] if 'model' in old else old
        model_dict = model.state_dict()
        # Load only backbone weights where shapes match
        mapped = {}
        for k, v in state_old.items():
            new_k = 'backbone.' + k if not k.startswith('backbone.') else k
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                mapped[new_k] = v
        model_dict.update(mapped)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded backbone weights from {weight_path}")

    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from model.swintransformerv2 import SwinTransformerV2
from model.freq_domain_fusion_v2 import FrequencyDomainFusionV2, SimpleFeatureFusion

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained models,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.apply(self._init_weights)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the models into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp_decoder(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6,
                 wavelet_rgb_chans=9):  # 3 (RGB通道) * 3 (LH, HL, HH) = 9
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=16, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        # 增加b1模块，加强最终特征学习
        self.b1 = Block(dim=decode_channels, num_heads=16, window_size=window_size)

        # 为两个尺度创建小波融合模块
        self.wave_fuse_8 = GatedWaveletFusion(decode_channels, wavelet_rgb_chans)  # 融合 1/8  尺度特征
        self.wave_fuse_4 = GatedWaveletFusion(decode_channels, wavelet_rgb_chans)  # 融合 1/4  尺度特征

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, wavelet_details, h, w):

        x = self.b4(self.pre_conv(res4))

        # 上采样到 1/16
        x = self.p3(x, res3)
        # 移除了 1/16 尺度的小波细节融合
        x = self.b3(x)

        # 上采样到 1/8
        x = self.p2(x, res2)
        # 融合 1/8 尺度的小波细节
        x = self.wave_fuse_8(x, wavelet_details['1_8'])
        x = self.b2(x)

        # 上采样到 1/4
        x = self.p1(x, res1)
        # 融合 1/4 尺度的小波细节
        x = self.wave_fuse_4(x, wavelet_details['1_4'])

        # 应用b1模块
        x = self.b1(x)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FTUNetFormerV2(nn.Module):
    """
    改进版本的FTUNetFormer，采用双分支架构解决预训练权重失配和分辨率差异问题
    """
    def __init__(self,
                 decode_channels=256,
                 dropout=0.2,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 freeze_stages=-1,
                 window_size=8,
                 num_classes=6,
                 in_channels_rgb=3,
                 in_channels_dsm=1,
                 use_fusion=True,
                 fusion_type='simple'  # 'simple' or 'frequency'
                 ):
        super().__init__()
        
        self.use_fusion = use_fusion
        self.fusion_type = fusion_type
        
        # 仅为RGB输入创建小波特征提取器
        self.wavelet_extractor_rgb = WaveletExtractor(in_chans=in_channels_rgb)
        
        # 新增：小波逆变换模块，用于计算重建损失
        self.wavelet_reconstructor = IDWT(wave='haar')
        
        # RGB分支 - 保持预训练权重
        self.rgb_backbone = SwinTransformerV2(
            in_chans=in_channels_rgb, 
            embed_dim=embed_dim, 
            depths=depths, 
            num_heads=num_heads, 
            frozen_stages=freeze_stages
        )
        
        if self.use_fusion:
            # DSM分支 - 独立学习
            self.dsm_backbone = SwinTransformerV2(
                in_chans=in_channels_dsm, 
                embed_dim=embed_dim, 
                depths=depths, 
                num_heads=num_heads, 
                frozen_stages=-1  # 不冻结，让其学习DSM特征
            )
            
            # 多尺度融合模块
            encoder_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
            self.fusion_modules = nn.ModuleList()
            
            for i in range(4):
                if fusion_type == 'frequency':
                    fusion_module = FrequencyDomainFusionV2(
                        in_channels1=encoder_channels[i], 
                        in_channels2=encoder_channels[i], 
                        out_channels=encoder_channels[i],
                        fusion_weight=0.3  # RGB权重更高
                    )
                else:  # simple fusion
                    fusion_module = SimpleFeatureFusion(
                        in_channels1=encoder_channels[i], 
                        in_channels2=encoder_channels[i], 
                        out_channels=encoder_channels[i]
                    )
                self.fusion_modules.append(fusion_module)
        
        encoder_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        # 初始化Decoder时传入小波特征的通道数
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes, 
                               wavelet_rgb_chans=in_channels_rgb * 3)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x_rgb, x_dsm=None, stage=1):
        h, w = x_rgb.size()[-2:]

        # 步骤1: 从原始RGB图像中提取多尺度高频细节, 同时保留低频和高频用于重建
        yl, yh_list, wavelet_details = self.wavelet_extractor_rgb(x_rgb)

        # 步骤1.5: 重建图像用于计算辅助损失
        x_rgb_recon = self.wavelet_reconstructor((yl, yh_list))

        # 步骤2: 正常通过主干网络提取语义特征
        rgb_features = self.rgb_backbone(x_rgb)

        # 根据 stage 决定是否进行融合（stage==2 时启用）
        use_fusion_now = (self.use_fusion and stage == 2 and x_dsm is not None)

        if use_fusion_now:
            # DSM分支特征提取
            dsm_features = self.dsm_backbone(x_dsm)

            # 多尺度特征融合
            fused_features = []
            for rgb_feat, dsm_feat, fusion_module in zip(rgb_features, dsm_features, self.fusion_modules):
                fused_feat = fusion_module(rgb_feat, dsm_feat)
                fused_features.append(fused_feat)

            # 步骤3: 将主干网络特征和小波特征一同送入解码器
            x_seg = self.decoder(fused_features[0], fused_features[1], fused_features[2], fused_features[3], wavelet_details, h, w)
        else:
            # 仅使用RGB特征
            # 对于纯RGB训练阶段，同样送入小波特征
            x_seg = self.decoder(rgb_features[0], rgb_features[1], rgb_features[2], rgb_features[3], wavelet_details, h, w)

        # 返回主分割结果和用于计算辅助损失的重建图像
        return x_seg, x_rgb_recon


def ft_unetformer_v2(pretrained=True, num_classes=6, freeze_stages=-1, decoder_channels=256,
                     use_fusion=True, fusion_type='simple',
                     weight_path='pretrain/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'):
    """
    创建改进版本的FTUNetFormer模型
    
    Args:
        pretrained: 是否使用预训练权重
        num_classes: 类别数
        freeze_stages: 冻结阶段数
        decoder_channels: 解码器通道数
        use_fusion: 是否使用DSM融合
        fusion_type: 融合类型 ('simple' 或 'frequency')
        weight_path: 预训练权重路径
    """
    model = FTUNetFormerV2(
        num_classes=num_classes,
        freeze_stages=freeze_stages,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        decode_channels=decoder_channels,
        use_fusion=use_fusion,
        fusion_type=fusion_type
    )

    if pretrained and weight_path is not None:
        try:
            old = torch.load(weight_path)
            old_dict = old['model'] if 'model' in old else old
            model_dict = model.state_dict()

            def add_mapped(prefix):
                mapped = {}
                for k, v in old_dict.items():
                    new_k = f"{prefix}{k}"
                    if new_k in model_dict and model_dict[new_k].shape == v.shape:
                        mapped[new_k] = v
                return mapped

            # 为RGB分支加载匹配形状的预训练权重
            mapped_rgb = add_mapped('rgb_backbone.')
            model_dict.update(mapped_rgb)

            # DSM分支仅加载形状匹配的权重（会自动跳过首层3->1不匹配的卷积）
            if use_fusion:
                mapped_dsm = add_mapped('dsm_backbone.')
                model_dict.update(mapped_dsm)

            model.load_state_dict(model_dict, strict=False)
            print('Loaded backbone weights from', weight_path)
            if use_fusion:
                print('Fusion mode:', fusion_type)
        except Exception as e:
            print(f'Warning: Could not load pretrained weights: {e}')
    
    return model 