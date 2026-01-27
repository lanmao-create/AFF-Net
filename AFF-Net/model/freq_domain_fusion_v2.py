from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT(nn.Module):
    """
    Discrete Wavelet Transform implementation using Haar wavelet.
    This module performs a 2D DWT on a given tensor.
    """
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for DWT using Haar wavelet.
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            A tuple of four tensors (LL, LH, HL, HH) representing the sub-bands.
        """
        # Haar DWT
        ll = (x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) / 2
        lh = (-x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) / 2
        hl = (-x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) / 2
        hh = (x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) / 2
        
        return ll, lh, hl, hh


class IDWT(nn.Module):
    """
    Inverse Discrete Wavelet Transform implementation using Haar wavelet.
    This module performs a 2D IDWT given the four sub-band components.
    """
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for IDWT.
        Args:
            ll: LL sub-band tensor of shape (B, C, H/2, W/2).
            lh: LH sub-band tensor of shape (B, C, H/2, W/2).
            hl: HL sub-band tensor of shape (B, C, H/2, W/2).
            hh: HH sub-band tensor of shape (B, C, H/2, W/2).
        Returns:
            Reconstructed tensor of shape (B, C, H, W).
        """
        b, c, h, w = ll.shape
        out = torch.zeros(b, c, h * 2, w * 2, device=ll.device, dtype=ll.dtype)

        # Reconstruct based on Haar IDWT filters
        out[:, :, 0::2, 0::2] = (ll - lh - hl + hh) / 2
        out[:, :, 1::2, 0::2] = (ll - lh + hl - hh) / 2
        out[:, :, 0::2, 1::2] = (ll + lh - hl - hh) / 2
        out[:, :, 1::2, 1::2] = (ll + lh + hl + hh) / 2

        return out


class ResolutionAlignment(nn.Module):
    """
    分辨率对齐模块，确保两个输入具有相同的空间分辨率
    """
    def __init__(self, align_mode='bilinear'):
        super(ResolutionAlignment, self).__init__()
        self.align_mode = align_mode
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐两个张量的空间分辨率
        Args:
            x1: 第一个输入张量 (B, C1, H1, W1)
            x2: 第二个输入张量 (B, C2, H2, W2)
        Returns:
            对齐后的两个张量，分辨率为较大的那个
        """
        h1, w1 = x1.shape[-2:]
        h2, w2 = x2.shape[-2:]
        
        # 选择较大的分辨率作为目标分辨率
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        # 对齐x1
        if h1 != target_h or w1 != target_w:
            x1 = F.interpolate(x1, size=(target_h, target_w), mode=self.align_mode, align_corners=False)
        
        # 对齐x2
        if h2 != target_h or w2 != target_w:
            x2 = F.interpolate(x2, size=(target_h, target_w), mode=self.align_mode, align_corners=False)
        
        return x1, x2


class FrequencyDomainFusionV2(nn.Module):
    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int, fusion_weight: float = 0.5):
        super(FrequencyDomainFusionV2, self).__init__()
        
        self.dwt = DWT()
        self.idwt = IDWT()
        self.resolution_align = ResolutionAlignment()
        # 可学习标量 α，用于控制双向交互强度
        self.alpha_param = nn.Parameter(torch.tensor(fusion_weight))

        # 子带权重生成器（分别对两个模态的每个子带做 GAP → 1x1 → Sigmoid，得到标量权重）
        self.weight_vis = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels1, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.weight_dsm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 按子带独立的 3×3 融合卷积（跨模态拼接后融合）
        self.subband_fusions = nn.ModuleDict({
            'll': nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=3, padding=1, bias=False),
            'lh': nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=3, padding=1, bias=False),
            'hl': nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=3, padding=1, bias=False),
            'hh': nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=3, padding=1, bias=False),
        })

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Step 0: 分辨率对齐
        x1_aligned, x2_aligned = self.resolution_align(x1, x2)

        # Step 1: DWT on both inputs
        ll1, lh1, hl1, hh1 = self.dwt(x1_aligned)
        ll2, lh2, hl2, hh2 = self.dwt(x2_aligned)

        # Step 2: 为每个子带生成权重（两路各自生成，再交叉使用）
        def gen_weights(vis_band, dsm_band):
            return self.weight_vis(vis_band), self.weight_dsm(dsm_band)

        w_ll_vis, w_ll_dsm = gen_weights(ll1, ll2)
        w_lh_vis, w_lh_dsm = gen_weights(lh1, lh2)
        w_hl_vis, w_hl_dsm = gen_weights(hl1, hl2)
        w_hh_vis, w_hh_dsm = gen_weights(hh1, hh2)

        # Step 3: 自适应标量 α 控制双向交互强度
        alpha = torch.sigmoid(self.alpha_param)

        # Step 4: 子带级残差式跨模态调制
        def enhance(vis_band, dsm_band, w_vis, w_dsm):
            vis_enh = vis_band + alpha * (vis_band * w_dsm)
            dsm_enh = dsm_band + (1 - alpha) * (dsm_band * w_vis)
            return vis_enh, dsm_enh

        ll1_e, ll2_e = enhance(ll1, ll2, w_ll_vis, w_ll_dsm)
        lh1_e, lh2_e = enhance(lh1, lh2, w_lh_vis, w_lh_dsm)
        hl1_e, hl2_e = enhance(hl1, hl2, w_hl_vis, w_hl_dsm)
        hh1_e, hh2_e = enhance(hh1, hh2, w_hh_vis, w_hh_dsm)

        # Step 5: 逐子带跨模态拼接 + 3×3 卷积融合
        def fuse_subband(key, vis_band, dsm_band):
            fused = torch.cat([vis_band, dsm_band], dim=1)
            return self.subband_fusions[key](fused)

        ll_f = fuse_subband('ll', ll1_e, ll2_e)
        lh_f = fuse_subband('lh', lh1_e, lh2_e)
        hl_f = fuse_subband('hl', hl1_e, hl2_e)
        hh_f = fuse_subband('hh', hh1_e, hh2_e)

        # Step 6: IDWT 将频域融合特征映射回空间域
        output = self.idwt(ll_f, lh_f, hl_f, hh_f)

        return output


class SimpleFeatureFusion(nn.Module):
    """
    简单的特征融合模块，用于特征层面的融合
    """
    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int):
        super(SimpleFeatureFusion, self).__init__()
        
        self.resolution_align = ResolutionAlignment()
        
        # 通道对齐
        self.conv1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1, bias=False)
        
        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力权重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 分辨率对齐
        x1_aligned, x2_aligned = self.resolution_align(x1, x2)
        
        # 通道对齐
        f1 = self.conv1(x1_aligned)
        f2 = self.conv2(x2_aligned)
        
        # 计算注意力权重
        concat_feat = torch.cat([f1, f2], dim=1)
        att_weights = self.attention(concat_feat)  # (B, 2, 1, 1)
        
        # 加权融合
        weighted_f1 = f1 * att_weights[:, 0:1, :, :]
        weighted_f2 = f2 * att_weights[:, 1:2, :, :]
        
        # 最终融合
        fused = torch.cat([weighted_f1, weighted_f2], dim=1)
        output = self.fusion_conv(fused)
        
        return output 
