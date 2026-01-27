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


class FrequencyDomainFusion(nn.Module):
    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int):
        super(FrequencyDomainFusion, self).__init__()
        
        self.dwt = DWT()
        self.idwt = IDWT()

        # Attention modules for cross-fusion
        # The number of input channels for attention is 4 times the number of input image channels,
        # as we concatenate the LL, LH, HL, HH sub-bands.
        self.attn1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * in_channels1, 4, kernel_size=1, bias=False), # Output 4 channels for 4 sub-bands
            nn.Sigmoid()
        )
        self.attn2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * in_channels2, 4, kernel_size=1, bias=False), # Output 4 channels for 4 sub-bands
            nn.Sigmoid()
        )
        
        # Fusion convolution
        # The input channels for this layer is the sum of channels from both streams,
        # multiplied by 4 (for the 4 sub-bands).
        fused_channels = 4 * (in_channels1 + in_channels2)
        self.conv_fusion = nn.Conv2d(fused_channels, 4 * out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(4 * out_channels)
        self.relu_fusion = nn.ReLU(inplace=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: e.g., RGB image (B, C1, H, W)
        # x2: e.g., DSM image (B, C2, H, W)

        # Step 1: DWT on both inputs
        ll1, lh1, hl1, hh1 = self.dwt(x1)
        ll2, lh2, hl2, hh2 = self.dwt(x2)

        # Step 2: Calculate attention weights from each stream
        dwt1_combined = torch.cat([ll1, lh1, hl1, hh1], dim=1)
        dwt2_combined = torch.cat([ll2, lh2, hl2, hh2], dim=1)
        
        # attn_w1 is from stream 1 (x1), to be applied to stream 2 (x2)
        attn_w1 = self.attn1(dwt1_combined)  # Shape: (B, 4, 1, 1)
        # attn_w2 is from stream 2 (x2), to be applied to stream 1 (x1)
        attn_w2 = self.attn2(dwt2_combined)  # Shape: (B, 4, 1, 1)

        # Step 3: Apply cross-attention as per the diagram
        # (Residual connection + attention modulation from other stream)
        ll1_a = ll1 + ll1 * attn_w2[:, 0:1, :, :]
        lh1_a = lh1 + lh1 * attn_w2[:, 1:2, :, :]
        hl1_a = hl1 + hl1 * attn_w2[:, 2:3, :, :]
        hh1_a = hh1 + hh1 * attn_w2[:, 3:4, :, :]
        
        ll2_a = ll2 + ll2 * attn_w1[:, 0:1, :, :]
        lh2_a = lh2 + lh2 * attn_w1[:, 1:2, :, :]
        hl2_a = hl2 + hl2 * attn_w1[:, 2:3, :, :]
        hh2_a = hh2 + hh2 * attn_w1[:, 3:4, :, :]
        
        # Step 4: Fuse sub-bands from both streams by concatenation
        ll_fused = torch.cat([ll1_a, ll2_a], dim=1)
        lh_fused = torch.cat([lh1_a, lh2_a], dim=1)
        hl_fused = torch.cat([hl1_a, hl2_a], dim=1)
        hh_fused = torch.cat([hh1_a, hh2_a], dim=1)
        
        # Step 5: Concat all fused sub-bands and apply 3x3 convolution
        fused_combined = torch.cat([ll_fused, lh_fused, hl_fused, hh_fused], dim=1)
        fused_conv = self.relu_fusion(self.bn_fusion(self.conv_fusion(fused_combined)))

        # Step 6: Split channels back into 4 sub-bands for IDWT
        b, c, h, w = fused_conv.shape
        c_out = c // 4
        ll_out, lh_out, hl_out, hh_out = torch.split(fused_conv, c_out, dim=1)

        # Step 7: Apply IDWT to get the final fused feature map in spatial domain
        output = self.idwt(ll_out, lh_out, hl_out, hh_out)

        return output 