import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from model.FTUNetFormer import Decoder
from model.swintransformerv2 import SwinTransformerV2

class WaveletFeatureExtractor(nn.Module):
    """
    Extracts multi-scale features from an image using Discrete Wavelet Transform.
    """
    def __init__(self, in_channels=3, out_channels=32, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')
        # After DWT, LL, LH, HL, HH are produced. Each has `in_channels`.
        # So, total channels = in_channels * 4
        self.channel_mixer = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        LL, H_list = self.dwt(x)
        LH, HL, HH = H_list[0][:,:,0,:,:], H_list[0][:,:,1,:,:], H_list[0][:,:,2,:,:]
        
        # Upsample high-frequency components to match LL size if needed
        # Depending on the wavelet and mode, sizes might differ slightly
        target_size = LL.shape[2:]
        LH = F.interpolate(LH, size=target_size, mode='bilinear', align_corners=False)
        HL = F.interpolate(HL, size=target_size, mode='bilinear', align_corners=False)
        HH = F.interpolate(HH, size=target_size, mode='bilinear', align_corners=False)

        # Concatenate all components
        x = torch.cat([LL, LH, HL, HH], dim=1)
        x = self.channel_mixer(x)
        return x

class DsmFeatureExtractor(nn.Module):
    """
    Extracts geometric features from a DSM image using simple convolutions.
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class CrossAttention(nn.Module):
    """
    Cross-Attention module.
    """
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_kv = nn.Linear(in_channels, in_channels * 2)
        self.to_out = nn.Linear(in_channels, in_channels)

    def forward(self, query, key_value):
        # query from one modality, key_value from the other
        # B, C, H, W -> B, N, C where N=H*W
        B, C, H, W = query.shape
        query_seq = query.view(B, C, -1).transpose(1, 2)
        key_value_seq = key_value.view(B, C, -1).transpose(1, 2)

        q = self.to_q(query_seq)
        k, v = self.to_kv(key_value_seq).chunk(2, dim=-1)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(out)
        
        # Reshape back to image format and add residual connection
        out = out.transpose(1, 2).view(B, C, H, W)
        return out + query


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels_rgb=3, in_channels_dsm=1, feature_channels=32, num_heads=8):
        super().__init__()
        self.top_extractor = WaveletFeatureExtractor(in_channels=in_channels_rgb, out_channels=feature_channels)
        self.dsm_extractor = DsmFeatureExtractor(in_channels=in_channels_dsm, out_channels=feature_channels)

        self.cross_attn_top_dsm = CrossAttention(in_channels=feature_channels, num_heads=num_heads)
        self.cross_attn_dsm_top = CrossAttention(in_channels=feature_channels, num_heads=num_heads)
        
        # The output of concatenation will have 2 * feature_channels
        self.output_channels = feature_channels * 2

    def forward(self, top, dsm):
        top_feat = self.top_extractor(top)
        dsm_feat = self.dsm_extractor(dsm)
        
        # Resize DSM features to match TOP features spatial dimensions
        dsm_feat = F.interpolate(dsm_feat, size=top_feat.shape[2:], mode='bilinear', align_corners=False)

        # Let DSM features guide TOP feature enhancement
        top_enhanced = self.cross_attn_top_dsm(query=top_feat, key_value=dsm_feat)
        
        # Let TOP features guide DSM feature enhancement
        dsm_enhanced = self.cross_attn_dsm_top(query=dsm_feat, key_value=top_feat)
        
        # Concat enhanced features
        fused_features = torch.cat([top_enhanced, dsm_enhanced], dim=1)
        
        return fused_features

class Modified_FTUNetFormer(nn.Module):
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
                 fusion_channels=32
                 ):
        super().__init__()
        self.fusion = CrossAttentionFusion(in_channels_rgb=in_channels_rgb, in_channels_dsm=in_channels_dsm, feature_channels=fusion_channels)
        
        # The backbone input channels must match the output of the fusion module
        backbone_in_channels = self.fusion.output_channels
        self.backbone = SwinTransformerV2(in_chans=backbone_in_channels, embed_dim=embed_dim, depths=depths, num_heads=num_heads, frozen_stages=freeze_stages)
        
        encoder_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x_rgb, x_dsm):
        h, w = x_rgb.size()[-2:]
        x = self.fusion(x_rgb, x_dsm)
        res1, res2, res3, res4 = self.backbone(x)
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x

def modified_ft_unetformer(pretrained=True, num_classes=6, freeze_stages=-1, decoder_channels=256,
                 weight_path='pretrain/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'):
    model = Modified_FTUNetFormer(num_classes=num_classes,
                          freeze_stages=freeze_stages,
                          embed_dim=128,
                          depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32),
                          decode_channels=decoder_channels,
                          fusion_channels=32)

    # Pretrained weights are for the backbone. Since the input channels have changed,
    # we cannot load the weights for the patch_embed layer directly.
    # We can load the rest of the backbone weights.
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['model']
        model_dict = model.state_dict()

        # Filter out patch_embed weights from pretrained dict
        old_dict = {k: v for k, v in old_dict.items() if 'patch_embed.proj.weight' not in k}
        
        # Add 'backbone.' prefix
        old_dict = {'backbone.'+ k: v for k, v in old_dict.items() if ('backbone.' + k in model_dict)}
        
        model_dict.update(old_dict)
        model.load_state_dict(model_dict, strict=False) # Use strict=False to ignore missing keys
        print(f'Loaded pretrained weights from {weight_path}, ignoring patch_embed layer.')
    
    return model
