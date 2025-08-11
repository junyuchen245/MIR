import torch.nn.functional as F
import torch.nn as nn
import torch

class SCSEBlock3D(nn.Module):
    """
    Concurrent spatial and channel 'squeeze & excite'
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # channel-wise excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # spatial excitation
        self.sSE = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel gate
        xc = self.cSE(x)
        # spatial gate
        xs = self.sSE(x)
        # combine
        return x * xc + x * xs


class ASPP3D(nn.Module):
    """
    Atrous Spatial Pyramid Pooling without any global pooling branch
    (so it still works on small patches).
    """
    def __init__(self, in_ch: int, out_ch: int, dilations=(1,6,12)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.GELU()
            )
            for d in dilations
        ])
        # fuse them
        self.project = nn.Sequential(
            nn.Conv3d(len(dilations)*out_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
            nn.Dropout3d(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [b(x) for b in self.branches]
        x   = torch.cat(res, dim=1)
        return self.project(x)


class AdvancedDecoder3D(nn.Module):
    """
    Advanced 3D decoder for multi-resolution encoder features.
    
    Args:
      encoder_channels: list of channel counts for each x_feats[i]
      aspp_out        : number of channels after ASPP
      num_classes     : number of output classes (e.g. 133)
    """
    def __init__(
        self,
        encoder_channels: list,
        aspp_out: int     = 128,
        num_classes: int  = 133
    ):
        super().__init__()
        self.num_levels = len(encoder_channels)
        # bottleneck ASPP on the deepest feature
        self.aspp = ASPP3D(encoder_channels[-1], aspp_out, dilations=(1,6,12))
        
        # decoder blocks: one per skip level
        self.blocks = nn.ModuleList()
        in_ch = aspp_out
        # for each skip (from deepest-1 down to highest-res)
        for skip_ch in reversed(encoder_channels[:-1]):
            out_ch = skip_ch
            block = nn.Sequential(
                # fuse upsample + skip
                nn.Conv3d(in_ch + skip_ch, out_ch,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.GELU(),
                
                nn.Conv3d(out_ch, out_ch,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.GELU(),
                
                SCSEBlock3D(out_ch)
            )
            self.blocks.append(block)
            in_ch = out_ch
        
        # final classifier → num_classes
        self.classifier = nn.Conv3d(in_ch, num_classes, kernel_size=1)

    def forward(self, x_feats: list) -> torch.Tensor:
        """
        x_feats: list of length L with
                 x_feats[i].shape == (B, encoder_channels[i], Di, Hi, Wi)
        
        returns:
          logits (B, num_classes, D0, H0, W0)  at the highest resolution
        """
        # start from deepest feature
        x = x_feats[-1]
        x = self.aspp(x)
        
        # progressively upsample and fuse with skips
        for block, skip in zip(self.blocks, reversed(x_feats[:-1])):
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode='trilinear',
                align_corners=False
            )
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        return self.classifier(x)