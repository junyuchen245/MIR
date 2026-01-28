"""Transformer-based VFA variants and feature adapters."""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import MIR.models.Swin_Transformer as swin
import MIR.models.Deformable_Swin_Transformer as dswin
import MIR.models.Deformable_Swin_Transformer_v2 as dswin_v2
import MIR.models as mir_m
from MIR.models.VFA import Decoder, DoubleConv3d, grid_to_flow

class VoxelShuffle(nn.Module):
    """3D voxel shuffle layer.

    Upscales a tensor by factor `r` in each spatial dimension where
    `C_in = C_out * r^3`.

    Inputs:
        x: Tensor of shape [B, C_in, H, W, D].

    Returns:
        Tensor of shape [B, C_out, H*r, W*r, D*r].
    """
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange channels into spatial dimensions."""
        B, C, H, W, D = x.shape
        r = self.r
        if C % (r**3) != 0:
            raise ValueError(
                f"Input channels ({C}) must be divisible by r^3 ({r**3})"
            )
        C_out = C // (r**3)
        # reshape to (B, C_out, r, r, r, H, W, D)
        x = x.view(B, C_out, r, r, r, H, W, D)
        # permute to (B, C_out, H, r, W, r, D, r)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        # merge interleaved dims → (B, C_out, H*r, W*r, D*r)
        x = x.view(B, C_out, H * r, W * r, D * r)
        return x

class Feature2VFA(nn.Module):
    """Convert transformer features into VFA multiscale features.

    Args:
        configs_sw: Swin Transformer config.
        vfa_channels: Channel widths for VFA decoder stages.
        embed_dim: Transformer embedding dimension.
        swin_type: Transformer type ('swin', 'dswin', 'dswinv2').
        in_norm: If True, apply instance normalization to inputs.

    Forward inputs:
        x: Tensor [B, 1, D, H, W] or tuple of tensors for deformable variants.

    Forward outputs:
        List of multiscale feature maps or tuple of lists for dual inputs.
    """
    def __init__(self, configs_sw, vfa_channels=[8, 16, 32, 64], embed_dim=128, swin_type='swin', in_norm=True):
        super().__init__()
        if swin_type == 'swin':
            self.encoder = swin.SwinTransformer(patch_size=configs_sw.patch_size,
                                           in_chans=1,
                                           embed_dim=configs_sw.embed_dim,
                                           depths=configs_sw.depths,
                                           num_heads=configs_sw.num_heads,
                                           window_size=configs_sw.window_size,
                                           mlp_ratio=configs_sw.mlp_ratio,
                                           qkv_bias=configs_sw.qkv_bias,
                                           drop_rate=configs_sw.drop_rate,
                                           drop_path_rate=configs_sw.drop_path_rate,
                                           ape=configs_sw.ape,
                                           spe=configs_sw.spe,
                                           rpe=configs_sw.rpe,
                                           patch_norm=configs_sw.patch_norm,
                                           use_checkpoint=configs_sw.use_checkpoint,
                                           out_indices=configs_sw.out_indices,
                                           pat_merg_rf=configs_sw.pat_merg_rf,
                                           )
        elif swin_type == 'dswin':
            self.encoder = dswin.DefSwinTransformer(patch_size=configs_sw.patch_size,
                                           in_chans=2,
                                           embed_dim=configs_sw.embed_dim,
                                           depths=configs_sw.depths,
                                           num_heads=configs_sw.num_heads,
                                           window_size=configs_sw.window_size,
                                           mlp_ratio=configs_sw.mlp_ratio,
                                           qkv_bias=configs_sw.qkv_bias,
                                           drop_rate=configs_sw.drop_rate,
                                           drop_path_rate=configs_sw.drop_path_rate,
                                           ape=configs_sw.ape,
                                           spe=configs_sw.spe,
                                           rpe=configs_sw.rpe,
                                           patch_norm=configs_sw.patch_norm,
                                           use_checkpoint=configs_sw.use_checkpoint,
                                           out_indices=configs_sw.out_indices,
                                           pat_merg_rf=configs_sw.pat_merg_rf,
                                           dwin_size=configs_sw.dwin_kernel_size,
                                           img_size=configs_sw.img_size,
                                           )
        elif swin_type == 'dswinv2':
            self.encoder = dswin_v2.DefSwinTransformerV2(patch_size=configs_sw.patch_size,
                                           in_chans=2,
                                           embed_dim=configs_sw.embed_dim,
                                           depths=configs_sw.depths,
                                           num_heads=configs_sw.num_heads,
                                           window_size=configs_sw.window_size,
                                           mlp_ratio=configs_sw.mlp_ratio,
                                           qkv_bias=configs_sw.qkv_bias,
                                           drop_rate=configs_sw.drop_rate,
                                           drop_path_rate=configs_sw.drop_path_rate,
                                           ape=configs_sw.ape,
                                           spe=configs_sw.spe,
                                           rpe=configs_sw.rpe,
                                           patch_norm=configs_sw.patch_norm,
                                           use_checkpoint=configs_sw.use_checkpoint,
                                           out_indices=configs_sw.out_indices,
                                           pat_merg_rf=configs_sw.pat_merg_rf,
                                           dwin_size=configs_sw.dwin_kernel_size,
                                           img_size=configs_sw.img_size,
                                           )
        else:
            raise ValueError(f'Unknown Transformer type: {swin_type}')
        if in_norm:
            self.norm_first = nn.InstanceNorm3d(1, affine=True)
        self.in_norm = in_norm
        if_transskip = True
        if_convskip = True
        self.swin_type = swin_type
        self.up0 = mir_m.DecoderBlock(embed_dim*4, vfa_channels[3], skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)
        self.up1 = mir_m.DecoderBlock(vfa_channels[3], vfa_channels[2], skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = mir_m.DecoderBlock(vfa_channels[2], vfa_channels[1], skip_channels=embed_dim//2 if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = mir_m.DecoderBlock(vfa_channels[1], vfa_channels[0], skip_channels=vfa_channels[0] if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        #self.up4 = DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = mir_m.Conv3dReLU(1, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = mir_m.Conv3dReLU(1, vfa_channels[0], 3, 1, use_batchnorm=False)
        self.cb = mir_m.Conv3dReLU(embed_dim*4, vfa_channels[4], 3, 1, use_batchnorm=False)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.vfa_channels = vfa_channels
        self.swin_type = swin_type
        
    def forward(self, x):
        """Run feature extraction and conversion."""
        if self.swin_type == 'swin':
            if self.in_norm:
                x = self.norm_first(x)
            x_s1 = self.avg_pool(x)
            f3 = self.c1(x_s1)
            f4 = self.c2(x)
            out_feats = self.encoder(x)
            f0 = out_feats[-1]
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            feats_0 = self.cb(f0)
            feats_1 = self.up0(f0, f1)
            feats_2 = self.up1(feats_1, f2)
            feats_3 = self.up2(feats_2, f3)
            feats_4 = self.up3(feats_3, f4)
            out_features = [feats_4, feats_3, feats_2, feats_1, feats_0]
            return out_features
        else:
            if self.in_norm:
                x = [self.norm_first(x[0]), self.norm_first(x[1])]
            x_in = x  
            out_feats = self.encoder((x_in[0], x_in[1]))
            x = self.avg_pool(x_in[0])
            f3 = self.c1(x)
            f4 = self.c2(x_in[0])
            f0 = out_feats[-1][0]
            f1 = out_feats[-2][0]
            f2 = out_feats[-3][0]
            feats_0 = self.cb(f0)
            feats_1 = self.up0(f0, f1)
            feats_2 = self.up1(feats_1, f2)
            feats_3 = self.up2(feats_2, f3)
            feats_4 = self.up3(feats_3, f4)
            
            out_features_m = [feats_4, feats_3, feats_2, feats_1, feats_0]
            
            x = self.avg_pool(x_in[1])
            f3 = self.c1(x)
            f4 = self.c2(x_in[1])
            f0 = out_feats[-1][1]
            f1 = out_feats[-2][1]
            f2 = out_feats[-3][1]
            feats_0 = self.cb(f0)
            feats_1 = self.up0(f0, f1)
            feats_2 = self.up1(feats_1, f2)
            feats_3 = self.up2(feats_2, f3)
            feats_4 = self.up3(feats_3, f4)
            out_features_f = [feats_4, feats_3, feats_2, feats_1, feats_0]
            return out_features_m, out_features_f
        

class TransVFA(nn.Module):
    """TransVFA model for image registration.

    Args:
        configs_sw: Swin Transformer config.
        configs: VFA config.
        device: Device to run the model on.
        swin_type: Transformer type ('swin', 'dswin', 'dswinv2').
        return_orginal: If True, return composed grids and stats.
        return_all_flows: If True, return flows for all decoder levels.

    Forward inputs:
        sample: Tuple `(mov, fix)` tensors of shape [B, 1, *spatial].

    Forward outputs:
        Flow(s) and optional auxiliary outputs depending on flags.
    """
    def __init__(self, configs_sw, configs, device, swin_type='swin', return_orginal=False, return_all_flows=False, max_channels=64):
        super().__init__()

        self.dim = len(configs.img_size)
        channels = [min(configs.start_channels * 2**(i+1), max_channels) for i in range(configs.downsamples+1)]
        print('VFA channels: ', channels)
        
        self.swin_type = swin_type
        self.encoder = Feature2VFA(configs_sw, channels, configs_sw.embed_dim, swin_type)
        
        self.decoder = Decoder(
                        dimension=self.dim,
                        downsamples=configs.downsamples,
                        matching_channels=configs.matching_channels,
                        start_channels=configs.start_channels,
                        max_channels=configs.max_channels,
                        skip=configs.skip,
                        initialize=configs.initialize,
                        int_steps=configs.int_steps,
        ).type(torch.float32)
        self.configs = configs
        self.device = device
        self.decoder.R = self.decoder.R.to(device)
        self.return_orginal = return_orginal
        self.return_all_flows = return_all_flows

    def forward(self, sample):
        """Run forward registration.

        Args:
            sample: Tuple `(mov, fix)` tensors.

        Returns:
            Output varies by flags (`return_orginal`, `return_all_flows`).
        """
        mov, fix = sample
        if self.swin_type == 'swin':
            F = self.encoder(fix)
            M = self.encoder(mov)
        else:
            M, F = self.encoder((mov, fix))
        composed_grids = self.decoder(F, M)
        if self.return_orginal:
            results = self.generate_results(composed_grids[-1], sample)
            results.update({'composed_grids': composed_grids,
                            'beta':self.decoder.beta.clone(),})

            if self.configs.affine:
                affine_results = self.generate_affine_results(
                                                        composed_grids[-1],
                                                        sample,
                )
                results.update(affine_results)
        
            return results
        elif self.return_all_flows:
            composed_flows = []
            for i in range(len(composed_grids)):
                composed_flows.append(grid_to_flow(composed_grids[i]))
            return composed_flows
        else:
            return grid_to_flow(composed_grids[-1])