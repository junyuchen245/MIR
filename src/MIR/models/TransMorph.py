'''
TransMorph model
Chen, J., Du, Y., He, Y., Segars, W. P., Li, Y., & Frey, E. C. (2021).
TransMorph: Transformer for unsupervised medical image registration.
arXiv preprint arXiv:2111.10480.
Swin-Transformer code retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
Original paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.
Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import MIR.models.configs_TransMorph as configs
import MIR.models.Swin_Transformer as swin
import MIR.models.registration_utils as utils
import MIR.models.Deformable_Swin_Transformer as dswin
import MIR.models.Deformable_Swin_Transformer_v2 as dswin_v2
import numpy as np

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class TransMorph(nn.Module):
    '''
    TransMorph model
    Args:
        config: Configuration object containing model parameters
        SVF: Boolean indicating whether to use SVF (Time Stationary Velocity Field) integration
        SVF_steps: Number of steps for SVF integration
        swin_type: Type of Swin Transformer to use ('swin' or 'dswin')
    '''
    def __init__(self, config, SVF=True, SVF_steps=7, swin_type='swin'):
        '''
        Original TransMorph Model
        '''
        super(TransMorph, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        if swin_type == 'swin':
            self.transformer = swin.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        elif swin_type == 'dswin':
            self.transformer = dswin.DefSwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           dwin_size=config.dwin_kernel_size,
                                           img_size=config.img_size,
                                           )
        elif swin_type == 'dswinv2':
            self.transformer = dswin_v2.DefSwinTransformerV2(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           dwin_size=config.dwin_kernel_size,
                                           img_size=config.img_size,
                                           )
        else:
            raise ValueError(f'Unknown Transformer type: {swin_type}')
        self.swin_type = swin_type
        self.up0 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        #self.up4 = DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = utils.SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.SVF = SVF
        if self.SVF:
            self.vec_int = utils.VecInt(config.img_size, SVF_steps)

    def forward(self, inputs):
        '''
        Forward pass for the TransMorph model.
        Args:
            inputs: Tuple of moving and fixed images (mov, fix).
        Returns:
            flow: The computed flow field for image registration.
        '''
        mov, fix = inputs
        x = torch.cat((mov, fix), dim=1)
        if self.if_convskip:
            x_s1 = self.avg_pool(x)
            f3 = self.c1(x_s1)
            f4 = self.c2(x)
        else:
            f3 = None
            f4 = None
        
        if self.swin_type == 'swin':
            out_feats = self.transformer(x)
        else:
            out_feats = self.transformer((mov, fix))

        if self.if_transskip:
            if self.swin_type == 'swin':
                f1 = out_feats[-2]
                f2 = out_feats[-3]
            else:
                mov_f1, fix_f1 = out_feats[-2]
                f1 = (mov_f1 + fix_f1)
                mov_f2, fix_f2 = out_feats[-3]
                f2 = (mov_f2 + fix_f2)
        else:
            f1 = None
            f2 = None
            f3 = None
        if self.swin_type == 'swin':
            f0 = out_feats[-1]
        else:
            mov_f0, fix_f0 = out_feats[-1]
            f0 = (mov_f0 + fix_f0)
        x = self.up0(f0, f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        flow = self.reg_head(x)
        
        if self.SVF:
            flow = self.vec_int(flow)
            flow = torch.clamp(flow, -100, 100)
        
        return flow

class TransMorphTVF(nn.Module):
    ''' 
    TransMorph TVF
    Args:
        config: Configuration object containing model parameters
        time_steps: Number of time steps for progressive registration
        SVF: Boolean indicating whether to use SVF (Time Stationary Velocity Field) integration
        SVF_steps: Number of steps for SVF integration
        composition: Type of composition for flow integration ('composition' or 'addition')
        swin_type: Type of Swin Transformer to use ('swin' or 'dswin')
    '''
    def __init__(self, config, time_steps=12, SVF=False, SVF_steps=7, composition='composition', swin_type='swin'):
        super(TransMorphTVF, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.time_steps = time_steps
        self.img_size = config.img_size
        self.composition = composition
        if swin_type == 'swin':
            self.transformer = swin.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        elif swin_type == 'dswin':
            self.transformer = dswin.DefSwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           dwin_size=config.dwin_kernel_size,
                                           img_size=config.img_size,
                                           )
        elif swin_type == 'dswinv2':
            self.transformer = dswin_v2.DefSwinTransformerV2(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           dwin_size=config.dwin_kernel_size,
                                           img_size=config.img_size,
                                           )
        else:
            raise ValueError(f'Unknown Transformer type: {swin_type}')
        self.swin_type = swin_type
        self.up0 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.c1 = Conv3dReLU(2, embed_dim//2, 3, 1)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.reg_heads = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.cs = nn.ModuleList()
        for t in range(self.time_steps):
            self.cs.append(Conv3dReLU(2, embed_dim // 2, 3, 1))
            self.reg_heads.append(RegistrationHead(in_channels=config.reg_head_chan, out_channels=3, kernel_size=3, ))
            self.up3s.append(DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=embed_dim // 2 if if_convskip else 0,
                                   use_batchnorm=False))
        self.spatial_trans_ = utils.SpatialTransformer(config.img_size)
        self.spatial_trans = utils.SpatialTransformer((config.img_size[0]*2, config.img_size[1]*2, config.img_size[2]*2))
        self.upsamp = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.SVF = SVF
        if self.SVF:
            self.vec_int = utils.VecInt(config.img_size, SVF_steps)
        self.eps = 1e-6

    def forward(self, inputs):
        '''
        Forward pass for the TransMorphTVF model.
        Args:
            inputs: Tuple of moving and fixed images (mov, fix).
        Returns:
            flow: The computed flow field for image registration.
        '''
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        x_s1 = self.avg_pool(x_cat)
        if self.swin_type == 'swin':
            out_feats = self.transformer(x_cat)
        else:
            out_feats = self.transformer((mov, fix))
        if self.if_convskip:
            f3 = self.c1(x_s1)
        else:
            f3 = None
        if self.if_transskip:
            if self.swin_type == 'swin':
                f1 = out_feats[-2]
                f2 = out_feats[-3]
            else:
                mov_f1, fix_f1 = out_feats[-2]
                f1 = (mov_f1 + fix_f1)
                mov_f2, fix_f2 = out_feats[-3]
                f2 = (mov_f2 + fix_f2)
        else:
            f1 = None
            f2 = None
        if self.swin_type == 'swin':
            f0 = out_feats[-1]
        else:
            mov_f0, fix_f0 = out_feats[-1]
            f0 = (mov_f0 + fix_f0)
        x = self.up0(f0, f1)
        x = self.up1(x, f2)
        xx = self.up2(x, f3)
        def_x = mov.clone()
        flow_previous = torch.zeros((mov.shape[0], 3, self.img_size[0], self.img_size[1], self.img_size[2])).to(mov.device)
        flows = []

        # flow integration
        xs = 0
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, fix), dim=1))
            x = self.up3s[t](xx, f_out)
            xs += x
            flow = self.reg_heads[t](x)
            flows.append(flow)
            if self.composition == 'composition':
                flow_new = flow_previous + self.spatial_trans_(flow, flow_previous)
            else:
                # essentially addition
                flow_new = flow_previous + self.spatial_trans_(flow, flow)
            def_x = self.spatial_trans_(mov, flow_new)
            flow_previous = flow_new
        flow = flow_new

        if self.SVF:
            flow = self.vec_int(flow)
            flow = torch.clamp(flow, -100, 100)
        return flow
    
class TransMorphTVFSPR(nn.Module):
    '''TransMorph TVF with Spatially-varying regularization
    Args:
        config: Configuration object containing model parameters
        time_steps: Number of time steps for progressive registration
        SVF: Boolean indicating whether to use SVF (Time Stationary Velocity Field) integration
        SVF_steps: Number of steps for SVF integration
        composition: Type of composition for flow integration ('composition' or 'addition')
        swin_type: Type of Swin Transformer to use ('swin' or 'dswin')
    '''
    def __init__(self, config, SVF=True, time_steps=12, SVF_steps=7, composition='composition', swin_type='swin'):
        '''
        Multi-resolution TransMorph
        '''
        super(TransMorphTVFSPR, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.time_steps = time_steps
        self.img_size = config.img_size
        self.composition = composition
        if swin_type == 'swin':
            self.transformer = swin.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        elif swin_type == 'dswin':
            self.transformer = dswin.DefSwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           dwin_size=config.dwin_kernel_size,
                                           img_size=config.img_size,
                                           )
        elif swin_type == 'dswinv2':
            self.transformer = dswin_v2.DefSwinTransformerV2(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           dwin_size=config.dwin_kernel_size,
                                           img_size=config.img_size,
                                           )
        else:
            raise ValueError(f'Unknown Transformer type: {swin_type}')
        self.swin_type = swin_type
        self.up0 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.c1 = Conv3dReLU(2, embed_dim//2, 3, 1, use_batchnorm=False)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.reg_heads = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.cs = nn.ModuleList()
        for t in range(self.time_steps):
            self.cs.append(Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False))
            self.reg_heads.append(RegistrationHead(in_channels=config.reg_head_chan, out_channels=3, kernel_size=3, ))
            self.up3s.append(DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=embed_dim // 2 if if_convskip else 0,
                                   use_batchnorm=False))
        self.spatial_trans_half = utils.SpatialTransformer(config.img_size)
        self.spatial_trans = utils.SpatialTransformer(
            (config.img_size[0] * 2, config.img_size[1] * 2, config.img_size[2] * 2))
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.SVF = SVF
        if self.SVF:
            self.integrate = utils.VecInt(config.img_size, SVF_steps)
        self.sigma_0 = Conv3dReLU(config.reg_head_chan, embed_dim // 2, 3, 3 // 2,
                                   use_batchnorm=False)
        self.sigma_1 = Conv3dReLU(embed_dim // 2, embed_dim // 2, 3, 3 // 2,
                                   use_batchnorm=False)
        self.sigma_2 = nn.Conv3d(embed_dim // 2, 1, kernel_size=3, padding=1)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.wts_act = nn.Sigmoid()
        self.eps = 1e-6

    def forward(self, inputs):
        '''
        Forward pass for the TransMorphTVFSPR model.
        Args:
            inputs: Tuple of moving and fixed images (mov, fix).
        Returns:
            flow: The computed flow field for image registration.
            x_weight: The spatial weights for regularization.
        '''
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        x_s1 = self.avg_pool(x_cat)
        if self.swin_type == 'swin':
            out_feats = self.transformer(x_cat)
        else:
            out_feats = self.transformer((mov, fix))
        if self.if_convskip:
            f3 = self.c1(x_s1)
        else:
            f3 = None
        if self.if_transskip:
            if self.swin_type == 'swin':
                f1 = out_feats[-2]
                f2 = out_feats[-3]
            else:
                mov_f1, fix_f1 = out_feats[-2]
                f1 = (mov_f1 + fix_f1)
                mov_f2, fix_f2 = out_feats[-3]
                f2 = (mov_f2 + fix_f2)
        else:
            f1 = None
            f2 = None

        if self.swin_type == 'swin':
            f0 = out_feats[-1]
        else:
            mov_f0, fix_f0 = out_feats[-1]
            f0 = (mov_f0 + fix_f0)
        x = self.up0(f0, f1)
        x = self.up1(x, f2)
        xx = self.up2(x, f3)
        def_x = mov.clone()
        flow_previous = torch.zeros((mov.shape[0], 3, self.img_size[0], self.img_size[1], self.img_size[2])).to(mov.device)
        flows = []

        # flow integration
        xs = 0
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, fix), dim=1))
            x = self.up3s[t](xx, f_out)
            xs += x
            flow = self.reg_heads[t](x)
            flows.append(flow)
            if self.composition == 'composition':
                flow_new = flow_previous + self.spatial_trans_half(flow, flow_previous)
            else:
                # essentially addition
                flow_new = flow_previous + self.spatial_trans_half(flow, flow)
            def_x = self.spatial_trans_half(mov, flow_new)
            flow_previous = flow_new
        flow = flow_new

        x_weight = self.sigma_0(xs)
        x_weight = self.sigma_1(x_weight)
        x_weight = self.sigma_2(x_weight)
        x_weight = self.wts_act(x_weight)
        x_weight = torch.clamp(x_weight, self.eps, 1.0)

        if self.SVF:
            pos_flow = self.integrate(flow)
            pos_flow = torch.clamp(pos_flow, -100, 100)
            neg_flow = -flow
            neg_flow = self.integrate(neg_flow)
            neg_flow = torch.clamp(neg_flow, -100, 100)
            return pos_flow, neg_flow, x_weight
        else:
            return flow, x_weight

class TransMorphAffine(nn.Module):
    '''
    TransMorph Affine
    Args:
        config: Configuration object containing model parameters
        swin_type: Type of Swin Transformer to use ('swin' or 'dswin')
    '''
    def __init__(self, config, swin_type):
        super(TransMorphAffine, self).__init__()
        out_size = (config.img_size[0]//16, config.img_size[1]//16, config.img_size[2]//16)
        embed_dim = config.embed_dim
        self.img_size = config.img_size
        if swin_type == 'swin':
            self.transformer = swin.SwinTransformer(patch_size=config.patch_size,
                                            in_chans=config.in_chans,
                                            embed_dim=config.embed_dim,
                                            depths=config.depths,
                                            num_heads=config.num_heads,
                                            window_size=config.window_size,
                                            mlp_ratio=config.mlp_ratio,
                                            qkv_bias=config.qkv_bias,
                                            drop_rate=config.drop_rate,
                                            drop_path_rate=config.drop_path_rate,
                                            ape=config.ape,
                                            spe=config.spe,
                                            rpe=config.rpe,
                                            patch_norm=config.patch_norm,
                                            use_checkpoint=config.use_checkpoint,
                                            out_indices=config.out_indices,
                                            pat_merg_rf=config.pat_merg_rf,
                                            )
        elif swin_type == 'dswin':
            self.transformer = dswin.DefSwinTransformer(patch_size=config.patch_size,
                                            in_chans=config.in_chans,
                                            embed_dim=config.embed_dim,
                                            depths=config.depths,
                                            num_heads=config.num_heads,
                                            window_size=config.window_size,
                                            mlp_ratio=config.mlp_ratio,
                                            qkv_bias=config.qkv_bias,
                                            drop_rate=config.drop_rate,
                                            drop_path_rate=config.drop_path_rate,
                                            ape=config.ape,
                                            spe=config.spe,
                                            rpe=config.rpe,
                                            patch_norm=config.patch_norm,
                                            use_checkpoint=config.use_checkpoint,
                                            out_indices=config.out_indices,
                                            pat_merg_rf=config.pat_merg_rf,
                                            dwin_size=config.dwin_kernel_size,
                                            img_size=config.img_size,
                                            )
        elif swin_type == 'dswinv2':
            self.transformer = dswin_v2.DefSwinTransformerV2(patch_size=config.patch_size,
                                            in_chans=config.in_chans,
                                            embed_dim=config.embed_dim,
                                            depths=config.depths,
                                            num_heads=config.num_heads,
                                            window_size=config.window_size,
                                            mlp_ratio=config.mlp_ratio,
                                            qkv_bias=config.qkv_bias,
                                            drop_rate=config.drop_rate,
                                            drop_path_rate=config.drop_path_rate,
                                            ape=config.ape,
                                            spe=config.spe,
                                            rpe=config.rpe,
                                            patch_norm=config.patch_norm,
                                            use_checkpoint=config.use_checkpoint,
                                            out_indices=config.out_indices,
                                            pat_merg_rf=config.pat_merg_rf,
                                            dwin_size=config.dwin_kernel_size,
                                            img_size=config.img_size,
                                            )
        else:
            raise ValueError(f'Unknown Transformer type: {swin_type}')
        self.swin_type = swin_type
        self.aff_mlp = nn.Sequential()
        aff_head = nn.Linear(embed_dim * 4 * np.prod(out_size), 100)
        self.aff_mlp.append(aff_head)
        relu_aff = nn.LeakyReLU()
        self.aff_mlp.append(relu_aff)
        aff_head_f = nn.Linear(100, 3)
        aff_head_f.weight = nn.Parameter(Normal(0, 1e-3).sample(aff_head_f.weight.shape))
        aff_head_f.bias = nn.Parameter(Normal(0, 1e-3).sample(aff_head_f.bias.shape))
        self.aff_mlp.append(aff_head_f)

        self.scl_mlp = nn.Sequential()
        scl_head = nn.Linear(embed_dim * 4 * np.prod(out_size), 100)
        self.scl_mlp.append(scl_head)
        relu_scl = nn.LeakyReLU()
        self.scl_mlp.append(relu_scl)
        scl_head_f = nn.Linear(100, 3)
        scl_head_f.weight = nn.Parameter(Normal(0, 1e-2).sample(scl_head_f.weight.shape))
        scl_head_f.bias = nn.Parameter(Normal(0, 1e-2).sample(scl_head_f.bias.shape))
        self.scl_mlp.append(scl_head_f)

        self.trans_mlp = nn.Sequential()
        trans_head = nn.Linear(embed_dim * 4 * np.prod(out_size), 100)
        self.trans_mlp.append(trans_head)
        relu_trans = nn.LeakyReLU()
        self.trans_mlp.append(relu_trans)
        trans_head_f = nn.Linear(100, 3)
        trans_head_f.weight = nn.Parameter(Normal(0, 1e-2).sample(trans_head_f.weight.shape))
        trans_head_f.bias = nn.Parameter(Normal(0, 1e-4).sample(trans_head_f.bias.shape))
        self.trans_mlp.append(trans_head_f)

        self.shear_mlp = nn.Sequential()
        shear_head = nn.Linear(embed_dim * 4 * np.prod(out_size), 100)
        self.shear_mlp.append(shear_head)
        relu_shear = nn.LeakyReLU()
        self.shear_mlp.append(relu_shear)
        shear_head_f = nn.Linear(100, 6)
        shear_head_f.weight = nn.Parameter(Normal(0, 1e-2).sample(shear_head_f.weight.shape))
        shear_head_f.bias = nn.Parameter(Normal(0, 1e-4).sample(shear_head_f.bias.shape))
        self.shear_mlp.append(shear_head_f)
        self.inst_norm = nn.InstanceNorm3d(embed_dim * 4)
    def softplus(self, x):  # Softplus
        return torch.log(1 + torch.exp(x))

    def forward(self, inputs):
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        if self.swin_type == 'swin':
            out_feats = self.transformer(x_cat)
            f0 = out_feats[-1]
        else:
            out_feats = self.transformer((mov, fix))
            mov_f0, fix_f0 = out_feats[-1]
            f0 = (mov_f0 + fix_f0)
        
        out = self.inst_norm(f0)
        x5 = torch.flatten(out, start_dim=1)

        aff_ = self.aff_mlp(x5)*0.1
        scl_ = self.scl_mlp(x5)*0.1
        trans_ = self.trans_mlp(x5)*0.1
        shr_ = self.shear_mlp(x5)*0.1
        trans = trans_.clone()
        aff = torch.clamp(aff_, min=-1, max=1) * np.pi
        scl = scl_ + 1
        scl = torch.clamp(scl, min=0, max=5)
        shr = torch.clamp(shr_, min=-1, max=1) * np.pi  # shr = torch.tanh(shr) * np.pi
        return aff, scl, trans, shr#, aff_, scl_, trans_, shr_

CONFIGS = {
    'TransMorph-3-LVL': configs.get_3DTransMorph3Lvl_config(),
    'TransMorph-3-LVL-DWin': configs.get_3DTransMorphDWin3Lvl_config(),
    'TransMorph': configs.get_3DTransMorph_config()
}