import torch
import torch.nn as nn
import torch.nn.functional as nnf
import MIR.models.Swin_Transformer as swin
import MIR.models.Deformable_Swin_Transformer as dswin
import MIR.models as mir_m
from MIR.models.VFA import Decoder, DoubleConv3d, grid_to_flow

class Feature2VFA(nn.Module):
    '''
    Feature converter module to convert Transformer features for VFA
    Args:
        configs_sw: Swin Transformer configs
        vfa_channels: Number of channels for VFA
        embed_dim: Embedding dimension for Swin Transformer
        swin_type: Type of Swin Transformer ('swin' or 'dswin')
        in_norm: Whether to apply instance normalization to the input
    '''
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
        else:
            raise ValueError(f'Unknown Transformer type: {swin_type}')
        if in_norm:
            self.norm_first = nn.InstanceNorm3d(1, affine=True)
        self.in_norm = in_norm
        self.c1 = DoubleConv3d(
                in_channels=1,
                mid_channels=vfa_channels[0]//4,
                out_channels=vfa_channels[0]
            )
        self.c2 = DoubleConv3d(
                in_channels=1,
                mid_channels=vfa_channels[1]//4,
                out_channels=vfa_channels[1]
            )
        self.cs = nn.ModuleList()
        for i in range(2, len(vfa_channels)):
            self.cs.append(mir_m.Conv3dReLU(embed_dim*2**(i-2), vfa_channels[i], 1, 0, use_batchnorm=False))
        self.vfa_channels = vfa_channels
        self.swin_type = swin_type
        
    def forward(self, x):
        if self.swin_type == 'swin':
            if self.in_norm:
                x = self.norm_first(x)
            out_features = [self.c1(x), self.c2(nnf.avg_pool3d(x, 2))]
            trans_feats = self.encoder(x)
            for i in range(2, len(self.vfa_channels)):
                feat = self.cs[i-2](trans_feats[i-2])
                out_features.append(feat)
            return out_features
        else:
            if self.in_norm:
                x = [self.norm_first(x[0]), self.norm_first(x[1])]
            out_features_m = [self.c1(x[0]), self.c2(nnf.avg_pool3d(x[0], 2))]
            out_features_f = [self.c1(x[1]), self.c2(nnf.avg_pool3d(x[1], 2))]
            trans_feats = self.encoder(x)
            for i in range(2, len(self.vfa_channels)):
                feat_m = self.cs[i-2](trans_feats[i-2][0])
                feat_f = self.cs[i-2](trans_feats[i-2][1])
                out_features_m.append(feat_m)
                out_features_f.append(feat_f)
            return out_features_m, out_features_f
        

class TransVFA(nn.Module):
    '''
    TransVFA model for image registration
    Args:
        configs_sw: Swin Transformer configs
        configs: VFA configs
        device: Device to run the model on
        swin_type: Type of Swin Transformer ('swin' or 'dswin')
        return_orginal: Whether to return the original deformation field used for original VFA, otherwise return the flow as displacement
        return_all_flows: Whether to return all flows
    '''
    def __init__(self, configs_sw, configs, device, swin_type='swin', return_orginal=False, return_all_flows=False):
        super().__init__()

        self.dim = len(configs.img_size)
        channels = [min(configs.start_channels * 2**(i+1), 64) for i in range(configs.downsamples+1)]
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