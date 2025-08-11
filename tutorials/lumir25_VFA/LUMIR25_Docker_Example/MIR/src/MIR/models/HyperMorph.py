'''
HyperMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:

Hoopes, Andrew, et al. 
"Hypermorph: Amortized hyperparameter learning for image registration." 
Information Processing in Medical Imaging: 27th International Conference, 
IPMI 2021, Virtual Event, June 28–June 30, 2021, Proceedings 27. 
Springer International Publishing, 2021.

Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal
import MIR.models.registration_utils as utils

class HyperBlocks(nn.Module):
    def __init__(self, nb_hyp_params=1, nb_hyp_layers=6, nb_hyp_units=128):
        super().__init__()
        self.fcs = nn.ModuleList()
        hyp_last = nb_hyp_params
        for _ in range(nb_hyp_layers):
            self.fcs.append(nn.Linear(hyp_last, nb_hyp_units))
            hyp_last = nb_hyp_units

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
            x = torch.relu(x)
        return x

class ConvBlock_(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, nb_hyp_units=128):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1, bias=False)
        self.activation = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(nb_hyp_units, out_channels)

    def forward(self, x, hyp_feat):
        out = self.main(x)
        bias = self.linear(hyp_feat).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        out = self.activation(out+bias)
        return out

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, nb_hyp_units=128):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        #self.linear = nn.Linear(nb_hyp_units, nb_hyp_units)
        self.activation = nn.LeakyReLU(0.2)
        self.linear_conv = nn.Linear(nb_hyp_units, 3**ndims*out_channels*in_channels, bias=False)
        self.linear_bias = nn.Linear(nb_hyp_units, out_channels, bias=False)
        #self.linear_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_conv.weight.shape))
        #self.linear_bias.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_bias.weight.shape))
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x, hyp_feat):
        #hyp_feat = self.linear(hyp_feat)
        #hyp_feat = torch.relu(hyp_feat)
        kernel = self.linear_conv(hyp_feat).reshape([self.out_channels, self.in_channels, 3, 3, 3])
        #self.main.weight.data = kernel
        #out = self.main(x)
        out = nnf.conv3d(x, kernel, stride=self.stride, padding=1)
        bias= self.linear_bias(hyp_feat).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        out = self.activation(out+bias)
        return out

class CustomConv(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, nb_hyp_units=128):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        #self.linear = nn.Linear(nb_hyp_units, nb_hyp_units)
        self.linear_conv = nn.Linear(nb_hyp_units, 3**ndims*out_channels*in_channels, bias=False)
        self.linear_bias = nn.Linear(nb_hyp_units, out_channels, bias=False)
        self.linear_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_conv.weight.shape))
        self.linear_bias.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_bias.weight.shape))
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x, hyp_feat):
        #out = self.main(x)
        #hyp_feat = self.linear(hyp_feat)
        #hyp_feat = torch.relu(hyp_feat)
        kernel = self.linear_conv(hyp_feat).reshape([self.out_channels, self.in_channels, 3, 3, 3])
        out = nnf.conv3d(x, kernel, stride=self.stride, padding=1)
        bias= self.linear_bias(hyp_feat).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        out = out+bias
        return out

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x, h):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1], h))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x, h)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x, h)

        return x
        
class HyperVxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        configs,
        gen_output=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()
        inshape = configs.img_size
        nb_unet_features= configs.nb_unet_features
        nb_unet_levels= configs.nb_unet_levels
        unet_feat_mult= configs.unet_feat_mult
        use_probs= configs.use_probs
        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.hyper_model = HyperBlocks()

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = CustomConv(ndims, self.unet_model.dec_nf[-1], ndims)

        # init flow layer with small weights and bias
        #self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        #self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')
        self.gen_output = gen_output
        if self.gen_output:
            # configure transformer
            self.spatial_trans = utils.SpatialTransformer(inshape)

    def forward(self, input_imgs, hyp_val):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        mov, fix = input_imgs
        x = torch.cat((mov, fix), dim=1)
        h = self.hyper_model(hyp_val)

        # concatenate inputs and propagate unet
        x = self.unet_model(x, h)

        # transform into flow field
        flow_field = self.flow(x, h)

        # resize flow for integration
        pos_flow = flow_field
        if self.gen_output:
            # warp image with flow field
            def_x = self.spatial_trans(mov, pos_flow)
            return def_x, pos_flow
        else:
            return pos_flow