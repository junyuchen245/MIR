import torch
import torch.nn as nn
import MIR.models.registration_utils as utils

class SSL_blocks(nn.Module):
    def __init__(self, dim, out_dim, scale_fac):
        super(SSL_blocks, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_dim),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=scale_fac, mode="trilinear", align_corners=False),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_dim),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        x = self.block(x)
        return x

class flow_blocks(nn.Module):
    def __init__(self, dim, if_KL=True):
        super(flow_blocks, self).__init__()
        self.if_KL = if_KL
        self.flow_mean = nn.Conv3d(dim, 3, kernel_size=3, stride=1, padding=1)
        self.flow_mean.weight.data.normal_(0., 1e-5)
        self.flow_mean.bias.data = nn.Parameter(torch.zeros(self.flow_mean.bias.shape))
        if self.if_KL:
            self.flow_sigma = nn.Conv3d(dim, 3, kernel_size=3, stride=1, padding=1)
            self.flow_sigma.weight.data.normal_(0., 1e-10)
            self.flow_sigma.bias.data = torch.Tensor([-10] * 3)
    def forward(self, x):
        mean = self.flow_mean(x)
        if self.if_KL:
            sigma = self.flow_sigma(x)
            return mean, sigma
        else:
            return mean

class SSLHeadNLvl(nn.Module):
    '''
    Self-supervised learning head with multiple levels
    Args:
        encoder: Encoder model
        img_size: Image size
        num_lvls: Number of levels
        channels: Number of channels
        if_upsamp: Whether to upsample
        encoder_output_type: Type of encoder output ('single' or 'multi')
        encoder_input_type: Type of encoder input ('single', 'multi', or 'separate')
        swap_encoder_order: Whether to swap encoder's output order
        gen_output: Whether to generate deformed output
    '''
    def __init__(self, encoder, img_size=(128, 128, 128), num_lvls=3, channels=(96*4, 96*2, 96), if_upsamp=True, encoder_output_type='single', encoder_input_type='single', swap_encoder_order=True, gen_output=True):
        super(SSLHeadNLvl, self).__init__()
        self.num_lvls = num_lvls
        self.encoder = encoder
        self.decoder_blocks = nn.ModuleList()
        self.flow_heads = nn.ModuleList()
        for i in range(num_lvls):
            self.decoder_blocks.append(SSL_blocks(channels[i], 16, 2**(num_lvls-i)))
            self.flow_heads.append(flow_blocks(16))
        self.flow_block_final = flow_blocks(16)
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.if_upsamp = if_upsamp
        self.spatial_trans = utils.SpatialTransformer(img_size)
        self.encoder_output_type = encoder_output_type
        self.encoder_input_type = encoder_input_type
        self.swap_encoder_order = swap_encoder_order
        self.gen_output = gen_output

    def forward(self, inputs):
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        if self.encoder_input_type == 'multi':
            x_out = self.encoder((mov, fix))
        elif self.encoder_input_type == 'separate':
            x_mov = self.encoder(mov)
            x_fix = self.encoder(fix)
            x_out = [[x_mov[i], x_fix[i]] for i in range(len(x_mov))]
        else:
            x_out = self.encoder(x_cat)
        if self.swap_encoder_order:
            x_out = x_out[::-1]
        if self.encoder_output_type == 'multi':
            x_out = [x_out[i][0]+x_out[i][1] for i in range(len(x_out))]
        stats = []
        flows = []
        x_sum = 0
        for i in range(self.num_lvls):
            x = self.decoder_blocks[i](x_out[i])
            x_sum = x_sum + x
            mean, std = self.flow_heads[i](x)
            stats.append((mean, std))
            noise = torch.randn(mean.shape).cuda()
            if self.training:
                mean = mean + torch.exp(std / 2.0) * noise
            if self.if_upsamp:
                flow = self.up_sample(mean) * 2
            else:
                flow = mean
            flows.append(flow)
        mean_final, std_final = self.flow_block_final(x_sum)
        stats.append((mean_final, std_final))
        noise = torch.randn(mean_final.shape).cuda()
        if self.training:
            mean_final = mean_final + torch.exp(std_final / 2.0) * noise
        if self.if_upsamp:
            flow_final = self.up_sample(mean_final) * 2
        else:
            flow_final = mean_final
        flows.append(flow_final)
        if self.gen_output:
            out = self.spatial_trans(mov, flow_final)
            return out, flow_final, stats
        else:
            return flow_final, stats
    
class SSLHead1Lvl(nn.Module):
    '''
    Self-supervised learning head with one level
    Args:
        encoder: Encoder model
        img_size: Image size
        num_lvls: Number of levels
        channels: Number of channels
        if_upsamp: Whether to upsample
        encoder_output_type: Type of encoder output ('single' or 'multi')
        encoder_input_type: Type of encoder input ('single', 'multi', or 'separate')
        swap_encoder_order: Whether to swap encoder's output order
        gen_output: Whether to generate deformed output
    '''
    def __init__(self, encoder, img_size=(128, 128, 128), num_lvls=3, channels=(96*4, 96*2, 96), if_upsamp=True, encoder_output_type='single', swap_encoder_order=True, gen_output=True):
        super(SSLHead1Lvl, self).__init__()
        self.num_lvls = num_lvls
        self.encoder = encoder
        self.decoder_blocks = nn.ModuleList()
        self.flow_heads = nn.ModuleList()
        self.decoder_blocks.append(SSL_blocks(channels[-1], 16, 2))
        self.flow_heads.append(flow_blocks(16, if_KL=False))
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.if_upsamp = if_upsamp
        self.encoder_output_type = encoder_output_type
        self.spatial_trans = utils.SpatialTransformer(img_size)
        self.swap_encoder_order = swap_encoder_order
        self.gen_output = gen_output

    def forward(self, inputs):
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        if self.encoder_output_type == 'multi':
            x_out = self.encoder((mov, fix))
        else:
            x_out = self.encoder(x_cat)
        if self.swap_encoder_order:
            x_out = x_out[::-1]
        if self.encoder_output_type == 'multi':
            x_out = [x_out[i][0]+x_out[i][1] for i in range(len(x_out))]
        x_sum = 0
        i = -1
        x = self.decoder_blocks[i](x_out[i])
        x_sum = x_sum + x
        mean = self.flow_heads[i](x)
        if self.training:
            flow = mean
        if self.if_upsamp:
            flow = self.up_sample(mean) * 2
        else:
            flow = mean
        if self.gen_output:
            out = self.spatial_trans(mov, flow)
            return out, flow
        else:
            return flow