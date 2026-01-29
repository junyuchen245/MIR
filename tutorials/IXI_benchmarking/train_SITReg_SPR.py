from torch.utils.tensorboard import SummaryWriter
import os
import sys
import glob

from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from torchvision import transforms

from MIR.models import SpatialTransformer, EncoderFeatureExtractor, SITReg
from MIR.models.SITReg import ReLUFactory, GroupNormalizerFactory
from MIR.models.SITReg.composable_mapping import DataFormat
from MIR.models.SITReg.deformation_inversion_layer.fixed_point_iteration import (
    AndersonSolver,
    AndersonSolverArguments,
    MaxElementWiseAbsStopCriterion,
    RelativeL2ErrorStopCriterion,
)
from MIR.utils import Logger, AverageMeter, mk_grid_img
from MIR.image_similarity import NCC_vxm
from MIR.deformation_regularizer import LocalGrad3d, logBeta
from MIR.accuracy_measures import dice_val_VOI

import matplotlib
matplotlib.use('Agg')

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "IXI_HyperTransMorph")
sys.path.append(os.path.abspath(DATA_ROOT))
from data import datasets, trans

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
INPUT_SHAPE = (160, 192, 224)
SPR_IN_CHANNELS = 12
ATLAS_DIR = '/scratch2/jchen/DATA/IXI/atlas.pkl'
TRAIN_DIR = '/scratch2/jchen/DATA/IXI/Train/'
VAL_DIR = '/scratch2/jchen/DATA/IXI/Val/'

def mapping_to_flow(mapping, data_format=DataFormat.voxel_displacements()):
    """Convert a SITReg mapping to a displacement field tensor."""
    return mapping.sample(data_format=data_format).generate_values()


class SPRHead(nn.Module):
    """Spatially varying regularization head for SITReg.

    Args:
        in_channels: Number of channels in the feature maps.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(in_channels, affine=True)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()
        self.eps = 1e-6

    def forward(self, mov_feat: torch.Tensor, fix_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat((mov_feat, fix_feat), dim=1)
        x = self.act(self.norm1(self.conv1(x)))
        wts = self.out_act(self.conv2(x))
        return torch.clamp(wts, self.eps, 1.0)

def to_cuda(batch):
    return [t.cuda(non_blocking=True) for t in batch]

def create_model():
    """Create SITReg model and SPR head from config."""
    feature_extractor = EncoderFeatureExtractor(
            n_input_channels=1,
            activation_factory=ReLUFactory(),
            n_features_per_resolution=[12, 16, 32, 64, 128, 128],
            n_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
            input_shape=INPUT_SHAPE,
            normalizer_factory=GroupNormalizerFactory(2),
        ).cuda()
    AndersonSolver_forward = AndersonSolver(
        MaxElementWiseAbsStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
        AndersonSolverArguments(memory_length=4),
    )
    AndersonSolver_backward = AndersonSolver(
        RelativeL2ErrorStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
        AndersonSolverArguments(memory_length=4),
    )
    network = SITReg(
        feature_extractor=feature_extractor,
        n_transformation_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
        n_transformation_features_per_resolution=[12, 64, 128, 256, 256, 256],
        max_control_point_multiplier=0.99,
        affine_transformation_type=None,
        input_voxel_size=(1.0, 1.0, 1.0),
        input_shape=INPUT_SHAPE,
        transformation_downsampling_factor=(1.0, 1.0, 1.0),
        forward_fixed_point_solver=AndersonSolver_forward,
        backward_fixed_point_solver=AndersonSolver_backward,
        activation_factory=ReLUFactory(),
        normalizer_factory=GroupNormalizerFactory(4),
            ).cuda()
    spr_head = SPRHead(SPR_IN_CHANNELS).cuda()
    return network, feature_extractor, spr_head

def main():
    batch_size = 1
    weights = [1, 3.35, 0.18] # loss weights
    save_dir = 'SITRegSPR_IXI_ncc_{}_LocalGrad3d_{}_logBeta_{}/'.format(weights[0], weights[1], weights[2])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 #learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    H, W, D = INPUT_SHAPE
    model, feature_extractor, spr_head = create_model()
    model.cuda()
    '''
    Initialize spatial transformation function
    '''
    spatial_trans = SpatialTransformer((H, W, D)).cuda()
    spatial_trans_nn = SpatialTransformer((H, W, D), 'nearest').cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 85
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([
        trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32)),
    ])
    val_composed = transforms.Compose([
        trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16)),
    ])
    train_set = datasets.IXIBrainDataset(glob.glob(TRAIN_DIR + '*.pkl'), ATLAS_DIR, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(VAL_DIR + '*.pkl'), ATLAS_DIR, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=updated_lr,
        weight_decay=0,
        amsgrad=True,
        foreach=False,
    )
    if cont_training:
        opt_state = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['optimizer']
        optimizer.load_state_dict(opt_state)
        print('Optimizer loaded!')

    best_dsc = 0
    criterion_ncc = NCC_vxm()
    criterion_reg = logBeta()
    criterion_reg2 = LocalGrad3d(penalty='l2')
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            
            beta_weight = 1. + weights[2]
            
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            x, y = (t.float() for t in to_cuda(data[:2]))
            mapping_pair = model(x, y, mappings_for_levels=((0, False),))[0]
            flow = mapping_to_flow(mapping_pair.forward_mapping)
            output = spatial_trans(x, flow)
            features = feature_extractor((x, y))
            mov_feat, fix_feat = torch.chunk(features[0], 2, dim=0)
            spatial_wts = spr_head(mov_feat, fix_feat)
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(spatial_wts, weights[1])
            loss_reg2 = criterion_reg2(flow, spatial_wts) * beta_weight
            loss = loss_ncc + loss_reg + loss_reg2
                
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mapping_pair = model(y, x, mappings_for_levels=((0, False),))[0]
            flow = mapping_to_flow(mapping_pair.forward_mapping)
            output = spatial_trans(y, flow)
            loss_ncc = criterion_ncc(output, x) * weights[0]
            features = feature_extractor((y, x))
            mov_feat, fix_feat = torch.chunk(features[0], 2, dim=0)
            spatial_wts = spr_head(fix_feat, mov_feat)
            loss_ncc = criterion_ncc(output, x) * weights[0]
            loss_reg = criterion_reg(spatial_wts, weights[1])
            loss_reg2 = criterion_reg2(flow, spatial_wts) * beta_weight
            loss = loss_ncc + loss_reg + loss_reg2
            loss_all.update(loss.item(), x.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #plt.figure()
            #plt.subplot(1, 3, 1)
            #plt.imshow(x_aug.cpu().detach().numpy()[0, 0, :, 120, :], vmax=1, vmin=0, cmap='gray')
            #plt.subplot(1, 3, 2)
            #plt.imshow(y_aug.cpu().detach().numpy()[0, 0, :, 120, :], vmax=1, vmin=0, cmap='gray')
            #plt.subplot(1, 3, 3)
            #plt.imshow(output.cpu().detach().numpy()[0, 0, :, 120, :], vmax=1, vmin=0, cmap='gray')
            #plt.savefig('tmp.png')
            #plt.close()


            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Reg2: {:.6f}'.format(idx, len(train_loader),
                                                                                            loss.item(),
                                                                                            loss_ncc.item(),
                                                                                            loss_reg.item(),
                                                                                            loss_reg2.item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                x, y, x_seg, y_seg = to_cuda(data)
                x = x.float()
                y = y.float()
                grid_img = mk_grid_img(8, 1, (H, W, D), dim=0).cuda()
                mapping_pair = model(x, y, mappings_for_levels=((0, False),))[0]
                flow = mapping_to_flow(mapping_pair.forward_mapping)
                #flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
                def_out = spatial_trans_nn(x_seg.cuda().float(), flow)
                def_grid = spatial_trans(grid_img.float(), flow)
                dsc = dice_val_VOI(def_out.long(), y_seg.long(), eval_labels=VOI_lbls)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
        plt.switch_backend('agg')
        pred_fig = comput_fig(flow)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x)
        tar_fig = comput_fig(y)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('moving', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('fixed', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('deformed', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=2):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main()