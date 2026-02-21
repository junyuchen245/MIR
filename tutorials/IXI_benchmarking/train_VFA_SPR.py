import os
import sys
import glob

from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from torchvision import transforms

from MIR.models import SpatialTransformer, EncoderFeatureExtractor, SITReg, VFA, TransMorphTVF, TransMorph, VFASPR
from MIR.utils import Logger, AverageMeter, mk_grid_img
from MIR.image_similarity import NCC_vxm
from MIR.deformation_regularizer import LocalGrad3d, logBeta
from MIR.accuracy_measures import dice_val_VOI
import MIR.models.configs_VFA as CONFIGS_VFA

import matplotlib
matplotlib.use('Agg')

from data import datasets, trans

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
INPUT_SHAPE = (160, 192, 224)
ATLAS_DIR = '/scratch2/jchen/DATA/IXI/atlas.pkl'
TRAIN_DIR = '/scratch2/jchen/DATA/IXI/Train/'
VAL_DIR = '/scratch2/jchen/DATA/IXI/Val/'

def to_cuda(batch):
    return [t.cuda(non_blocking=True) for t in batch]

def main():
    batch_size = 1
    weights = [1, 3.35, 0.18] # loss weights
    save_dir = 'VFASPR_ncc_{}_LocalGrad3d_{}_logBeta_{}/'.format(weights[0], weights[1], weights[2])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 #learning rate
    epoch_start = 0
    max_epoch = 250 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    H, W, D = 160, 192, 224
    scale_factor=1
    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    model = VFASPR(config, device='cuda:0', SVF=True)
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
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            
            beta_weight = 1. + weights[2]
            
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            x, y = (t.float() for t in to_cuda(data[:2]))
            flow, spatial_wts = model((x,y))
            output = spatial_trans(x, flow)
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(spatial_wts, weights[1])
            loss_reg2 = criterion_reg2(flow, spatial_wts) * beta_weight
            loss = loss_ncc + loss_reg + loss_reg2
                
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flow, spatial_wts = model((y,x))
            output = spatial_trans(y, flow)
            loss_ncc = criterion_ncc(output, x) * weights[0]
            loss_reg = criterion_reg(spatial_wts, weights[1])
            loss_reg2 = criterion_reg2(flow, spatial_wts) * beta_weight
            loss = loss_ncc + loss_reg + loss_reg2
            loss_all.update(loss.item(), x.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Reg2: {:.6f}'.format(idx, len(train_loader),
                                                                                            loss.item(),
                                                                                            loss_ncc.item(),
                                                                                            loss_reg.item(),
                                                                                            loss_reg2.item()))

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
                flow, spatial_wts = model((x,y))
                def_out = spatial_trans_nn(x_seg.cuda().float(), flow)
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
        loss_all.reset()

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