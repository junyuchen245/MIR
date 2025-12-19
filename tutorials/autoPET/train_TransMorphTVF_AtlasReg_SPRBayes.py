"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import logging
import optuna
from optuna.trial import TrialState
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys, random
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import TransMorphTVFSPR, SpatialTransformer
import MIR.models.configs_TransMorph as configs_TransMorph
import nibabel as nib
from MIR.image_similarity import NCC
from MIR.deformation_regularizer import logBeta, logGaussian, LocalGrad3d 
from torch.utils.data import Dataset
from MIR.accuracy_measures import dice_val_VOI
from MIR.utils import Logger, AverageMeter

class AutoPETTrainDataset(Dataset):
    def __init__(self, data_path, data_names):
        self.path = data_path
        self.data_names = data_names

    def norm_img(self, img):
        img[img < -300] = -300
        img[img > 300] = 300
        norm = (img - img.min()) / (img.max() - img.min())
        return norm

    def norm_suv(self, img):
        img_max = 15.#np.percentile(img, 95)
        img_min = 0.#np.percentile(img, 5)
        norm = (img - img_min)/(img_max - img_min)
        norm[norm < 0] = 0
        norm[norm > 1] = 1
        return norm

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def remap_lbl(self, lbl):
        #grouping_table = [[1,], [2, 3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,],
        #                  [15,], [16,], [17,], [18,], [19,], [20,], [21,], [22,], [23,], [24, 25, 26,], [27,],
        #                  [28, 29, 30, 31, 32,], [33, 34,], [35, 36,], [37, 38,], [39,]]
        grouping_table = [[1,], [2,], [3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,], [15, 16, 17,],
                  [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], [42,],
                  [43,], [44,], [45,], [46,], [47,], [48,], [49,], [55,], [56,], [57,], [58, 59, 60, 61, 62, 63, 64, 65,
                                                                                         66, 67, 68, 69, 70, 71, 72, 73,
                                                                                         74, 75, 76, 77, 78, 79, 80, 81,],
                  [82, 84, 86,], [83, 85, 87,], [88, 90,], [89, 91], [92], [94, 96, 98,], [95, 97, 99,], [100,], [101],
                  [102,], [103,], [104,]]
        label_out = np.zeros_like(lbl)

        for idx, item in enumerate(grouping_table):
            for seg_i in item:
                label_out[lbl == seg_i] = idx + 1
        return label_out

    def __getitem__(self, index):
        mov_name = self.data_names[index]
        x = nib.load(self.path + '{}_CTRes.nii.gz'.format(mov_name))
        x = self.norm_img(x.get_fdata())
        x_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(mov_name))
        x_suv = self.norm_suv(x_suv.get_fdata())
        x_seg = nib.load(self.path + '{}_CTRes_seg.nii.gz'.format(mov_name))#_segsimple#seg
        x_seg = x_seg.get_fdata()
        x_seg = self.remap_lbl(x_seg)#Affine reg does not have this.
        x = x[None, ...]
        x_suv = x_suv[None, ...]
        x_seg = x_seg[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv = np.ascontiguousarray(x_suv)  # [Bsize,channelsHeight,,Width,Depth]
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x = torch.from_numpy(x)
        x_seg = torch.from_numpy(x_seg)
        x_suv = torch.from_numpy(x_suv)
        return x, x_suv, x_seg

    def __len__(self):
        return len(self.data_names)

batch_size = 1
train_dir = '/scratch2/jchen/DATA/AutoPET/affine_aligned/network/train/'
val_dir = '/scratch2/jchen/DATA/AutoPET/affine_aligned/network/val/'
atlas_dir = '/scratch/jchen/DATA/AutoPET/'
ct_atlas_dir = '/scratch/jchen/DATA/AutoPET/CT_atlas.nii.gz'
pt_atlas_dir = '/scratch/jchen/DATA/AutoPET/PET_atlas.nii.gz'
save_dir = 'TransMorphSPR/'
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
H, W, D = 192, 192, 256
num_clus = 29
x = nib.load(ct_atlas_dir)
x = x.get_fdata()[None, None, ...]
x = torch.from_numpy(x).cuda().float()
x_suv = nib.load(pt_atlas_dir)
x_suv = x_suv.get_fdata()[None, None, ...]
x_suv = torch.from_numpy(x_suv).cuda().float()
x_seg = nib.load(atlas_dir+'seg_simple_atlas_from_reg.nii.gz')
x_seg = x_seg.get_fdata()[None, None, ...]
x_seg = torch.from_numpy(x_seg).cuda().float()
x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=num_clus)
x_seg_oh = torch.squeeze(x_seg_oh, 1)
x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

def objective(trial):
    # Generate the model.
    config = configs_TransMorph.get_3DTransMorphAutoPET3Lvl_config()
    config.img_size = (H//2, W//2, D//2)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    model = TransMorphTVFSPR(config, time_steps=5)
    model.cuda()

    # Generate the optimizers.
    train_names = glob.glob(train_dir + '*_segsimple*')
    train_names = [name.split('\\')[-1].split('_')[0] for name in train_names]
    val_names = glob.glob(val_dir + '*_segsimple*')
    val_names = [name.split('\\')[-1].split('_')[0] for name in val_names]
    train_set = AutoPETTrainDataset(train_dir, train_names)
    val_set = AutoPETTrainDataset(val_dir, val_names)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion_ncc = NCC()
    criterion_reg = logBeta()
    criterion_reg2 = LocalGrad3d(penalty='l2')
    wt_logBeta = trial.suggest_float("wt_logBeta", 0.001, 0.1)
    wt_ncc = 1
    wt_localGrad = trial.suggest_float("wt_localGrad", 0.8, 3.)
    best_dsc = 0
    print('LogBeta weight: {}, LocalGrad weight: {}'.format(wt_logBeta, wt_localGrad))
    # Training of the model.
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
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            with torch.no_grad():
                y = data[0].cuda().float()
                y_half = F.avg_pool3d(y, 2).cuda()
                y_suv = data[1].cuda().float()
                #y_seg = data[2].cuda().float()

            smo_weight = 1. + wt_logBeta

            x_half = F.avg_pool3d(x, 2).cuda()

            pos_flow, neg_flow, wts = model((x_half, y_half))
            wts = F.interpolate(wts.cuda(), scale_factor=2, mode='trilinear', align_corners=False)
            pos_flow = F.interpolate(pos_flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            neg_flow = F.interpolate(neg_flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2

            x_suv_def = model.spatial_trans(x_suv.float(), pos_flow.float())
            y_suv_def = model.spatial_trans(y_suv.float(), neg_flow.float())

            x_def = model.spatial_trans(x.float(), pos_flow.float())
            y_def = model.spatial_trans(y.float(), neg_flow.float())

            loss_suv = criterion_ncc(x_suv_def, y_suv) * wt_ncc / 4 + criterion_ncc(y_suv_def, x_suv) * wt_ncc / 4
            loss_img = criterion_ncc(x_def, y) * wt_ncc / 4 + criterion_ncc(y_def, x) * wt_ncc / 4
            loss_reg = criterion_reg(wts, smo_weight)
            loss_reg2 = criterion_reg2(pos_flow, wts) * wt_localGrad / 2 + criterion_reg2(neg_flow, wts) * wt_localGrad / 2
            loss = loss_suv + loss_img + loss_reg + loss_reg2
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Suv Sim: {:.6f}, Reg: {:.6f}, Reg2: {:.6f}'.format(idx,
                                                                                                                  len(train_loader),
                                                                                                                  loss.item(),
                                                                                                                  loss_img.item(),
                                                                                                                  loss_suv.item(),
                                                                                                                  loss_reg.item(),
                                                                                                                  loss_reg2.item()))

        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        # Validation of the model.
        model.eval()
        eval_dsc = AverageMeter()
        dsc_raw = []
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                y = data[0].cuda().float()
                x_half = F.avg_pool3d(x, 2).cuda()
                y_half = F.avg_pool3d(y, 2).cuda()
                y_seg = data[2].cuda().float()
                pos_flow, neg_flow, wts = model((x_half, y_half))
                pos_flow = F.interpolate(pos_flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
                def_seg = model.spatial_trans(x_seg_oh.cuda().float(), pos_flow.cuda())
                def_seg = torch.argmax(def_seg, dim=1, keepdim=True)
                dsc = dice_val_VOI((def_seg).long(), y_seg.long(), num_clus)
                if epoch == 0:
                    dsc_raw.append(dice_val_VOI(x_seg.long(), y_seg.long()).item())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        if epoch == 0:
            print('raw dice: {}'.format(np.mean(dsc_raw)))
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
        loss_all.reset()

        accuracy = eval_dsc.avg

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, :, 10:26]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[2]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, :, i], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[1], grid_step):
        grid_img[:, i+line_thickness-1, :] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def seedBasic(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
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
    DEFAULT_RANDOM_SEED = 42

    seedBasic(DEFAULT_RANDOM_SEED)
    seedTorch(DEFAULT_RANDOM_SEED)

    #optuna.delete_study(study_name="TransMorph-SPR", storage="sqlite:///db.TM_SPR")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(storage="sqlite:///db.TM_SPR", study_name="TransMorph-SPR", direction="maximize", load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=10))
    study.optimize(objective, n_trials=45)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))