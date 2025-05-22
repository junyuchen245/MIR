"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import matplotlib
matplotlib.use('Agg')
import os
import logging
import optuna
from optuna.trial import TrialState
import torch.nn.functional as F
import os, glob
import sys, random
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import TransMorphTVFSPR, SpatialTransformer
from MIR.image_similarity import NCC_vxm 
from MIR.deformation_regularizer import logBeta, LocalGrad3d
from MIR.deformation_regularizer import Grad3d
from MIR.utils import Logger, AverageMeter
import MIR.models.configs_TransMorph as configs_TransMorph
from MIR.accuracy_measures import dice_val_VOI

VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
batch_size = 1
atlas_dir = '/scratch2/jchen/DATA/IXI/atlas.pkl'
train_dir = '/scratch2/jchen/DATA/IXI/Train/'
val_dir = '/scratch2/jchen/DATA/IXI/Val/'
save_dir = 'IXI_TransMorphSPR/'
if not os.path.exists('experiments/'+save_dir):
    os.makedirs('experiments/'+save_dir)

if not os.path.exists('logs/'+save_dir):
    os.makedirs('logs/'+save_dir)
sys.stdout = Logger('logs/'+save_dir)

lr = 0.0001 #learning rate
epoch_start = 0
max_epoch = 300 #max traning epoch
cont_training = False #if continue training
'''
Initialize model
'''
H, W, D = 160, 192, 224
trail_idx = 0
def objective(trial):
    global trail_idx
    trail_idx += 1
    # Generate the model.
    config = configs_TransMorph.get_3DTransMorph3Lvl_config()
    config.img_size = (H//2, W//2, D//2)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    model = TransMorphTVFSPR(config, SVF=True)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model_nn = SpatialTransformer((H, W, D), 'nearest')
    reg_model_nn.cuda()

    # Generate the optimizers.
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion_ncc = NCC_vxm()
    criterion_reg = logBeta()
    criterion_reg2 = LocalGrad3d(penalty='l2')
    wt_logBeta = trial.suggest_float("wt_logBeta", 0.01, 0.2)
    wt_ncc = 1
    wt_localGrad = trial.suggest_float("wt_localGrad", 0.5, 5.)
    best_dsc = 0
    print('LogBeta weight: {}, LocalGrad weight: {}'.format(wt_logBeta, wt_localGrad))
    # Training of the model.
    val_dscs = []
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
                x = data[0].cuda().float()
                y = data[1].cuda().float()
                x_half = F.avg_pool3d(x, 2).cuda()
                y_half = F.avg_pool3d(y, 2).cuda()

            smo_weight = 1. + wt_logBeta

            flow, _, wts = model((x_half,y_half))
            wts = F.interpolate(wts.cuda(), scale_factor=2, mode='trilinear', align_corners=False)
            flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            output = model.spatial_trans(x, flow)
            loss_ncc = criterion_ncc(output, y) * wt_ncc
            loss_reg = criterion_reg(wts, smo_weight)
            loss_reg2 = criterion_reg2(flow, wts) * wt_localGrad

            loss = loss_ncc + loss_reg + loss_reg2
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Reg2: {:.6f}'.format(idx, len(train_loader),
                                                                                                 loss.item(),
                                                                                                 loss_ncc.item(),
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
                x = data[0].cuda().float()
                y = data[1].cuda().float()
                x_half = F.avg_pool3d(x, 2).cuda()
                y_half = F.avg_pool3d(y, 2).cuda()
                x_seg = data[2]
                y_seg = data[3]
                flow, _, wts = model((x_half, y_half))
                flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
                def_seg = reg_model_nn(x_seg.cuda().float(), flow.cuda())
                dsc = dice_val_VOI(def_seg.long(), y_seg.long(), eval_labels=VOI_lbls)
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

        accuracy = best_dsc  # eval_dsc.avg
        trial.report(accuracy, epoch)
        val_dscs.append(eval_dsc.avg)
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(range(0, epoch + 1), val_dscs)
        ax1.set_ylim([best_dsc - eval_dsc.std, best_dsc + eval_dsc.std])
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(range(0, epoch + 1), val_dscs)
        ax2.set_ylim([0.5, 0.82])
        plt.savefig('logs/' + save_dir + 'trail_{}.png'.format(trail_idx))
        plt.close()

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

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

    #optuna.delete_study(study_name="IXI-SPR", storage="sqlite:///db.IXI")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(storage="sqlite:///db.IXI", study_name="IXI-SPR", direction="maximize", load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
    #study.enqueue_trial({"wt_localGrad": 3.354403461716447, "wt_logBeta":0.17530050372389117})
    study.optimize(objective, n_trials=20)

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