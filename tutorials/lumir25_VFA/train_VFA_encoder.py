from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys, gdown, json
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import SpatialTransformer, SSLHeadNLvl, VFA
from MIR.image_similarity import NCC_vxm, LocalCorrRatio, MIND_loss, NCC_gauss, NCC_mok, CorrRatio, NCC_vfa, FastNCC, NCC_fp16
from MIR.deformation_regularizer import Grad3d, GradICON3d, GradICONExact3d
from MIR.utils import Logger, AverageMeter
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.accuracy_measures import dice_val_VOI
from MIR.utils import mk_grid_img
from MIR import RandomMultiContrastRemap, ModelWeights
import matplotlib, random
matplotlib.use('Agg')
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
import nibabel as nib


class L2RLUMIRJSONDataset(Dataset):
    def __init__(self, base_dir, json_path, stage='train'):
        with open(json_path) as f:
            d = json.load(f)
        if stage.lower()=='train':
            self.imgs = d['training']
        elif stage.lower()=='validation':
            self.imgs = d['validation']
        else:
            self.imgs = d['validation']
        self.base_dir = base_dir
        self.stage = stage
    
    def to_torch(self, x):
        x = x[None, ...]
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        return x
    
    def __getitem__(self, index):
        if self.stage == 'train':
            mov_dict = self.imgs[index]
            fix_dicts = self.imgs.copy()
            fix_dicts.remove(mov_dict)
            random.shuffle(fix_dicts)
            fix_dict = fix_dicts[0]
            x = nib.load(self.base_dir+mov_dict['image'])
            y = nib.load(self.base_dir+fix_dict['image'])

        else:
            img_dict = self.imgs[index]
            mov_path = img_dict['moving']
            fix_path = img_dict['fixed']
            x = nib.load(self.base_dir + mov_path)
            y = nib.load(self.base_dir + fix_path)
        x = x.get_fdata() / 255.
        y = y.get_fdata() / 255.
        x, y = x[None, ...], y[None, ...]
        x = np.ascontiguousarray(x)  # [channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.stage == 'validation':
            path_mov = self.base_dir+'labelsVal/'+mov_path.split('/')[-1]
            path_fix = self.base_dir+'labelsVal/'+fix_path.split('/')[-1]
            x_lbl = nib.load(path_mov)
            x_lbl = self.to_torch(x_lbl.get_fdata())
            y_lbl = nib.load(path_fix)
            y_lbl = self.to_torch(y_lbl.get_fdata())
            return x.float(), y.float(), x_lbl.float(), y_lbl.float()
        else:
            return x.float(), y.float()

    def __len__(self):
        return len(self.imgs)

def main():
    batch_size = 1
    weights = [1, 1] # loss weights
    train_dir = '/scratch2/jchen/DATA/LUMIR_L2R24_Internal/'
    val_dir = '/scratch/jchen/DATA/LUMIR/LUMIR25/'
    save_dir = 'VFA_with_aug_LUMIR25_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
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
    H, W, D = 160, 224, 192
    scale_factor=1
    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    model = VFA(config, device='cuda:0')
    pretrained_dir = 'pretrained_wts/'
    pretrained_wts_mono = 'VFA_LUMIR24.pth'
    if not os.path.isdir("pretrained_wts/"):
        os.makedirs("pretrained_wts/")
    if not os.path.isfile(pretrained_dir+pretrained_wts_mono):
        # download model
        file_id = ModelWeights['VFA-LUMIR24-MonoModal']['wts']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pretrained_dir+pretrained_wts_mono, quiet=False)
    pretrained = torch.load(pretrained_dir)['state_dict']
    model.load_state_dict(pretrained)
    print('model: pretrained.pth.tar loaded!')
    #del sslencoder
    model.cuda()
    '''
    Initialize spatial transformation function
    '''
    reg_model = SpatialTransformer((H, W, D), 'nearest').cuda()
    spatial_trans = SpatialTransformer((H, W, D)).cuda()

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

    random_mri = RandomMultiContrastRemap(p=0.85)
    
    '''
    Initialize training
    '''
    train_set = L2RLUMIRJSONDataset(base_dir=train_dir, json_path=val_dir+'LUMIR25_dataset_wo_landmarks.json', stage='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_set = L2RLUMIRJSONDataset(base_dir=val_dir, json_path=val_dir+'LUMIR25_dataset_wo_landmarks.json', stage='validation')
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    optimizer = optim.AdamW(model.encoder.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    if cont_training:
        opt_state = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['optimizer']
        optimizer.load_state_dict(opt_state)
        print('Optimizer loaded!')

    best_dsc = 0
    criterion_ncc = NCC_vxm()
    criterion_reg = Grad3d(penalty='l2')
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
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            with torch.no_grad():
                x = data[0].cuda()
                y = data[1].cuda()
                x_aug = random_mri(x, x>0)
                y_aug = random_mri(y, y>0)
                x_half = x_aug.cuda()#F.avg_pool3d(x_aug, 2).cuda()
                y_half = y_aug.cuda()#F.avg_pool3d(y_aug, 2).cuda()
            if x_half.max() < 0.2 or y_half.max() < 0.2:
                continue
            flow = model((x_half, y_half))
            output = spatial_trans(x, flow)
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(flow, y) * weights[1]
            loss = loss_ncc + loss_reg
                
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flow = model((y_half, x_half))
            output = spatial_trans(y, flow)
            loss_ncc = criterion_ncc(output, x) * weights[0]
            loss_reg = criterion_reg(flow, x) * weights[1]
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), x.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                            loss.item(),
                                                                                            loss_ncc.item(),
                                                                                            loss_reg.item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_half = x.cuda()
                y_half = y.cuda()
                x_seg = data[2].cuda()
                y_seg = data[3].cuda()
                grid_img = mk_grid_img(8, 1, config.img_size, dim=0).cuda()
                flow = model((x_half, y_half))
                def_out = reg_model(x_seg.cuda().float(), flow.cuda())
                def_grid = spatial_trans(grid_img.float(), flow.cuda())
                dsc = dice_val_VOI(def_out.long(), y_seg.long(), num_clus=10)
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