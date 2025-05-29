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
from MIR.deformation_regularizer import logBeta, LocalGrad3d, logGaussian

def main():
    num_samples = 1000
    sigma_prime = [0.002, 0.525, 1.416, 0.024]
    sigma_names = ['Whole-body CT', 'Brain MRI', 'Cardiac MRI', 'Lung CT']
    plt.figure(dpi=150)
    for s in range(4):
        logGaussian_sigma = logGaussian()
        weights = torch.linspace(0, 1, num_samples).cuda()
        penalties = []
        for i in range(0, num_samples):
            weight = weights[i].unsqueeze(0)
            penalties.append(logGaussian_sigma(weight, sigma_prime[s]))
        penalties = torch.stack(penalties).detach().cpu().numpy() 
        plt.plot(weights.detach().cpu().numpy(), penalties)
    plt.title('logGaussian penalty')
    plt.xlabel('weights')
    plt.ylabel('penalty')
    plt.legend(sigma_names)
    plt.savefig('logGaussian_penalty.png')
    plt.close()
    
    alpha_prime = [0.005, 0.175, 0.493, 0.062]
    sigma_names = ['Whole-body CT', 'Brain MRI', 'Cardiac MRI', 'Lung CT']
    plt.figure(dpi=150)
    for s in range(4):
        logBeta_alpha = logBeta()
        weights = torch.linspace(0, 1, num_samples).cuda()
        penalties = []
        for i in range(0, num_samples):
            weight = weights[i].unsqueeze(0)
            penalties.append(logBeta_alpha(weight, alpha_prime[s]+1))
        penalties = torch.stack(penalties).detach().cpu().numpy() 
        plt.plot(weights.detach().cpu().numpy(), penalties)
    plt.title('logBeta penalty')
    plt.xlabel('weights')
    plt.ylabel('penalty')
    plt.legend(sigma_names)
    plt.savefig('logBeta_penalty.png')
    plt.close()
    

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