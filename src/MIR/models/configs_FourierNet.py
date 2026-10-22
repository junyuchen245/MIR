"""Configuration presets for Fourier-Net models."""

import ml_collections


def get_FourierNet_default_config():
    """Return the upstream 3-D Fourier-Net reproduction config.

    The upstream README uses:
    `--start_channel 8 --using_l2 2 --smth_labda 1.0 --lr 1e-4`.
    """
    config = ml_collections.ConfigDict()
    config.name = 'FourierNet'
    config.img_size = (160, 192, 224)
    config.image_channels = 1
    config.in_channels = 1
    config.out_channels = 3
    config.n_dims = 3
    config.start_channel = 8
    config.SVF = False
    config.SVF_steps = 7
    config.using_l2 = 2
    config.similarity_loss = 'NCC_mok'
    config.smoothness_loss = 'Grad3d'
    config.smoothness_penalty = 'l2'
    config.smth_labda = 1.0
    config.lr = 1e-4
    config.bs = 1
    config.iteration = 403001
    config.checkpoint = 403
    config.trainingset = 4
    return config


def get_FourierNet_SVF_config():
    """Return the upstream 3-D Fourier-Net-Diff style config."""
    config = get_FourierNet_default_config()
    config.name = 'FourierNetSVF'
    config.SVF = True
    config.smth_labda = 5.0
    return config


def get_FourierNet_SPR_config():
    """Return Fourier-Net with SPR/SVF defaults."""
    config = get_FourierNet_SVF_config()
    config.name = 'FourierNetSPR'
    config.spr = True
    return config


def get_FourierNetPlus_default_config():
    """Return the upstream 3-D Fourier-Net+ checkpoint/inference config."""
    config = get_FourierNet_default_config()
    config.name = 'FourierNetPlus'
    config.start_channel = 16
    config.low_img_size = (80, 96, 112)
    config.smth_labda = 5.0
    return config


def get_FourierNetPlus_SVF_config():
    """Return Fourier-Net+ with SVF integration enabled."""
    config = get_FourierNetPlus_default_config()
    config.name = 'FourierNetPlusSVF'
    config.SVF = True
    return config


def get_FourierNetPlus_SPR_config():
    """Return Fourier-Net+ with SPR/SVF defaults."""
    config = get_FourierNetPlus_SVF_config()
    config.name = 'FourierNetPlusSPR'
    config.spr = True
    return config
