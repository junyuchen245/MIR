"""Configuration presets for LessNet models."""

import ml_collections


def get_LessNet_default_config():
    """Return the upstream LessNet README/default reproduction config.

    The original README uses:
    `--start_channel 8 --using_l2 2 --smth_labda 5.0 --lr 1e-4`.
    """
    config = ml_collections.ConfigDict()
    config.name = 'LessNet'
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
    config.smth_labda = 5.0
    config.lr = 1e-4
    config.bs = 1
    config.iteration = 201501
    config.checkpoint = 403
    config.trainingset = 4
    return config


def get_LessNet_SVF_config():
    """Return LessNet config with SVF integration enabled."""
    config = get_LessNet_default_config()
    config.name = 'LessNetSVF'
    config.SVF = True
    config.smth_labda = 2.0
    return config


def get_LessNet_SPR_config():
    """Return the default LessNetSPR config."""
    config = get_LessNet_default_config()
    config.name = 'LessNetSPR'
    config.SVF = True
    config.spr = True
    config.smth_labda = 2.0
    return config
