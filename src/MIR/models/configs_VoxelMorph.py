"""Configuration presets for VoxelMorph models."""

import ml_collections

def get_VXM_1_config():
    """Return VoxelMorph-1 config.

    Returns:
        ml_collections.ConfigDict with model hyperparameters.
    """
    config = ml_collections.ConfigDict()
    config.img_size = (160, 192, 224)
    config.nb_unet_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))
    config.nb_unet_levels = None
    config.unet_feat_mult = 1
    config.use_probs = False
    return config

def get_VXM_default_config():
    """Return default VoxelMorph-2 config.

    Returns:
        ml_collections.ConfigDict with model hyperparameters.
    """
    config = ml_collections.ConfigDict()
    config.img_size = (160, 192, 224)
    config.nb_unet_features = ((16, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16))
    config.nb_unet_levels = None
    config.unet_feat_mult = 1
    config.use_probs = False
    return config

def get_VXM_BJ_config():
    """Return VoxelMorph-BJ config.

    Returns:
        ml_collections.ConfigDict with model hyperparameters.
    """
    config = ml_collections.ConfigDict()
    config.img_size = (160, 192, 224)
    config.nb_unet_features = ((16,32,64,96,128), (128, 128, 96, 64, 32, 32))
    config.nb_unet_levels = None
    config.unet_feat_mult = 1
    config.use_probs = False
    return config