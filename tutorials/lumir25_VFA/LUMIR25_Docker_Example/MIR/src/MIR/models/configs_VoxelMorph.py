import ml_collections

def get_VXM_1_config():
    '''
    VoxelMorph-1 config.
    config.img_size: tuple, size of the input image.
    config.nb_unet_features: tuple, number of features in the encoder and decoder.
    '''
    config = ml_collections.ConfigDict()
    config.img_size = (160, 192, 224)
    config.nb_unet_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))
    config.nb_unet_levels = None
    config.unet_feat_mult = 1
    config.use_probs = False
    return config

def get_VXM_default_config():
    '''
    VoxelMorph-2 config.
    config.img_size: tuple, size of the input image.
    config.nb_unet_features: tuple, number of features in the encoder and decoder.
    '''
    config = ml_collections.ConfigDict()
    config.img_size = (160, 192, 224)
    config.nb_unet_features = ((16, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16))
    config.nb_unet_levels = None
    config.unet_feat_mult = 1
    config.use_probs = False
    return config

def get_VXM_BJ_config():
    '''
    VoxelMorph_BJ config.
    config.img_size: tuple, size of the input image.
    config.nb_unet_features: tuple, number of features in the encoder and decoder.
    '''
    config = ml_collections.ConfigDict()
    config.img_size = (160, 192, 224)
    config.nb_unet_features = ((16,32,64,96,128), (128, 128, 96, 64, 32, 32))
    config.nb_unet_levels = None
    config.unet_feat_mult = 1
    config.use_probs = False
    return config