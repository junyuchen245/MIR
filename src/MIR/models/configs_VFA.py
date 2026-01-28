"""Configuration presets for VFA models."""

import ml_collections

def get_VFA_default_config():
    '''
    VFA default config.
    config.name: str, name of the model.
    config.skip: int, skip certain displacement fields in the decoder.
    config.initialize: float, initialize beta in the decoder.
    config.downsamples: int, number of downsampling layers in the encoder.
    config.start_channels: int, number of channels in the first layer of the encoder.
    config.matching_channels: int, number of channels in the matching layer of the encoder.
    config.int_steps: int, number of integration steps in the decoder.
    config.affine: int, whether to use affine transformation in the decoder.
    config.img_size: tuple, size of the input image.
    config.in_channels: int, number of input channels.
    config.max_channels: int, maximum number of channels in the encoder.
    '''
    config = ml_collections.ConfigDict()
    config.name ="VFA"
    config.skip = 0
    config.initialize = 0.1
    config.downsamples = 4
    config.start_channels = 8
    config.matching_channels = 8
    config.int_steps = 0
    config.affine = 0
    config.img_size = (160, 192, 224)
    config.in_channels = 1
    config.max_channels = 64
    return config