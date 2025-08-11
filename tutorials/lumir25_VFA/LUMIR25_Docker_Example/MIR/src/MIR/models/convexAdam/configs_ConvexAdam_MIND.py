import ml_collections

def get_ConvexAdam_MIND_brain_default_config():
    '''
    ConvexAdam MIND default config.
    '''
    config = ml_collections.ConfigDict()
    config.mind_r = 1
    config.mind_d = 2
    config.lambda_weight = 1.6
    config.grid_sp = 2
    config.disp_hw = 2
    config.selected_niter = 80
    config.selected_smooth = 1
    config.grid_sp_adam = 2
    config.ic = True
    config.use_mask = False
    config.path_fixed_mask = None
    config.path_moving_mask = None
    config.result_path = './'
    config.verbose = False
    return config

def get_ConvexAdam_MIND_lung_default_config():
    '''
    ConvexAdam MIND default config.
    '''
    config = ml_collections.ConfigDict()
    config.mind_r = 1
    config.mind_d = 2
    config.lambda_weight = 0.65
    config.grid_sp = 4
    config.disp_hw = 6
    config.selected_niter = 80
    config.selected_smooth = 1
    config.grid_sp_adam = 2
    config.ic = True
    config.use_mask = False
    config.path_fixed_mask = None
    config.path_moving_mask = None
    config.result_path = './'
    config.verbose = False
    return config