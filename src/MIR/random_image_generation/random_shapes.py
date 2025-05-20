import torch
import torch.nn.functional as F
from MIR.random_image_generation.perlin_numpy import generate_fractal_noise_3d_torch, generate_perlin_noise_3d_torch
import numpy as np
import random
import MIR.models.registration_utils as utils

def gen_defs(img_size, down_factor_def=4, freq = 1,lacunarity = 2, res = (5, 6, 7), rescale=False, magnitude=None):
    '''
    Generate random deformation fields
    '''
    with torch.no_grad():
        if rescale and magnitude is not None:
            recale_factor = magnitude
        elif rescale and magnitude is None:
            recale_factor = down_factor_def
        else:
            recale_factor = 1
        H, W, D = img_size
        out_size = (H // down_factor_def, W // down_factor_def, D // down_factor_def)
        rand_defs = []
        for i_ in range(3):
            r1s = []
            r2s = []
            frequency = freq
            octaves = np.random.randint(1, 3, size=1)[0]
            for i in range(octaves):
                r1 = np.random.rand(frequency * res[0] + 1, frequency * res[1] + 1, frequency * res[2] + 1)
                r2 = np.random.rand(frequency * res[0] + 1, frequency * res[1] + 1, frequency * res[2] + 1)
                r1s.append(r1)
                r2s.append(r2)
                frequency *= lacunarity
            randdef = generate_fractal_noise_3d_torch(out_size, res, octaves, rand1=r1s, rand2=r2s)
            randdef = F.interpolate(randdef[None, None], scale_factor=down_factor_def, mode='trilinear',
                                    align_corners=False)
            randdef = randdef*recale_factor
            rand_defs.append(randdef)
        rand_def = torch.cat(rand_defs, dim=1)
        rand_def = utils.VecInt(img_size, nsteps=5).cuda()(rand_def.cuda())
        return rand_def

def gen_shapes(img_size, down_factor=16, freq = 1, res = (5, 6, 7), num_label=16, mean_min=None, mean_max=None, std_min=None, std_max=None, add_noise=False):
    '''
    Generate random shapes using perlin noise
    Args:
        img_size (tuple): Size of the images to be generated.
        down_factor (int): Downsampling factor for the generated Perlin noise.
        freq (int): Frequency of the Perlin noise.
        res (tuple): Resolution of the Perlin noise.
        num_label (int): Number of labels to be generated.
        mean_min (list): Minimum mean values for each label.
        mean_max (list): Maximum mean values for each label.
        std_min (list): Minimum standard deviation values for each label.
        std_max (list): Maximum standard deviation values for each label.
        add_noise (bool): Whether to add noise to the generated images.
    Returns:
        img_1 (torch.Tensor): First generated image.
        img_2 (torch.Tensor): Second generated image.
        lbl_1 (torch.Tensor): Label map for the first image.
        lbl_2 (torch.Tensor): Label map for the second image.
    '''
    with torch.no_grad():
        H, W, D = img_size
        out_size = (H // down_factor, W // down_factor, D // down_factor)
        octaves = 1
        lbls = []
        for i_ in range(num_label):
            r1s = []
            r2s = []
            frequency = freq
            r1 = np.random.rand(frequency * res[0] + 1, frequency * res[1] + 1, frequency * res[2] + 1)
            r2 = np.random.rand(frequency * res[0] + 1, frequency * res[1] + 1, frequency * res[2] + 1)
            r1s.append(r1)
            r2s.append(r2)
            noise = generate_fractal_noise_3d_torch(out_size, res, octaves, rand1=r1s, rand2=r2s)
            noise = F.interpolate(noise[None, None], scale_factor=down_factor, mode='trilinear', align_corners=False)
            noise = (noise - noise.min()) / (noise.max() - noise.min())#*rnd[0]
            lbls.append(noise)

        '''
        Generate random deformation using perlin noise
        '''
        rand_def_1 = gen_defs(img_size, res=res)
        rand_def_2 = gen_defs(img_size, res=res)
        lbl = torch.argmax(torch.cat(lbls, dim=1), dim=1, keepdim=True)

        '''
        Assign voxel values (adding noise is possible)
        '''
        if mean_min is None:
            mean_min = [0] + [25] * (num_label - 1)
        if mean_max is None:
            mean_max = [225] * num_label
        if std_min is None:
            std_min = [0] + [5] * (num_label - 1)
        if std_max is None:
            std_max = [25] * num_label
        m0, m1, s0, s1 = map(np.asarray, (mean_min, mean_max, std_min, std_max))
        mean_img = lbl.float().clone()
        mean = torch.from_numpy(m0 - m1).cuda() * torch.rand(num_label).cuda() + torch.from_numpy(m1).cuda()
        if add_noise:
            std_img = lbl.float().clone()
            std = torch.from_numpy(s0 - s1).cuda() * torch.rand(num_label).cuda() + torch.from_numpy(s1).cuda()
            noise = torch.rand(lbl.shape).cuda()
        for i in range(num_label):
            mean_img = torch.where(lbl.float()==i, mean[i], mean_img)
            if add_noise:
                std_img = torch.where(lbl.float()==i, std[i], std_img)
        if add_noise:
            img = noise*std_img+mean_img
        else:
            img = mean_img
        
        '''
        Apply random deformation to the generated shapes -> a pair of moving and fixed image
        '''
        amps = [4, 6, 8, 14, 16, 18]
        random.shuffle(amps)
        rand_def_1 = rand_def_1 * amps[0]
        random.shuffle(amps)
        rand_def_2 = rand_def_2 * amps[0]
        lbl_1 = utils.SpatialTransformer(img_size, 'nearest').cuda()(lbl.float().cuda(), rand_def_1.float().cuda())
        img_1 = utils.SpatialTransformer(img_size, 'bilinear').cuda()(img.float().cuda(), rand_def_1.float().cuda())
        lbl_2 = utils.SpatialTransformer(img_size, 'nearest').cuda()(lbl.float().cuda(), rand_def_2.float().cuda())
        img_2 = utils.SpatialTransformer(img_size, 'bilinear').cuda()(img.float().cuda(), rand_def_2.float().cuda())
        img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min())
        img_2 = (img_2 - img_2.min()) / (img_2.max() - img_2.min())
    return img_1, img_2, lbl_1, lbl_2

def gen_shapes_multimodal(img_size, down_factor=16, freq=1, res=(5, 6, 7), num_label=16, mean_min=None, mean_max=None, std_min=None, std_max=None, add_noise=False, return_all_combinations=False, probability=-1):
    """
    Generate a pair of 3‑D images + label maps.
    Each image gets its own random intensity per label so they
    resemble two imaging modalities of the same anatomy.
    Args:
        img_size (tuple): Size of the images to be generated.
        down_factor (int): Downsampling factor for the generated Perlin noise.
        freq (int): Frequency of the Perlin noise.
        res (tuple): Resolution of the Perlin noise.
        num_label (int): Number of labels to be generated.
        mean_min (list): Minimum mean values for each label.
        mean_max (list): Maximum mean values for each label.
        std_min (list): Minimum standard deviation values for each label.
        std_max (list): Maximum standard deviation values for each label.
        add_noise (bool): Whether to add noise to the generated images.
        return_all_combinations (bool): Whether to return all combinations of images.
        probability (float): Probability of generating a second image.
    Returns:
        img_1 (torch.Tensor): First generated image.
        img_2 (torch.Tensor): Second generated image.
        lbl_1 (torch.Tensor): Label map for the first image.
        lbl_2 (torch.Tensor): Label map for the second image.
    """
    with torch.no_grad():
        # --------------------------------------------------
        # 1) Shape generation with Perlin noise
        # --------------------------------------------------
        H, W, D = img_size
        out_size = (H // down_factor, W // down_factor, D // down_factor)
        octaves = 1
        lbls = []
        for _ in range(num_label):
            r1s, r2s = [], []
            r1s.append(np.random.rand(freq * res[0] + 1,
                                      freq * res[1] + 1,
                                      freq * res[2] + 1))
            r2s.append(np.random.rand(freq * res[0] + 1,
                                      freq * res[1] + 1,
                                      freq * res[2] + 1))
            noise = generate_fractal_noise_3d_torch(out_size, res, octaves,
                                                    rand1=r1s, rand2=r2s)
            noise = F.interpolate(noise[None, None],
                                   scale_factor=down_factor,
                                   mode='trilinear',
                                   align_corners=False)
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            lbls.append(noise)

        # stack into a multi‑channel tensor and take arg‑max → hard labels
        lbl = torch.argmax(torch.cat(lbls, dim=1), dim=1, keepdim=True)

        # --------------------------------------------------
        # 2) Intensity assignment (done twice, once per image)
        # --------------------------------------------------
        def _make_intensity_map():
            """Return an intensity volume where each label has its own mean (and std)."""
            # set default bounds if none provided
            _mean_min = [0] + [25] * (num_label - 1) if mean_min is None else mean_min
            _mean_max = [225] * num_label            if mean_max is None else mean_max
            _std_min  = [0] + [5]  * (num_label - 1) if std_min  is None else std_min
            _std_max  = [25] * num_label             if std_max  is None else std_max

            m0, m1, s0, s1 = map(np.asarray, (_mean_min, _mean_max,
                                              _std_min, _std_max))
            # draw random mean/std per label
            mean = torch.from_numpy(m0 - m1).cuda() * torch.rand(num_label).cuda() + \
                   torch.from_numpy(m1).cuda()

            if add_noise:
                std = torch.from_numpy(s0 - s1).cuda() * torch.rand(num_label).cuda() + \
                      torch.from_numpy(s1).cuda()
                noise_vol = torch.rand(lbl.shape).cuda()

            # build the volume
            vol = lbl.float().clone()
            if add_noise:
                std_vol = lbl.float().clone()

            for i in range(num_label):
                vol = torch.where(lbl.float() == i, mean[i], vol)
                if add_noise:
                    std_vol = torch.where(lbl.float() == i, std[i], std_vol)

            return noise_vol * std_vol + vol if add_noise else vol

        if probability > 0 and random.random() < probability:
            img_base_1 = _make_intensity_map()
            img_base_2 = img_base_1
        else:
            img_base_1 = _make_intensity_map()
            img_base_2 = _make_intensity_map()
        # --------------------------------------------------
        # 3) Random deformations → moving / fixed pair
        # --------------------------------------------------
        rand_def_1 = gen_defs(img_size, res=res)
        rand_def_2 = gen_defs(img_size, res=res)
        amps = [4, 6, 8, 14, 16, 18]

        random.shuffle(amps)
        rand_def_1 = rand_def_1 * amps[0]
        random.shuffle(amps)
        rand_def_2 = rand_def_2 * amps[0]

        st_nearest = utils.SpatialTransformer(img_size, 'nearest').cuda()
        st_linear  = utils.SpatialTransformer(img_size, 'bilinear').cuda()

        lbl_1 = st_nearest(lbl.float().cuda(), rand_def_1.float().cuda())
        img_1 = st_linear(img_base_1.cuda(), rand_def_1.float().cuda())
        lbl_2 = st_nearest(lbl.float().cuda(), rand_def_2.float().cuda())
        img_2 = st_linear(img_base_2.cuda(), rand_def_2.float().cuda())

        # rescale to [0, 1] just as before
        img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min())
        img_2 = (img_2 - img_2.min()) / (img_2.max() - img_2.min())
        
    if return_all_combinations:
        img_1_m1 = img_1
        img_2_m2 = img_2
        
        img_1_m2 = st_linear(img_base_2.cuda(), rand_def_1.float().cuda())
        img_2_m1 = st_linear(img_base_1.cuda(), rand_def_2.float().cuda())
        
        img_1_m2 = (img_1_m2 - img_1_m2.min()) / (img_1_m2.max() - img_1_m2.min())
        img_2_m1 = (img_2_m1 - img_2_m1.min()) / (img_2_m1.max() - img_2_m1.min())
        return img_1_m1, img_1_m2, img_2_m1, img_2_m2, lbl_1, lbl_2
        
    return img_1, img_2, lbl_1, lbl_2

def dice_Shape_VOI(y_pred, y_true, num_label=16):
    VOI_lbls = np.arange(num_label)
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)