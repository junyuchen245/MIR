import os
import sys
import json
import glob
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nibabel as nib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MIR.models import SpatialTransformer, VecInt, fit_warp_to_svf_fast, convex_adam_MIND
import MIR.models.convexAdam.configs_ConvexAdam_MIND as configs_ConvexAdam

INPUT_SHAPE = (192, 192, 256)
DATA_BASE_DIR = '/scratch2/jchen/DATA/PSMA_autoPET/Preprocessed/autoPET/'
OUT_DIR = "template_outputs"
NUM_ITERS = 10
BATCH_SIZE = 1
NUM_WORKERS = 4
DEVICE = "cuda:0"
USE_SVF_MODEL = False
SHAPE_AVG = True
SHAPE_AVG_LOGDOMAIN = True
SHAPE_AVG_NB_STEPS = 7
SHAPE_AVG_ITERS = 200
SHAPE_AVG_LR = 0.1
SHAPE_AVG_VERBOSE = True
LOG_EVERY = 10

class JHUPSMADataset(Dataset):
    def __init__(self, data_path, data_names):
        self.path = data_path
        self.data_names = data_names

    def norm_img(self, img):
        img[img < -300] = -300
        img[img > 300] = 300
        norm = (img - img.min()) / (img.max() - img.min())
        return norm

    def norm_suv(self, img):
        img_max = 15.#np.percentile(img, 95)
        img_min = 0.#np.percentile(img, 5)
        norm = (img - img_min)/(img_max - img_min)
        norm[norm < 0] = 0
        norm[norm > 1] = 1
        return norm

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        mov_name = self.data_names[index]
        x = nib.load(self.path + '{}_CT.nii.gz'.format(mov_name))
        x = self.norm_img(x.get_fdata())
        x_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(mov_name))
        x_suv = self.norm_suv(x_suv.get_fdata())
        x = x[None, ...]
        x_suv = x_suv[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv = np.ascontiguousarray(x_suv)  # [Bsize,channelsHeight,,Width,Depth]
        x = torch.from_numpy(x)
        x_suv = torch.from_numpy(x_suv)
        return x, x_suv

    def __len__(self):
        return len(self.data_names)

def build_model():
    return convex_adam_MIND, configs_ConvexAdam.get_ConvexAdam_MIND_petct_default_config()

def ensure_channel_dim(tensor):
    if tensor.dim() == 4:
        return tensor.unsqueeze(1)
    return tensor

def init_template_from_atlas(loader, device):
    for data in loader:
        x_ct, x_suv = data[:2]
        x_ct = ensure_channel_dim(x_ct)
        x_suv = ensure_channel_dim(x_suv)
        return x_ct.float().to(device), x_suv.float().to(device)
    raise RuntimeError("No data found to initialize template.")

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

def save_template(template, out_dir, iteration, suffix):
    os.makedirs(out_dir, exist_ok=True)
    template_np = template.detach().cpu().numpy()
    out_path = os.path.join(out_dir, f"template_{suffix}_iter_{iteration:02d}.nii.gz")
    target_pixdim = [2.8, 2.8, 3.8]
    affine = make_affine_from_pixdim(target_pixdim)
    nii = nib.Nifti1Image(template_np[0, 0], affine=affine)
    nib.save(nii, out_path)

def build_template():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_names = glob.glob(DATA_BASE_DIR + '*_CT_seg*')
    full_names = [name.split('/')[-1].split('_CT_seg')[0] for name in full_names]
    train_names = full_names[0:-8]
    val_names = full_names[-8:]
    print('Training on {} patients, validating on {} patients'.format(len(train_names), len(val_names)))
    train_set = JHUPSMADataset(DATA_BASE_DIR, train_names)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Train samples: {len(train_set)}")

    model_fn, model_cfg = build_model()
    print("Model type: ConvexAdam")
    print(f"SVF model output: {USE_SVF_MODEL}")
    spatial_trans = SpatialTransformer(INPUT_SHAPE).to(device)

    template_ct, template_suv = init_template_from_atlas(train_loader, device)
    print(
        f"Init CT template stats: min={template_ct.min().item():.4f}, "
        f"mean={template_ct.mean().item():.4f}, max={template_ct.max().item():.4f}"
    )
    print(
        f"Init SUV template stats: min={template_suv.min().item():.4f}, "
        f"mean={template_suv.mean().item():.4f}, max={template_suv.max().item():.4f}"
    )
    save_template(template_ct, OUT_DIR, 0, "ct")
    save_template(template_suv, OUT_DIR, 0, "suv")

    with torch.no_grad():
        for iteration in range(1, NUM_ITERS + 1):
            accum_ct = torch.zeros_like(template_ct)
            accum_suv = torch.zeros_like(template_suv)
            flow_sum_ct = torch.zeros((1, 3) + tuple(INPUT_SHAPE), device=device)
            flow_sum_suv = torch.zeros((1, 3) + tuple(INPUT_SHAPE), device=device)
            vel_sum_ct = torch.zeros((1, 3) + tuple(INPUT_SHAPE), device=device)
            vel_sum_suv = torch.zeros((1, 3) + tuple(INPUT_SHAPE), device=device)
            count = 0
            for moving_ct, moving_suv in train_loader:
                moving_ct = ensure_channel_dim(moving_ct).float().to(device)
                moving_suv = ensure_channel_dim(moving_suv).float().to(device)
                template_batch_ct = template_ct.expand(moving_ct.shape[0], -1, -1, -1, -1)
                template_batch_suv = template_suv.expand(moving_suv.shape[0], -1, -1, -1, -1)

                with torch.enable_grad():
                    flow_ct = model_fn(moving_ct, template_batch_ct, model_cfg)
                    flow_suv = model_fn(moving_suv, template_batch_suv, model_cfg)
                warped_ct = spatial_trans(moving_ct, flow_ct)
                warped_suv = spatial_trans(moving_suv, flow_suv)

                if count % LOG_EVERY == 0:
                    print(
                        f"Iter {iteration} | batch {count + 1}: "
                        f"CT flow(min/mean/max)=({flow_ct.min().item():.4f}, {flow_ct.mean().item():.4f}, {flow_ct.max().item():.4f}) "
                        f"CT warped(min/mean/max)=({warped_ct.min().item():.4f}, {warped_ct.mean().item():.4f}, {warped_ct.max().item():.4f}) "
                        f"| SUV flow(min/mean/max)=({flow_suv.min().item():.4f}, {flow_suv.mean().item():.4f}, {flow_suv.max().item():.4f}) "
                        f"SUV warped(min/mean/max)=({warped_suv.min().item():.4f}, {warped_suv.mean().item():.4f}, {warped_suv.max().item():.4f})"
                    )

                accum_ct += warped_ct.sum(dim=0, keepdim=True)
                accum_suv += warped_suv.sum(dim=0, keepdim=True)
                if SHAPE_AVG:
                    if SHAPE_AVG_LOGDOMAIN:
                        if USE_SVF_MODEL:
                            vel_ct = flow_ct.detach()
                            vel_suv = flow_suv.detach()
                        else:
                            with torch.enable_grad():
                                vel_ct = fit_warp_to_svf_fast(
                                    flow_ct.detach(),
                                    nb_steps=SHAPE_AVG_NB_STEPS,
                                    iters=SHAPE_AVG_ITERS,
                                    lr=SHAPE_AVG_LR,
                                    output_type="disp",
                                    verbose=SHAPE_AVG_VERBOSE,
                                )
                                vel_suv = fit_warp_to_svf_fast(
                                    flow_suv.detach(),
                                    nb_steps=SHAPE_AVG_NB_STEPS,
                                    iters=SHAPE_AVG_ITERS,
                                    lr=SHAPE_AVG_LR,
                                    output_type="disp",
                                    verbose=SHAPE_AVG_VERBOSE,
                                )
                        vel_sum_ct += vel_ct.sum(dim=0, keepdim=True)
                        vel_sum_suv += vel_suv.sum(dim=0, keepdim=True)
                    else:
                        if USE_SVF_MODEL:
                            disp_ct = VecInt(flow_ct.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(flow_ct)
                            disp_suv = VecInt(flow_suv.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(flow_suv)
                            flow_sum_ct += disp_ct.sum(dim=0, keepdim=True)
                            flow_sum_suv += disp_suv.sum(dim=0, keepdim=True)
                        else:
                            flow_sum_ct += flow_ct.sum(dim=0, keepdim=True)
                            flow_sum_suv += flow_suv.sum(dim=0, keepdim=True)
                count += moving_ct.shape[0]

            template_ct = accum_ct / max(count, 1)
            template_suv = accum_suv / max(count, 1)
            print(
                f"Iter {iteration} | CT template pre-shape-avg stats: "
                f"min={template_ct.min().item():.4f}, mean={template_ct.mean().item():.4f}, max={template_ct.max().item():.4f}"
            )
            print(
                f"Iter {iteration} | SUV template pre-shape-avg stats: "
                f"min={template_suv.min().item():.4f}, mean={template_suv.mean().item():.4f}, max={template_suv.max().item():.4f}"
            )
            if SHAPE_AVG:
                if SHAPE_AVG_LOGDOMAIN:
                    avg_vel_ct = vel_sum_ct / max(count, 1)
                    avg_vel_suv = vel_sum_suv / max(count, 1)
                    inv_disp_ct = VecInt(avg_vel_ct.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(-avg_vel_ct)
                    inv_disp_suv = VecInt(avg_vel_suv.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(-avg_vel_suv)
                    template_ct = spatial_trans(template_ct, inv_disp_ct)
                    template_suv = spatial_trans(template_suv, inv_disp_suv)
                    print(
                        f"Iter {iteration} | shape-avg(log-domain) CT stats: "
                        f"avg_vel(min/mean/max)=({avg_vel_ct.min().item():.4f}, {avg_vel_ct.mean().item():.4f}, {avg_vel_ct.max().item():.4f}) "
                        f"inv_disp(min/mean/max)=({inv_disp_ct.min().item():.4f}, {inv_disp_ct.mean().item():.4f}, {inv_disp_ct.max().item():.4f})"
                    )
                    print(
                        f"Iter {iteration} | shape-avg(log-domain) SUV stats: "
                        f"avg_vel(min/mean/max)=({avg_vel_suv.min().item():.4f}, {avg_vel_suv.mean().item():.4f}, {avg_vel_suv.max().item():.4f}) "
                        f"inv_disp(min/mean/max)=({inv_disp_suv.min().item():.4f}, {inv_disp_suv.mean().item():.4f}, {inv_disp_suv.max().item():.4f})"
                    )
                else:
                    avg_flow_ct = flow_sum_ct / max(count, 1)
                    avg_flow_suv = flow_sum_suv / max(count, 1)
                    with torch.enable_grad():
                        inv_vel_ct = fit_warp_to_svf_fast(
                            avg_flow_ct.detach(),
                            nb_steps=SHAPE_AVG_NB_STEPS,
                            iters=SHAPE_AVG_ITERS,
                            lr=SHAPE_AVG_LR,
                            output_type="disp",
                            verbose=SHAPE_AVG_VERBOSE,
                        )
                        inv_vel_suv = fit_warp_to_svf_fast(
                            avg_flow_suv.detach(),
                            nb_steps=SHAPE_AVG_NB_STEPS,
                            iters=SHAPE_AVG_ITERS,
                            lr=SHAPE_AVG_LR,
                            output_type="disp",
                            verbose=SHAPE_AVG_VERBOSE,
                        )
                    inv_disp_ct = VecInt(avg_flow_ct.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(-inv_vel_ct)
                    inv_disp_suv = VecInt(avg_flow_suv.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(-inv_vel_suv)
                    template_ct = spatial_trans(template_ct, inv_disp_ct)
                    template_suv = spatial_trans(template_suv, inv_disp_suv)
                    print(
                        f"Iter {iteration} | shape-avg(flow) CT stats: "
                        f"avg_flow(min/mean/max)=({avg_flow_ct.min().item():.4f}, {avg_flow_ct.mean().item():.4f}, {avg_flow_ct.max().item():.4f}) "
                        f"inv_disp(min/mean/max)=({inv_disp_ct.min().item():.4f}, {inv_disp_ct.mean().item():.4f}, {inv_disp_ct.max().item():.4f})"
                    )
                    print(
                        f"Iter {iteration} | shape-avg(flow) SUV stats: "
                        f"avg_flow(min/mean/max)=({avg_flow_suv.min().item():.4f}, {avg_flow_suv.mean().item():.4f}, {avg_flow_suv.max().item():.4f}) "
                        f"inv_disp(min/mean/max)=({inv_disp_suv.min().item():.4f}, {inv_disp_suv.mean().item():.4f}, {inv_disp_suv.max().item():.4f})"
                    )
            print(
                f"Iter {iteration} | CT template final stats: "
                f"min={template_ct.min().item():.4f}, mean={template_ct.mean().item():.4f}, max={template_ct.max().item():.4f}"
            )
            print(
                f"Iter {iteration} | SUV template final stats: "
                f"min={template_suv.min().item():.4f}, mean={template_suv.mean().item():.4f}, max={template_suv.max().item():.4f}"
            )
            save_template(template_ct, OUT_DIR, iteration, "ct")
            save_template(template_suv, OUT_DIR, iteration, "suv")


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
    torch.manual_seed(0)
    build_template()
