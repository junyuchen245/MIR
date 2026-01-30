import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nibabel as nib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MIR.models import SpatialTransformer, TransMorphTVF, VecInt, fit_warp_to_svf_fast, VFA
import MIR.models.configs_TransMorph as configs_TransMorph
import MIR.models.configs_VFA as configs_VFA
from MIR import ModelWeights, DatasetJSONs
import gdown

INPUT_SHAPE = (160, 224, 192)
LUMIR_BASE_DIR = "/scratch2/jchen/DATA/LUMIR/"
LUMIR_JSON = "LUMIR_dataset.json"
MODEL_TYPE = "TransMorphTVF"  # "TransMorphTVF" or "VFA"
WEIGHTS_PATH = "pretrained_wts/"
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

if not os.path.isdir(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)
if MODEL_TYPE == "VFA":
    pretrained_wts = "VFA_LUMIR24.pth.tar"
    MODEL_SUBTYPE = "VFA-LUMIR24-MonoModal"
elif MODEL_TYPE == "TransMorphTVF":
    pretrained_wts = "TransMorphTVF.pth.tar"
    MODEL_SUBTYPE = "TransMorphTVF-LUMIR24-MonoModal"
if not os.path.isfile(WEIGHTS_PATH+pretrained_wts):
    # download model
    file_id = ModelWeights[MODEL_SUBTYPE]['wts']
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, WEIGHTS_PATH+pretrained_wts, quiet=False)

if not os.path.isfile(LUMIR_JSON):
    # download dataset json file
    file_id = DatasetJSONs['LUMIR24']
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, LUMIR_JSON, quiet=False)

class L2RLUMIRJSONDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, json_path, stage="train"):
        with open(json_path) as f:
            d = json.load(f)
        if stage.lower() == "train":
            self.imgs = d["training"]
        elif stage.lower() == "validation":
            self.imgs = d["validation"]
        else:
            self.imgs = d["validation"]
        self.base_dir = base_dir
        self.stage = stage

    def __getitem__(self, index):
        if self.stage == "train":
            mov_dict = self.imgs[index]
            x = nib.load(self.base_dir + mov_dict["image"])
        else:
            img_dict = self.imgs[index]
            x = nib.load(self.base_dir + img_dict["moving"])
        x = x.get_fdata() / 255.
        x = x[None, ...]
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        return x.float(), x.float()

    def __len__(self):
        return len(self.imgs)

def build_model(device, input_shape):
    h, w, d = input_shape
    if MODEL_TYPE == "VFA":
        config = configs_VFA.get_VFA_default_config()
        config.img_size = (h, w, d)
        model = VFA(config, device=str(device))
        return model.to(device)
    if MODEL_TYPE == "TransMorphTVF":
        scale_factor = 2
        config = configs_TransMorph.get_3DTransMorph3Lvl_config()
        config.img_size = (h // scale_factor, w // scale_factor, d // scale_factor)
        config.window_size = (h // 64, w // 64, d // 64)
        config.out_chan = 3
        model = TransMorphTVF(config, SVF=USE_SVF_MODEL, time_steps=7).to(device)
        return model
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

def load_checkpoint(model, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)[ModelWeights[MODEL_SUBTYPE]['wts_key']]
    model.load_state_dict(state_dict)

def ensure_channel_dim(tensor):
    if tensor.dim() == 4:
        return tensor.unsqueeze(1)
    return tensor

def init_template_from_atlas(loader, device):
    for data in loader:
        x, _ = data[:2]
        x = ensure_channel_dim(x)
        return x.float().to(device)
    raise RuntimeError("No data found to initialize template.")

def save_template(template, out_dir, iteration):
    os.makedirs(out_dir, exist_ok=True)
    template_np = template.detach().cpu().numpy()
    out_path = os.path.join(out_dir, f"template_iter_{iteration:02d}.nii.gz")
    nii = nib.Nifti1Image(template_np[0, 0], np.eye(4))
    nib.save(nii, out_path)

def build_template():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_set = L2RLUMIRJSONDataset(
        base_dir=LUMIR_BASE_DIR,
        json_path=LUMIR_JSON,
        stage="validation",
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Train samples: {len(train_set)}")

    model = build_model(device, INPUT_SHAPE)
    load_checkpoint(model, WEIGHTS_PATH+pretrained_wts, device)
    model.eval()
    print(f"Loaded weights from: {WEIGHTS_PATH+pretrained_wts}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"SVF model output: {USE_SVF_MODEL}")
    print(f"Model subtype: {MODEL_SUBTYPE}")
    spatial_trans = SpatialTransformer(INPUT_SHAPE).to(device)

    template = init_template_from_atlas(train_loader, device)
    print(
        f"Init template stats: min={template.min().item():.4f}, "
        f"mean={template.mean().item():.4f}, max={template.max().item():.4f}"
    )
    save_template(template, OUT_DIR, 0)

    with torch.no_grad():
        for iteration in range(1, NUM_ITERS + 1):
            accum = torch.zeros_like(template)
            flow_sum = torch.zeros((1, 3) + tuple(INPUT_SHAPE), device=device)
            vel_sum = torch.zeros((1, 3) + tuple(INPUT_SHAPE), device=device)
            count = 0
            for _, moving in train_loader:
                moving = ensure_channel_dim(moving).float().to(device)
                template_batch = template.expand(moving.shape[0], -1, -1, -1, -1)

                if MODEL_TYPE == "TransMorphTVF":
                    moving_half = F.avg_pool3d(moving, 2)
                    template_half = F.avg_pool3d(template_batch, 2)
                    flow = model((moving_half, template_half))
                    flow = F.interpolate(flow, scale_factor=2, mode="trilinear", align_corners=False) * 2
                else:
                    flow = model((moving, template_batch))
                warped = spatial_trans(moving, flow)

                if count % LOG_EVERY == 0:
                    print(
                        f"Iter {iteration} | batch {count + 1}: "
                        f"flow(min/mean/max)=({flow.min().item():.4f}, {flow.mean().item():.4f}, {flow.max().item():.4f}) "
                        f"warped(min/mean/max)=({warped.min().item():.4f}, {warped.mean().item():.4f}, {warped.max().item():.4f})"
                    )

                accum += warped.sum(dim=0, keepdim=True)
                if SHAPE_AVG:
                    if SHAPE_AVG_LOGDOMAIN:
                        if USE_SVF_MODEL:
                            vel = flow.detach()
                        else:
                            with torch.enable_grad():
                                vel = fit_warp_to_svf_fast(
                                    flow.detach(),
                                    nb_steps=SHAPE_AVG_NB_STEPS,
                                    iters=SHAPE_AVG_ITERS,
                                    lr=SHAPE_AVG_LR,
                                    output_type="disp",
                                    verbose=SHAPE_AVG_VERBOSE,
                                )
                        vel_sum += vel.sum(dim=0, keepdim=True)
                    else:
                        if USE_SVF_MODEL:
                            disp = VecInt(flow.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(flow)
                            flow_sum += disp.sum(dim=0, keepdim=True)
                        else:
                            flow_sum += flow.sum(dim=0, keepdim=True)
                count += moving.shape[0]

            template = accum / max(count, 1)
            print(
                f"Iter {iteration} | template pre-shape-avg stats: "
                f"min={template.min().item():.4f}, mean={template.mean().item():.4f}, max={template.max().item():.4f}"
            )
            if SHAPE_AVG:
                if SHAPE_AVG_LOGDOMAIN:
                    avg_vel = vel_sum / max(count, 1)
                    inv_disp = VecInt(avg_vel.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(-avg_vel)
                    template = spatial_trans(template, inv_disp)
                    print(
                        f"Iter {iteration} | shape-avg(log-domain) stats: "
                        f"avg_vel(min/mean/max)=({avg_vel.min().item():.4f}, {avg_vel.mean().item():.4f}, {avg_vel.max().item():.4f}) "
                        f"inv_disp(min/mean/max)=({inv_disp.min().item():.4f}, {inv_disp.mean().item():.4f}, {inv_disp.max().item():.4f})"
                    )
                else:
                    avg_flow = flow_sum / max(count, 1)
                    with torch.enable_grad():
                        inv_vel = fit_warp_to_svf_fast(
                            avg_flow.detach(),
                            nb_steps=SHAPE_AVG_NB_STEPS,
                            iters=SHAPE_AVG_ITERS,
                            lr=SHAPE_AVG_LR,
                            output_type="disp",
                            verbose=SHAPE_AVG_VERBOSE,
                        )
                    inv_disp = VecInt(avg_flow.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(-inv_vel)
                    template = spatial_trans(template, inv_disp)
                    print(
                        f"Iter {iteration} | shape-avg(flow) stats: "
                        f"avg_flow(min/mean/max)=({avg_flow.min().item():.4f}, {avg_flow.mean().item():.4f}, {avg_flow.max().item():.4f}) "
                        f"inv_disp(min/mean/max)=({inv_disp.min().item():.4f}, {inv_disp.mean().item():.4f}, {inv_disp.max().item():.4f})"
                    )
            print(
                f"Iter {iteration} | template final stats: "
                f"min={template.min().item():.4f}, mean={template.mean().item():.4f}, max={template.max().item():.4f}"
            )
            save_template(template, OUT_DIR, iteration)


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
