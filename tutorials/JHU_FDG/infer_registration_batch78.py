import os
import sys
import glob
import json

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MIR.models import SpatialTransformer, TransMorphTVF
import MIR.models.configs_TransMorph as configs_TransMorph


# -----------------------------
# User-configurable parameters
# -----------------------------
DEVICE = "cuda:0"
FLIP_ATLAS_AXIS1 = True  # keep consistent with preprocessing.py convention
OVERWRITE = False
SAVE_FLOW = True
CT_PAD_VALUE = -1000.0
VALID_EPS = 1e-4
VALID_THR = 0.995

INPUT_BATCH_DIRS = {
    "batch7": "/scratch2/jchen/DATA/JHU_FDG/batch7/preprocessed",
    "batch8": "/scratch2/jchen/DATA/JHU_FDG/batch8/preprocessed",
}
ATLAS_CT_PATH = "/scratch/jchen/python_projects/AutoPET/atlas/ct/TransMorphAtlas_MAE_1_MS_1_diffusion_1/dsc0.5425.nii.gz"
CKPT_PATH = "/scratch/jchen/python_projects/AutoPET/experiments/TransMorphAtlasReg_ncc_1_diffusion_1/dsc0.6284.pth.tar"
CKPT_KEY = "state_dict"

OUTPUT_ROOT = "./"


def load_nifti_as_tensor(path, device):
    nii = nib.load(path)
    arr = nii.get_fdata().astype(np.float32)
    ten = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    return nii, ten


def save_nifti_like(data_np, out_path, ref_nii):
    out_img = nib.Nifti1Image(data_np, ref_nii.affine, ref_nii.header)
    nib.save(out_img, out_path)


def norm_img(img):
    img = img.clone()
    img[img < -300] = -300
    img[img > 300] = 300
    denom = img.max() - img.min()
    if denom == 0:
        return torch.zeros_like(img)
    norm = (img - img.min()) / denom
    return norm


def fill_outside_valid_region(warped_img, flow, spatial_transformer, pad_value=CT_PAD_VALUE, eps=VALID_EPS, valid_thr=VALID_THR):
    ones = torch.ones_like(warped_img)
    valid_w = spatial_transformer(ones, flow)
    corrected = warped_img / torch.clamp(valid_w, min=eps)
    valid_mask = valid_w > valid_thr
    corrected = torch.clamp(corrected, min=-1024.0, max=3071.0)
    return torch.where(valid_mask, corrected, torch.full_like(warped_img, pad_value))


def find_cases(batch_dirs):
    cases = []
    for batch_name, in_dir in batch_dirs.items():
        ct_paths = sorted(glob.glob(os.path.join(in_dir, "*_CT.nii.gz")))
        for ct_path in ct_paths:
            sid = os.path.basename(ct_path).replace("_CT.nii.gz", "")
            pet_path = os.path.join(in_dir, f"{sid}_PET.nii.gz")
            seg_path = os.path.join(in_dir, f"{sid}_CTSeg.nii.gz")
            item = {
                "batch": batch_name,
                "sid": sid,
                "ct": ct_path,
                "pet": pet_path if os.path.isfile(pet_path) else None,
                "seg": seg_path if os.path.isfile(seg_path) else None,
            }
            cases.append(item)
    return cases


def build_model(device, input_shape):
    h, w, d = input_shape
    config = configs_TransMorph.get_3DTransMorphAtlas3Lvl_config()
    config.img_size = (h // 2, w // 2, d // 2)
    config.window_size = (h // 32, w // 32, d // 32)
    config.out_chan = 3
    model = TransMorphTVF(config, time_steps=5, SVF=True).to(device)
    return model

def load_checkpoint(model, ckpt_path, ckpt_key, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt[ckpt_key] if ckpt_key in ckpt else ckpt

    # Support checkpoints saved from TemplateCreation/legacy wrappers:
    # - strip leading "module." (DDP)
    # - strip leading "reg_model." (TemplateCreation)
    # - map legacy "spatial_trans_half" -> current "spatial_trans_"
    # - ignore incompatible/non-essential keys (e.g., mean stream/integrator grids)
    remapped = {}
    for k, v in state_dict.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        if kk.startswith("reg_model."):
            kk = kk[len("reg_model."):]
        if kk.startswith("mean_stream."):
            continue
        if kk.startswith("spatial_trans_half."):
            kk = kk.replace("spatial_trans_half.", "spatial_trans_.", 1)
        if kk.startswith("integrate."):
            continue
        if kk.startswith("spatial_trans."):
            continue
        remapped[kk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"Checkpoint load summary | missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("Unexpected keys (first 10):", unexpected[:10])
    if len(missing) > 0:
        print("Missing keys (first 10):", missing[:10])


def infer_flows(model, moving, atlas):
    # TransMorphTVF predicts moving->fixed flow at half resolution.
    moving_half = F.avg_pool3d(moving, 2)
    atlas_half = F.avg_pool3d(atlas, 2)
    flow_half = model((moving_half, atlas_half))
    flow_full = F.interpolate(flow_half, scale_factor=2, mode="trilinear", align_corners=False) * 2.0
    # Keep variable names used in the current script.
    pos_flow = flow_full
    neg_flow = flow_full
    print(pos_flow.shape, neg_flow.shape)
    return pos_flow, neg_flow


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    atlas_nii, atlas_t = load_nifti_as_tensor(ATLAS_CT_PATH, device)   
    atlas_t_norm = norm_img(atlas_t)

    input_shape = tuple(atlas_t.shape[2:])
    print(f"Atlas shape: {input_shape}")

    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model = build_model(device, input_shape)
    load_checkpoint(model, CKPT_PATH, CKPT_KEY, device)
    model.eval()
    print(f"Loaded pretrained weights: {CKPT_PATH}")

    spatial_linear = SpatialTransformer(input_shape, mode="bilinear").to(device)
    spatial_nn = SpatialTransformer(input_shape, mode="nearest").to(device)

    cases = find_cases(INPUT_BATCH_DIRS)
    print(f"Found {len(cases)} preprocessed cases across batch7+batch8")
    if len(cases) == 0:
        raise RuntimeError("No cases found. Check INPUT_BATCH_DIRS and *_CT.nii.gz files.")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log = {"model_type": "TransMorphTVF", "device": str(device), "atlas": ATLAS_CT_PATH, "cases": []}

    with torch.no_grad():
        for i, c in enumerate(cases, 1):
            case_out_dir = os.path.join(OUTPUT_ROOT, c["batch"], c["sid"])
            print(c["sid"])
            if c['sid'] != '4ce91ddb_20230703':
                continue
            
            os.makedirs(case_out_dir, exist_ok=True)

            out_ct = os.path.join(case_out_dir, f"{c['sid']}_CT_warped.nii.gz")
            out_pet = os.path.join(case_out_dir, f"{c['sid']}_PET_warped.nii.gz")
            out_seg = os.path.join(case_out_dir, f"{c['sid']}_CTSeg_warped.nii.gz")
            out_flow = os.path.join(case_out_dir, f"{c['sid']}_flow_to_atlas.nii.gz")

            if (not OVERWRITE) and os.path.isfile(out_ct):
                print(f"[{i}/{len(cases)}] Skip existing: {c['batch']}/{c['sid']}")
                continue

            ct_nii, ct_t = load_nifti_as_tensor(c["ct"], device)
            ct_t = torch.flip(ct_t, dims=[3])
            if tuple(ct_t.shape[2:]) != input_shape:
                print(
                    f"[{i}/{len(cases)}] Shape mismatch for {c['sid']}: "
                    f"moving={tuple(ct_t.shape[2:])}, atlas={input_shape}. Skipped."
                )
                log["cases"].append(
                    {
                        "batch": c["batch"],
                        "sid": c["sid"],
                        "status": "skipped_shape_mismatch",
                        "moving_shape": list(ct_t.shape[2:]),
                        "atlas_shape": list(input_shape),
                    }
                )
                continue

            ct_t_norm = norm_img(ct_t)
            pos_flow, neg_flow = infer_flows(model, ct_t_norm, atlas_t_norm)
            # neg_flow warps moving -> atlas (same as AutoPET infer_TransMorph_atlas.py)
            ct_w_raw = spatial_linear(ct_t, neg_flow)
            ct_w = fill_outside_valid_region(ct_w_raw, neg_flow, spatial_linear)

            save_nifti_like(ct_w[0, 0].detach().cpu().numpy(), out_ct, ct_nii)

            if c["pet"] is not None:
                pet_nii, pet_t = load_nifti_as_tensor(c["pet"], device)
                pet_t = torch.flip(pet_t, dims=[3])
                if tuple(pet_t.shape[2:]) == input_shape:
                    pet_w = spatial_linear(pet_t, neg_flow)
                    save_nifti_like(pet_w[0, 0].detach().cpu().numpy(), out_pet, pet_nii)
                else:
                    print(f"[{i}/{len(cases)}] PET shape mismatch for {c['sid']}, skip PET warp.")

            if c["seg"] is not None:
                seg_nii, seg_t = load_nifti_as_tensor(c["seg"], device)
                seg_t = torch.flip(seg_t, dims=[3])
                if tuple(seg_t.shape[2:]) == input_shape:
                    seg_w = spatial_nn(seg_t, neg_flow)
                    save_nifti_like(np.rint(seg_w[0, 0].detach().cpu().numpy()).astype(np.int16), out_seg, seg_nii)
                else:
                    print(f"[{i}/{len(cases)}] CTSeg shape mismatch for {c['sid']}, skip CTSeg warp.")

            if SAVE_FLOW:
                flow_np = neg_flow[0].detach().cpu().numpy()
                flow_np = np.moveaxis(flow_np, 0, -1)  # [H, W, D, 3]
                save_nifti_like(flow_np.astype(np.float32), out_flow, atlas_nii)

            print(
                f"[{i}/{len(cases)}] Done {c['batch']}/{c['sid']} | "
                f"flow(min/mean/max)=({neg_flow.min().item():.4f}, {neg_flow.mean().item():.4f}, {neg_flow.max().item():.4f})"
            )
            log["cases"].append(
                {
                    "batch": c["batch"],
                    "sid": c["sid"],
                    "status": "done",
                    "flow_direction": "moving_to_atlas (neg_flow)",
                    "ct": c["ct"],
                    "pet": c["pet"],
                    "seg": c["seg"],
                    "out_dir": case_out_dir,
                }
            )
            sys.exit(0)

    log_path = os.path.join(OUTPUT_ROOT, "run_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Finished. Log saved to: {log_path}")


if __name__ == "__main__":
    gpu_num = torch.cuda.device_count()
    print("Number of GPU:", gpu_num)
    if gpu_num > 0 and torch.cuda.is_available():
        for gpu_idx in range(gpu_num):
            print(f"  GPU #{gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")
    main()
