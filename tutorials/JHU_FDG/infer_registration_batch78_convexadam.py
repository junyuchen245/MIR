import os
import sys
import glob
import json

import nibabel as nib
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MIR.models import SpatialTransformer, convex_adam_MIND
import MIR.models.convexAdam.configs_ConvexAdam_MIND as configs_ConvexAdam


# -----------------------------
# User-configurable parameters
# -----------------------------
DEVICE = "cuda:0"
OVERWRITE = False
SAVE_FLOW = False
FLIP_INPUT_AXIS1 = True

CT_PAD_VALUE = -1000.0
VALID_EPS = 1e-4
VALID_THR = 0.995

# Set to a case id string to debug one case, e.g. "4ce91ddb_20230703".
# Keep as None to process all cases.
PROCESS_ONLY_SID = "4ce91ddb_20230703"

INPUT_BATCH_DIRS = {
    "batch7": "/scratch2/jchen/DATA/JHU_FDG/batch7/preprocessed",
    "batch8": "/scratch2/jchen/DATA/JHU_FDG/batch8/preprocessed",
}
ATLAS_CT_PATH = "/scratch/jchen/python_projects/AutoPET/atlas/ct/TransMorphAtlas_MAE_1_MS_1_diffusion_1/dsc0.5425.nii.gz"
OUTPUT_ROOT = "./convexadam_outputs"


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


def infer_flow_convexadam(moving_ct_norm, atlas_ct_norm, cfg):
    # ConvexAdam performs instance optimization; gradients are required.
    with torch.enable_grad():
        flow = convex_adam_MIND(moving_ct_norm, atlas_ct_norm, cfg)
    return flow


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    atlas_nii, atlas_t = load_nifti_as_tensor(ATLAS_CT_PATH, device)
    atlas_t_norm = norm_img(atlas_t)

    input_shape = tuple(atlas_t.shape[2:])
    print(f"Atlas shape: {input_shape}")

    convex_cfg = configs_ConvexAdam.get_ConvexAdam_MIND_brain_default_config()

    spatial_linear = SpatialTransformer(input_shape, mode="bilinear").to(device)
    spatial_nn = SpatialTransformer(input_shape, mode="nearest").to(device)

    cases = find_cases(INPUT_BATCH_DIRS)
    print(f"Found {len(cases)} preprocessed cases across batch7+batch8")
    if len(cases) == 0:
        raise RuntimeError("No cases found. Check INPUT_BATCH_DIRS and *_CT.nii.gz files.")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log = {
        "model_type": "ConvexAdam_MIND",
        "device": str(device),
        "atlas": ATLAS_CT_PATH,
        "process_only_sid": PROCESS_ONLY_SID,
        "cases": [],
    }

    for i, c in enumerate(cases, 1):
        if PROCESS_ONLY_SID is not None and c["sid"] != PROCESS_ONLY_SID:
            continue

        case_out_dir = os.path.join(OUTPUT_ROOT, c["batch"], c["sid"])
        os.makedirs(case_out_dir, exist_ok=True)

        out_ct = os.path.join(case_out_dir, f"{c['sid']}_CT_warped.nii.gz")
        out_pet = os.path.join(case_out_dir, f"{c['sid']}_PET_warped.nii.gz")
        out_seg = os.path.join(case_out_dir, f"{c['sid']}_CTSeg_warped.nii.gz")
        out_flow = os.path.join(case_out_dir, f"{c['sid']}_flow_to_atlas.nii.gz")

        if (not OVERWRITE) and os.path.isfile(out_ct):
            print(f"[{i}/{len(cases)}] Skip existing: {c['batch']}/{c['sid']}")
            continue

        ct_nii, ct_t = load_nifti_as_tensor(c["ct"], device)
        if FLIP_INPUT_AXIS1:
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
        flow = infer_flow_convexadam(ct_t_norm, atlas_t_norm, convex_cfg)

        for i in range(10):
            ct_w_raw = spatial_linear(ct_t, flow/(10/(i+1)))
            ct_w = fill_outside_valid_region(ct_w_raw, flow, spatial_linear)
            save_nifti_like(ct_w[0, 0].detach().cpu().numpy(), case_out_dir+f'{c["sid"]}_CT_warped_step_{i+1}.nii.gz', ct_nii)
            

        if c["pet"] is not None:
            pet_nii, pet_t = load_nifti_as_tensor(c["pet"], device)
            if FLIP_INPUT_AXIS1:
                pet_t = torch.flip(pet_t, dims=[3])
            if tuple(pet_t.shape[2:]) == input_shape:
                pet_w = spatial_linear(pet_t, flow)
                for i in range(10):
                    pet_w = spatial_linear(pet_t, flow/(10/(i+1)))
                    save_nifti_like(pet_w[0, 0].detach().cpu().numpy(), case_out_dir+f'{c["sid"]}_PET_warped_step_{i+1}.nii.gz', pet_nii)
                #save_nifti_like(pet_w[0, 0].detach().cpu().numpy(), out_pet, pet_nii)
            else:
                print(f"[{i}/{len(cases)}] PET shape mismatch for {c['sid']}, skip PET warp.")

        if c["seg"] is not None:
            seg_nii, seg_t = load_nifti_as_tensor(c["seg"], device)
            if FLIP_INPUT_AXIS1:
                seg_t = torch.flip(seg_t, dims=[3])
            if tuple(seg_t.shape[2:]) == input_shape:
                seg_w = spatial_nn(seg_t, flow)
                save_nifti_like(np.rint(seg_w[0, 0].detach().cpu().numpy()).astype(np.int16), out_seg, seg_nii)
            else:
                print(f"[{i}/{len(cases)}] CTSeg shape mismatch for {c['sid']}, skip CTSeg warp.")

        if SAVE_FLOW:
            flow_np = flow[0].detach().cpu().numpy()
            flow_np = np.moveaxis(flow_np, 0, -1)  # [H, W, D, 3]
            save_nifti_like(flow_np.astype(np.float32), out_flow, atlas_nii)

        print(
            f"[{i}/{len(cases)}] Done {c['batch']}/{c['sid']} | "
            f"flow(min/mean/max)=({flow.min().item():.4f}, {flow.mean().item():.4f}, {flow.max().item():.4f})"
        )
        log["cases"].append(
            {
                "batch": c["batch"],
                "sid": c["sid"],
                "status": "done",
                "flow_direction": "moving_to_atlas",
                "ct": c["ct"],
                "pet": c["pet"],
                "seg": c["seg"],
                "out_dir": case_out_dir,
            }
        )

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
