import glob
import os
import sys

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MIR.models import SpatialTransformer, VecInt, fit_warp_to_svf_fast, convex_adam_MIND_SVF
import MIR.models.convexAdam.configs_ConvexAdam_MIND as configs_ConvexAdam

INPUT_SHAPE = (192, 192, 144)
PREPROCESSED_DIR = "/scratch2/jchen/DATA/HECKTOR25/preprocessed"
REGISTRATION_MODALITY = "CT"  # "CT" | "PET"
OUT_DIR = os.path.join(THIS_DIR, "template_outputs_hecktor_convexadam_svf_paired")
NUM_ITERS = 10
BATCH_SIZE = 1
NUM_WORKERS = 4
DEVICE = "cuda:0"
USE_LOGDOMAIN_SHAPE_AVG = True
SHAPE_AVG_NB_STEPS = 7
SHAPE_AVG_ITERS = 100
SHAPE_AVG_LR = 0.1
SHAPE_AVG_VERBOSE = False
LOG_EVERY = 10
MAX_CASES = None


class HECKTORPreprocessedPairDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_dir, max_cases=None):
        self.preprocessed_dir = preprocessed_dir
        self.items = self._collect_items(max_cases)
        if not self.items:
            raise RuntimeError(f"No paired preprocessed HECKTOR CT/PET images found in {preprocessed_dir}")

    def _collect_items(self, max_cases=None):
        pattern = os.path.join(self.preprocessed_dir, "*_CT.nii.gz")
        items = []
        for ct_path in sorted(glob.glob(pattern)):
            case_id = os.path.basename(ct_path)[: -len("_CT.nii.gz")]
            pet_path = os.path.join(self.preprocessed_dir, f"{case_id}_PET.nii.gz")
            if not os.path.isfile(pet_path):
                continue
            items.append({"case_id": case_id, "ct_path": ct_path, "pet_path": pet_path})
        if max_cases is not None:
            items = items[:max_cases]
        return items

    def __getitem__(self, index):
        item = self.items[index]
        ct_nib = nib.load(item["ct_path"])
        pet_nib = nib.load(item["pet_path"])
        ct = torch.from_numpy(np.ascontiguousarray(ct_nib.get_fdata().astype(np.float32)[None, ...]))
        pet = torch.from_numpy(np.ascontiguousarray(pet_nib.get_fdata().astype(np.float32)[None, ...]))
        meta = {
            "case_id": item["case_id"],
            "ct_path": item["ct_path"],
            "pet_path": item["pet_path"],
            "affine": ct_nib.affine.astype(np.float32),
        }
        return ct.float(), pet.float(), meta

    def __len__(self):
        return len(self.items)


def ensure_channel_dim(tensor):
    if tensor.dim() == 4:
        return tensor.unsqueeze(1)
    return tensor


def build_model():
    config = configs_ConvexAdam.get_ConvexAdam_MIND_petct_default_config()
    config.svf_steps = SHAPE_AVG_NB_STEPS
    config.verbose = False
    config.return_velocity = True
    return convex_adam_MIND_SVF, config


def init_template_from_first_case(loader, device):
    for batch in loader:
        ct, pet, meta = batch
        ct = ensure_channel_dim(ct).float().to(device)
        pet = ensure_channel_dim(pet).float().to(device)
        affine = meta["affine"]
        if isinstance(affine, torch.Tensor):
            affine_np = affine[0].detach().cpu().numpy()
        else:
            affine_np = np.asarray(affine[0], dtype=np.float32)
        return ct, pet, affine_np
    raise RuntimeError("No cases available to initialize template.")


def save_template(template, affine, out_dir, iteration, modality):
    os.makedirs(out_dir, exist_ok=True)
    template_np = template.detach().cpu().numpy()[0, 0]
    out_path = os.path.join(out_dir, f"template_iter_{iteration:02d}_{modality}.nii.gz")
    nib.save(nib.Nifti1Image(template_np, affine), out_path)


def build_template():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Preprocessed dir: {PREPROCESSED_DIR}")
    print("Paired modalities: CT + PET")
    print(f"Registration modality: {REGISTRATION_MODALITY}")

    dataset = HECKTORPreprocessedPairDataset(
        preprocessed_dir=PREPROCESSED_DIR,
        max_cases=MAX_CASES,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Cases: {len(dataset)}")

    model_fn, model_cfg = build_model()
    spatial_trans = SpatialTransformer(INPUT_SHAPE).to(device)

    template_ct, template_pet, template_affine = init_template_from_first_case(loader, device)
    print(
        f"Init CT template stats: min={template_ct.min().item():.4f}, "
        f"mean={template_ct.mean().item():.4f}, max={template_ct.max().item():.4f}"
    )
    print(
        f"Init PET template stats: min={template_pet.min().item():.4f}, "
        f"mean={template_pet.mean().item():.4f}, max={template_pet.max().item():.4f}"
    )
    save_template(template_ct, template_affine, OUT_DIR, 0, "CT")
    save_template(template_pet, template_affine, OUT_DIR, 0, "PET")

    with torch.no_grad():
        for iteration in range(1, NUM_ITERS + 1):
            accum_ct = torch.zeros_like(template_ct)
            accum_pet = torch.zeros_like(template_pet)
            vel_sum = torch.zeros((1, 3) + tuple(INPUT_SHAPE), device=device)
            flow_sum = torch.zeros((1, 3) + tuple(INPUT_SHAPE), device=device)
            count = 0

            for batch_idx, batch in enumerate(loader):
                moving_ct, moving_pet, meta = batch
                moving_ct = ensure_channel_dim(moving_ct).float().to(device)
                moving_pet = ensure_channel_dim(moving_pet).float().to(device)
                template_batch_ct = template_ct.expand(moving_ct.shape[0], -1, -1, -1, -1)
                template_batch_pet = template_pet.expand(moving_pet.shape[0], -1, -1, -1, -1)

                moving_reg = moving_ct if REGISTRATION_MODALITY == "CT" else moving_pet
                template_reg = template_batch_ct if REGISTRATION_MODALITY == "CT" else template_batch_pet

                with torch.enable_grad():
                    flow_fwd, flow_rev, velocity = model_fn(moving_reg, template_reg, model_cfg)
                warped_ct = spatial_trans(moving_ct, flow_fwd)
                warped_pet = spatial_trans(moving_pet, flow_fwd)

                if batch_idx % LOG_EVERY == 0:
                    case_id = meta["case_id"][0] if isinstance(meta["case_id"], (list, tuple)) else meta["case_id"]
                    print(
                        f"Iter {iteration} | batch {batch_idx + 1} | case={case_id}: "
                        f"fwd(min/mean/max)=({flow_fwd.min().item():.4f}, {flow_fwd.mean().item():.4f}, {flow_fwd.max().item():.4f}) "
                        f"rev(min/mean/max)=({flow_rev.min().item():.4f}, {flow_rev.mean().item():.4f}, {flow_rev.max().item():.4f}) "
                        f"vel(min/mean/max)=({velocity.min().item():.4f}, {velocity.mean().item():.4f}, {velocity.max().item():.4f}) "
                        f"warped_ct(min/mean/max)=({warped_ct.min().item():.4f}, {warped_ct.mean().item():.4f}, {warped_ct.max().item():.4f}) "
                        f"warped_pet(min/mean/max)=({warped_pet.min().item():.4f}, {warped_pet.mean().item():.4f}, {warped_pet.max().item():.4f})"
                    )

                accum_ct += warped_ct.sum(dim=0, keepdim=True)
                accum_pet += warped_pet.sum(dim=0, keepdim=True)
                if USE_LOGDOMAIN_SHAPE_AVG:
                    vel_sum += velocity.detach().sum(dim=0, keepdim=True)
                else:
                    flow_sum += flow_fwd.sum(dim=0, keepdim=True)
                count += moving_ct.shape[0]

            template_ct = accum_ct / max(count, 1)
            template_pet = accum_pet / max(count, 1)
            print(
                f"Iter {iteration} | CT template pre-shape-avg stats: "
                f"min={template_ct.min().item():.4f}, mean={template_ct.mean().item():.4f}, max={template_ct.max().item():.4f}"
            )
            print(
                f"Iter {iteration} | PET template pre-shape-avg stats: "
                f"min={template_pet.min().item():.4f}, mean={template_pet.mean().item():.4f}, max={template_pet.max().item():.4f}"
            )

            if USE_LOGDOMAIN_SHAPE_AVG:
                avg_vel = vel_sum / max(count, 1)
                inv_disp = VecInt(avg_vel.shape[2:], SHAPE_AVG_NB_STEPS).to(device)(-avg_vel)
                template_ct = spatial_trans(template_ct, inv_disp)
                template_pet = spatial_trans(template_pet, inv_disp)
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
                template_ct = spatial_trans(template_ct, inv_disp)
                template_pet = spatial_trans(template_pet, inv_disp)
                print(
                    f"Iter {iteration} | shape-avg(flow) stats: "
                    f"avg_flow(min/mean/max)=({avg_flow.min().item():.4f}, {avg_flow.mean().item():.4f}, {avg_flow.max().item():.4f}) "
                    f"inv_disp(min/mean/max)=({inv_disp.min().item():.4f}, {inv_disp.mean().item():.4f}, {inv_disp.max().item():.4f})"
                )

            print(
                f"Iter {iteration} | CT template final stats: "
                f"min={template_ct.min().item():.4f}, mean={template_ct.mean().item():.4f}, max={template_ct.max().item():.4f}"
            )
            print(
                f"Iter {iteration} | PET template final stats: "
                f"min={template_pet.min().item():.4f}, mean={template_pet.mean().item():.4f}, max={template_pet.max().item():.4f}"
            )
            save_template(template_ct, template_affine, OUT_DIR, iteration, "CT")
            save_template(template_pet, template_affine, OUT_DIR, iteration, "PET")


if __name__ == "__main__":
    gpu_index = 0
    gpu_count = torch.cuda.device_count()
    print('Number of GPU: ' + str(gpu_count))
    for gpu_idx in range(gpu_count):
        print('     GPU #' + str(gpu_idx) + ': ' + torch.cuda.get_device_name(gpu_idx))
    if gpu_count > 0:
        torch.cuda.set_device(gpu_index)
        print('Currently using: ' + torch.cuda.get_device_name(gpu_index))
    print('If the GPU is available? ' + str(torch.cuda.is_available()))
    torch.manual_seed(0)
    build_template()
