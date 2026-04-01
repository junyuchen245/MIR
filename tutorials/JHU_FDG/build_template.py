import os
import sys
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nibabel as nib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MIR.models import SpatialTransformer, TransMorphTVF, VecInt, fit_warp_to_svf_fast, VFA, convex_adam_MIND
import MIR.models.configs_TransMorph as configs_TransMorph
import MIR.models.configs_VFA as configs_VFA
import MIR.models.convexAdam.configs_ConvexAdam_MIND as configs_ConvexAdam
from MIR import ModelWeights, DatasetJSONs
import gdown

INPUT_SHAPE = (160, 224, 192)
LUMIR_BASE_DIR = "/scratch2/jchen/DATA/JHU_FDG/"
LUMIR_JSON = "LUMIR_dataset.json"
CSV_PATH = "/scratch/jchen/python_projects/clinical_reports/batch78_img_clinical_os_merged_annotated.csv"
CSV_IMAGE_ROOT = "/scratch2/jchen/DATA/JHU_FDG/batch7/preprocessed"
CSV_BATCH_COLUMN = "batch"
CSV_BATCHES = ["batch7", "batch8"]
CSV_BATCH_IMAGE_ROOTS = {
    "batch7": "/scratch2/jchen/DATA/JHU_FDG/batch7/preprocessed",
    "batch8": "/scratch2/jchen/DATA/JHU_FDG/batch8/preprocessed",
}
CSV_IMAGE_COLUMN = ""  # e.g. "pet_path" if absolute/relative image path exists in csv
CSV_ID_COLUMN = "XNATSessionID"
CSV_IMAGE_SUFFIX = "_PET.nii.gz"
CSV_FILTER_FLAG = None  # examples: None, "metastatic", "primary", "disease_category:Cancer", "disease_subtype:NSCLC"
CSV_FILTER_COLUMN = None  # optional explicit column filter, e.g. "disease_category"
MODEL_TYPE = "ConvexAdam"  # "TransMorphTVF" | "VFA" | "ConvexAdam"
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

class PETCTCSVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        image_root,
        image_column="",
        id_column="XNATSessionID",
        image_suffix="_PET.nii.gz",
        filter_flag=None,
        filter_column=None,
        batch_column="batch",
        allowed_batches=None,
        batch_image_roots=None,
    ):
        self.items = []
        self.csv_path = csv_path
        self.image_root = image_root
        self.image_column = image_column
        self.id_column = id_column
        self.image_suffix = image_suffix
        self.filter_flag = filter_flag
        self.filter_column = filter_column
        self.batch_column = batch_column
        self.allowed_batches = set([(b or "").strip().lower() for b in (allowed_batches or [])])
        self.batch_image_roots = {
            (k or "").strip().lower(): v
            for k, v in (batch_image_roots or {}).items()
        }

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        missing_count = 0
        batch_filtered_count = 0
        for row in rows:
            if not self._row_matches_batch(row):
                batch_filtered_count += 1
                continue
            if not self._row_matches_flag(row):
                continue
            img_path = self._resolve_image_path(row)
            if img_path is None:
                missing_count += 1
                continue
            self.items.append({"img_path": img_path, "row": row, "meta": self._build_meta(row, img_path)})

        print(
            f"CSV dataset loaded from {csv_path}: selected={len(self.items)}, "
            f"missing_paths={missing_count}, batch_filtered={batch_filtered_count}, "
            f"batches={sorted(list(self.allowed_batches)) if self.allowed_batches else 'all'}, flag={filter_flag}"
        )

    def _get_row_batch(self, row):
        b = (row.get(self.batch_column, "") or "").strip().lower()
        if not b:
            b = (row.get("image_dir_batch", "") or "").strip().lower()
        return b

    def _row_matches_batch(self, row):
        if not self.allowed_batches:
            return True
        row_batch = self._get_row_batch(row)
        return row_batch in self.allowed_batches

    def _get_root_for_row(self, row):
        row_batch = self._get_row_batch(row)
        if row_batch and row_batch in self.batch_image_roots:
            return self.batch_image_roots[row_batch]
        return self.image_root

    def _norm_path(self, p, root=None):
        p = (p or "").strip()
        if not p:
            return None
        p = p.replace("\\", "/")
        if os.path.isabs(p):
            return p if os.path.isfile(p) else None
        if root is None:
            root = self.image_root
        p = os.path.join(root, p)
        return p if os.path.isfile(p) else None

    def _resolve_image_path(self, row):
        row_root = self._get_root_for_row(row)

        if self.image_column and row.get(self.image_column, "").strip():
            p = self._norm_path(row.get(self.image_column, ""), row_root)
            if p is not None:
                return p

        for col in ["image", "moving", "img_path", "pet_path", "PET_PATH"]:
            if row.get(col, "").strip():
                p = self._norm_path(row.get(col, ""), row_root)
                if p is not None:
                    return p

        sid = (row.get(self.id_column, "") or row.get("XNATSessionID", "")).strip()
        if sid:
            p = os.path.join(row_root, f"{sid}{self.image_suffix}")
            if os.path.isfile(p):
                return p
        return None

    def _build_meta(self, row, img_path):
        return {
            "img_path": img_path,
            "batch": row.get("batch", ""),
            "image_dir_batch": row.get("image_dir_batch", ""),
            "XNATSessionID": row.get("XNATSessionID", ""),
            "XNATSubjectID": row.get("XNATSubjectID", ""),
            "PID": row.get("PID", ""),
            "CSN": row.get("CSN", ""),
            "AccNum": row.get("AccNum", ""),
            "study_date": row.get("study_date", ""),
            "death_date": row.get("death_date", ""),
            "last_followup_date": row.get("last_followup_date", ""),
            "os_event": row.get("os_event", ""),
            "os_days": row.get("os_days", ""),
            "os_months": row.get("os_months", ""),
            "disease": row.get("disease", ""),
            "disease_category": row.get("disease_category", ""),
            "disease_subtype": row.get("disease_subtype", ""),
            "disease_details": row.get("disease_details", ""),
            "confidence": row.get("confidence", ""),
            "demo_Age": row.get("demo_Age", ""),
            "demo_Gender": row.get("demo_Gender", ""),
            "demo_Ethnicity": row.get("demo_Ethnicity", ""),
            "demo_Status": row.get("demo_Status", ""),
        }

    def _row_matches_flag(self, row):
        flag = (self.filter_flag or "")
        if not flag:
            return True
        flag_l = flag.lower().strip()

        if self.filter_column:
            return flag_l in (row.get(self.filter_column, "").lower())

        disease_details = (row.get("disease_details", "") or "").lower()
        disease = (row.get("disease", "") or "").lower()
        disease_category = (row.get("disease_category", "") or "").lower()
        disease_subtype = (row.get("disease_subtype", "") or "").lower()
        impression = (row.get("IMPRESSION", "") or "").lower()

        if flag_l == "metastatic":
            return ("extent=metastatic" in disease_details) or ("metast" in impression)

        if flag_l in {"primary", "primary_disease", "non_metastatic"}:
            if "cancer" not in disease_category and "carcin" not in disease and "lymphoma" not in disease:
                return False
            if "extent=metastatic" in disease_details or "metast" in impression:
                return False
            return True

        # Support flag syntax like: disease_category:Cancer or disease_subtype:NSCLC
        if ":" in flag:
            key, val = flag.split(":", 1)
            key = key.strip()
            val = val.strip().lower()
            if key in row:
                return val in (row.get(key, "").lower())

        blob = " | ".join([
            row.get("disease", ""),
            row.get("disease_category", ""),
            row.get("disease_subtype", ""),
            row.get("disease_details", ""),
            row.get("IMPRESSION", ""),
        ]).lower()
        return flag_l in blob

    def __getitem__(self, index):
        item = self.items[index]
        x = nib.load(item["img_path"])
        x = x.get_fdata().astype(np.float32)
        x = x[None, ...]
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        return x.float(), x.float(), item["meta"]

    def __len__(self):
        return len(self.items)

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
    if MODEL_TYPE == "ConvexAdam":
        return convex_adam_MIND, configs_ConvexAdam.get_ConvexAdam_MIND_brain_default_config()
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

def load_checkpoint(model, ckpt_path, device):
    if MODEL_TYPE == "ConvexAdam":
        return
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

    train_set = PETCTCSVDataset(
        csv_path=CSV_PATH,
        image_root=CSV_IMAGE_ROOT,
        image_column=CSV_IMAGE_COLUMN,
        id_column=CSV_ID_COLUMN,
        image_suffix=CSV_IMAGE_SUFFIX,
        filter_flag=CSV_FILTER_FLAG,
        filter_column=CSV_FILTER_COLUMN,
        batch_column=CSV_BATCH_COLUMN,
        allowed_batches=CSV_BATCHES,
        batch_image_roots=CSV_BATCH_IMAGE_ROOTS,
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
    if len(train_set) == 0:
        raise RuntimeError(
            "No training samples found. Check CSV_PATH, CSV_IMAGE_ROOT, CSV_ID_COLUMN/CSV_IMAGE_COLUMN, and CSV_FILTER_FLAG."
        )

    model = build_model(device, INPUT_SHAPE)
    if MODEL_TYPE == "ConvexAdam":
        model_fn, model_cfg = model
        print(f"Model type: {MODEL_TYPE}")
        print(f"SVF model output: {USE_SVF_MODEL}")
        model = (model_fn, model_cfg)
    else:
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
            for batch_data in train_loader:
                if len(batch_data) >= 3:
                    _, moving, meta = batch_data[:3]
                else:
                    _, moving = batch_data[:2]
                    meta = None
                moving = ensure_channel_dim(moving).float().to(device)
                template_batch = template.expand(moving.shape[0], -1, -1, -1, -1)

                if MODEL_TYPE == "TransMorphTVF":
                    moving_half = F.avg_pool3d(moving, 2)
                    template_half = F.avg_pool3d(template_batch, 2)
                    flow = model((moving_half, template_half))
                    flow = F.interpolate(flow, scale_factor=2, mode="trilinear", align_corners=False) * 2
                elif MODEL_TYPE == "ConvexAdam":
                    model_fn, model_cfg = model
                    with torch.enable_grad():
                        flow = model_fn(moving, template_batch, model_cfg)
                else:
                    flow = model((moving, template_batch))
                warped = spatial_trans(moving, flow)

                if count % LOG_EVERY == 0:
                    meta_tag = ""
                    if isinstance(meta, dict):
                        sid = meta.get("XNATSessionID", [""])
                        bch = meta.get("batch", [""])
                        sid_str = sid[0] if isinstance(sid, list) and len(sid) > 0 else sid
                        bch_str = bch[0] if isinstance(bch, list) and len(bch) > 0 else bch
                        meta_tag = f" sample={sid_str} batch={bch_str}"
                    print(
                        f"Iter {iteration} | batch {count + 1}: "
                        f"flow(min/mean/max)=({flow.min().item():.4f}, {flow.mean().item():.4f}, {flow.max().item():.4f}) "
                        f"warped(min/mean/max)=({warped.min().item():.4f}, {warped.mean().item():.4f}, {warped.max().item():.4f})"
                        f"{meta_tag}"
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
