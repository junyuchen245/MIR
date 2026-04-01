from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.filebasedimages import ImageFileError
from natsort import natsorted
import torch.nn as nn
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MIR.models import EncoderFeatureExtractor, SITReg, SpatialTransformer, TemplateCreation, VFA
from MIR.models.SITReg import GroupNormalizerFactory, ReLUFactory
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.models.SITReg.deformation_inversion_layer.fixed_point_iteration import (
    AndersonSolver,
    AndersonSolverArguments,
    MaxElementWiseAbsStopCriterion,
    RelativeL2ErrorStopCriterion,
)


DEFAULT_STEP1_ROOT = THIS_DIR / 'wholebody_petct_sitreg_step1'
DEFAULT_JHU_FDG_BATCH7_DIR = Path('/scratch2/jchen/DATA/JHU_FDG/batch7/preprocessed')
DEFAULT_JHU_FDG_BATCH8_DIR = Path('/scratch2/jchen/DATA/JHU_FDG/batch8/preprocessed')
DEFAULT_JHU_FDG_BATCH7_SEG_DIR = Path('/scratch2/jchen/DATA/JHU_FDG/batch7/preprocessed')
DEFAULT_JHU_FDG_BATCH8_SEG_DIR = Path('/scratch2/jchen/DATA/JHU_FDG/batch8/preprocessed')
DEFAULT_JHU_FDG_CLINICAL = Path('/scratch/jchen/python_projects/clinical_reports/batch78_img_clinical_os_merged_annotated.csv')
DEFAULT_PSMA_DIR = Path('/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU')
DEFAULT_DATASET_FLIP_AXES_TO_JHU_FDG: Dict[str, Tuple[int, ...]] = {
    'jhu_fdg': (),
    'jhu_psma': (1,),
}


@dataclass(frozen=True)
class SegAtlasRecord:
    stem: str
    dataset: str
    sex: str
    image_path: str
    label_path: str


def clean_text(value: object) -> str:
    if value is None:
        return ''
    return str(value).strip()


def choose_first(record: Dict[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        text = clean_text(record.get(key))
        if text:
            return text
    return ''


def normalize_sex(value: object) -> Optional[str]:
    text = clean_text(value).lower()
    if text in {'m', 'male', 'man', 'men'}:
        return 'men'
    if text in {'f', 'female', 'woman', 'women'}:
        return 'women'
    return None


def parse_flip_axes(text: str) -> Tuple[int, ...]:
    cleaned = clean_text(text).lower()
    if not cleaned or cleaned in {'none', 'no', 'off'}:
        return ()
    axes = []
    for token in cleaned.split('|'):
        axis = int(token)
        if axis not in {0, 1, 2}:
            raise ValueError(f'Flip axis must be 0, 1, or 2; got {axis}.')
        axes.append(axis)
    return tuple(axes)


def parse_dataset_flip_config(config_text: str) -> Dict[str, Tuple[int, ...]]:
    dataset_flips = dict(DEFAULT_DATASET_FLIP_AXES_TO_JHU_FDG)
    if not clean_text(config_text):
        return dataset_flips
    for item in config_text.split(','):
        entry = clean_text(item)
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError(f"Invalid dataset flip entry '{entry}'. Expected dataset:axes.")
        dataset_name, axes_text = entry.split(':', 1)
        dataset_name = clean_text(dataset_name)
        if not dataset_name:
            raise ValueError(f"Invalid dataset flip entry '{entry}'. Dataset name is empty.")
        dataset_flips[dataset_name] = parse_flip_axes(axes_text)
    return dataset_flips


def load_jhu_fdg_sex_map(clinical_csv: Path) -> Dict[str, str]:
    csv.field_size_limit(sys.maxsize)
    mapping: Dict[str, str] = {}
    with clinical_csv.open('r', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_id = choose_first(row, ['XNATSessionID', 'patient_id', 'XnatSessionID'])
            tracer = choose_first(row, ['tracer', 'Tracer'])
            sex = normalize_sex(choose_first(row, ['demo_Gender', 'Gender', 'gender']))
            if not session_id or sex is None:
                continue
            if 'fdg' not in tracer.lower():
                continue
            mapping.setdefault(session_id, sex)
    return mapping


def build_fdg_ct_records(
    preprocessed_roots: Sequence[Path],
    seg_roots: Sequence[Path],
    sex_map: Dict[str, str],
) -> List[SegAtlasRecord]:
    records: List[SegAtlasRecord] = []
    for preprocessed_root, seg_root in zip(preprocessed_roots, seg_roots):
        for ct_path in sorted(preprocessed_root.glob('*_CT.nii.gz')):
            stem = ct_path.name[:-len('_CT.nii.gz')]
            sex = sex_map.get(stem)
            seg_path = seg_root / f'{stem}_vista_CTSeg.nii.gz'
            if sex is None or not seg_path.exists():
                continue
            records.append(
                SegAtlasRecord(
                    stem=stem,
                    dataset='jhu_fdg',
                    sex=sex,
                    image_path=str(ct_path),
                    label_path=str(seg_path),
                )
            )
    return records


def build_psma_pet_records(psma_dir: Path) -> List[SegAtlasRecord]:
    records: List[SegAtlasRecord] = []
    for ct_path in sorted(psma_dir.glob('*_CT.nii.gz')):
        stem = ct_path.name[:-len('_CT.nii.gz')]
        if stem.endswith('_CT_seg'):
            continue
        suv_seg_path = psma_dir / f'{stem}_SUV_seg.nii.gz'
        if not suv_seg_path.exists():
            continue
        records.append(
            SegAtlasRecord(
                stem=stem,
                dataset='jhu_psma',
                sex='men',
                image_path=str(ct_path),
                label_path=str(suv_seg_path),
            )
        )
    return records


def normalize_ct(image: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(image, dtype=np.float32), -300.0, 300.0)
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max <= image_min:
        return np.zeros_like(image, dtype=np.float32)
    image = (image - image_min) / (image_max - image_min)
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def remap_psma_suv_lbl(lbl: np.ndarray, include_lesions: bool = False) -> np.ndarray:
    grouping_table = [
        [101],
        [201],
        [501],
        [601],
        [701],
        [801],
        [901],
        [1001],
        [1101],
        [1201],
        [1301],
        [1401],
        [1501],
        [102, 103, 104, 202, 203, 204, 302, 303, 304, 402, 403, 404, 502, 503, 504,
         602, 603, 604, 702, 703, 704, 802, 803, 804, 902, 903, 904, 1002, 1003, 1004,
         1102, 1103, 1104, 1202, 1203, 1204, 1302, 1303, 1304, 1402, 1403, 1404, 1502, 1503, 1504,
         1602, 1603, 1604, 1802, 1803, 1804],
    ]
    if not include_lesions:
        grouping_table = grouping_table[:-1]
    label_out = np.zeros_like(lbl, dtype=np.int16)
    for index, group in enumerate(grouping_table, start=1):
        for seg_i in group:
            label_out[lbl == seg_i] = index
    return label_out


def load_array(
    path: str,
    dataset: str,
    dataset_flip_axes: Dict[str, Tuple[int, ...]],
    standardize_orientation: bool,
) -> np.ndarray:
    image_nib = nib.load(path)
    if standardize_orientation:
        image_nib = nib.as_closest_canonical(image_nib, enforce_diag=False)
    image = image_nib.get_fdata()
    if standardize_orientation:
        for axis in dataset_flip_axes.get(dataset, ()):
            image = np.flip(image, axis=axis)
    return np.ascontiguousarray(image)


def load_image_tensor(
    path: str,
    dataset: str,
    target_shape: Sequence[int],
    dataset_flip_axes: Dict[str, Tuple[int, ...]],
    standardize_orientation: bool,
) -> torch.Tensor:
    image = normalize_ct(load_array(path, dataset, dataset_flip_axes, standardize_orientation))
    tensor = torch.from_numpy(image)[None, None]
    if tuple(tensor.shape[-3:]) != tuple(target_shape):
        tensor = F.interpolate(tensor, size=tuple(target_shape), mode='trilinear', align_corners=False)
    return torch.clamp(tensor.float(), min=0.0, max=1.0)


def load_label_tensor(
    path: str,
    dataset: str,
    target_shape: Sequence[int],
    dataset_flip_axes: Dict[str, Tuple[int, ...]],
    standardize_orientation: bool,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> torch.Tensor:
    label = load_array(path, dataset, dataset_flip_axes, standardize_orientation)
    label = np.rint(label).astype(np.int16)
    if transform is not None:
        label = transform(label)
    tensor = torch.from_numpy(label.astype(np.float32))[None, None]
    if tuple(tensor.shape[-3:]) != tuple(target_shape):
        tensor = F.interpolate(tensor, size=tuple(target_shape), mode='nearest')
    return tensor.float()


def create_sitreg_model(input_shape: Sequence[int], device: torch.device) -> SITReg:
    feature_extractor = EncoderFeatureExtractor(
        n_input_channels=1,
        activation_factory=ReLUFactory(),
        n_features_per_resolution=[12, 16, 32, 64, 128, 128],
        n_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
        input_shape=tuple(input_shape),
        normalizer_factory=GroupNormalizerFactory(2),
    ).to(device)
    anderson_solver_forward = AndersonSolver(
        MaxElementWiseAbsStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
        AndersonSolverArguments(memory_length=4),
    )
    anderson_solver_backward = AndersonSolver(
        RelativeL2ErrorStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
        AndersonSolverArguments(memory_length=4),
    )
    return SITReg(
        feature_extractor=feature_extractor,
        n_transformation_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
        n_transformation_features_per_resolution=[12, 64, 128, 256, 256, 256],
        max_control_point_multiplier=0.99,
        affine_transformation_type=None,
        input_voxel_size=(1.0, 1.0, 1.0),
        input_shape=tuple(input_shape),
        transformation_downsampling_factor=(1.0, 1.0, 1.0),
        forward_fixed_point_solver=anderson_solver_forward,
        backward_fixed_point_solver=anderson_solver_backward,
        activation_factory=ReLUFactory(),
        normalizer_factory=GroupNormalizerFactory(4),
    ).to(device)


def build_registration_model(model_type: str, input_shape: Sequence[int], device: torch.device) -> TemplateCreation:
    if model_type == 'VFA':
        config = CONFIGS_VFA.get_VFA_default_config()
        config.img_size = tuple(input_shape)
        base_model = VFA(config, device=str(device), SVF=True, return_full=True).to(device)
        return TemplateCreation(base_model, tuple(input_shape)).to(device)
    if model_type == 'SITReg':
        base_model = create_sitreg_model(tuple(input_shape), device)
        return TemplateCreation(base_model, tuple(input_shape), use_sitreg=True).to(device)
    raise ValueError(f'Unsupported model type: {model_type}')


def latest_checkpoint_file(experiments_dir: Path) -> Path:
    files = natsorted(experiments_dir.glob('*.pth.tar'))
    if not files:
        raise FileNotFoundError(f'No checkpoint files found in {experiments_dir}')
    return files[-1]


def resolve_atlas_file(folder: Path) -> Path:
    preview_path = folder / 'epoch0006_ssim0.9546.nii.gz'
    if preview_path.exists():
        return preview_path
    files = natsorted(folder.glob('*.nii.gz'))
    if not files:
        raise FileNotFoundError(f'No atlas NIfTI files found in {folder}')
    return files[-1]


def labels_to_one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    y_seg_oh = nn.functional.one_hot(label.cpu().long(), num_classes=num_classes)
    y_seg_oh = torch.squeeze(y_seg_oh, 1)
    y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
    return y_seg_oh


def compute_hard_atlas_from_probabilities(
    prob_sums: np.memmap,
    observed_labels: np.ndarray,
    chunk_size: int = 131072,
) -> np.ndarray:
    num_classes = prob_sums.shape[0]
    flat_probs = prob_sums.reshape(num_classes, -1)
    flat_out = np.zeros(flat_probs.shape[1], dtype=np.uint16)
    for start in range(0, flat_probs.shape[1], chunk_size):
        end = min(start + chunk_size, flat_probs.shape[1])
        dense_indices = np.argmax(flat_probs[:, start:end], axis=0).astype(np.int64)
        flat_out[start:end] = observed_labels[dense_indices].astype(np.uint16)
    return flat_out.reshape(prob_sums.shape[1:])


def save_nifti(array: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), str(output_path))

def save_npz(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), array=array)

def write_summary(output_path: Path, lines: Iterable[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as handle:
        for line in lines:
            handle.write(f'{line}\n')


def make_progress(iterable, *, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def extract_model_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f'Unsupported checkpoint format: {type(checkpoint)!r}')

def remap_ct_lbls(lbl: np.ndarray) -> np.ndarray:
    remap_table = {
        156: 118,
        157: 119,
        158: 120,
        159: 121,
        163: 122,
        164: 123, 
    }
    lbl_out = np.copy(lbl)
    for src_lbl, dst_lbl in remap_table.items():
        lbl_out[lbl == src_lbl] = dst_lbl
    return lbl_out

def run_atlas_generation(
    *,
    title: str,
    records: Sequence[SegAtlasRecord],
    atlas_tensor: torch.Tensor,
    atlas_affine: np.ndarray,
    output_dir: Path,
    checkpoint_path: Path,
    model_type: str,
    device: torch.device,
    dataset_flip_axes: Dict[str, Tuple[int, ...]],
    standardize_orientation: bool,
    num_classes: int,
    label_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    enhance_ribs: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_shape = tuple(int(v) for v in atlas_tensor.shape[-3:])
    model = build_registration_model(model_type, target_shape, device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    label_spatial_transform = getattr(model.reg_model, 'spatial_trans', None)
    if label_spatial_transform is None:
        label_spatial_transform = SpatialTransformer(target_shape).to(device)
        
    prob_sums = torch.zeros((1, num_classes,) + target_shape, dtype=torch.float32, device='cpu')
    case_count = 0
    skipped_cases: List[Tuple[str, str]] = []

    print(f'{title}: {len(records)} input cases')
    with torch.no_grad():
        progress = make_progress(records, total=len(records), desc=title)
        for index, record in enumerate(progress, start=1):
            try:
                moving_image = load_image_tensor(
                    record.image_path,
                    record.dataset,
                    target_shape,
                    dataset_flip_axes,
                    standardize_orientation,
                ).to(device)
                moving_label = load_label_tensor(
                    record.label_path,
                    record.dataset,
                    target_shape,
                    dataset_flip_axes,
                    standardize_orientation,
                    transform=label_transform,
                ).to(device)
            except (FileNotFoundError, ImageFileError, OSError, EOFError, ValueError) as exc:
                skipped_cases.append((record.stem, str(exc)))
                if tqdm is not None:
                    progress.set_postfix_str(f'skipped:{record.stem}')
                else:
                    print(f'  skipped {record.stem}: {exc}')
                continue
            moving_label_oh = labels_to_one_hot(moving_label, num_classes)
            outputs = model((atlas_tensor, moving_image))
            neg_flow = outputs[3]
            def_segs = []
            for i in range(num_classes):
                def_seg = label_spatial_transform(moving_label_oh[:, i:i + 1, ...].to(device).float(), neg_flow.float())
                def_segs.append(def_seg.cpu())
            def_segs = torch.cat(def_segs, dim=1)
            prob_sums += def_segs
            case_count += 1
            if tqdm is not None:
                progress.set_postfix_str(f'{record.stem} | labels={num_classes}')
            elif index % 25 == 0 or index == len(records):
                print(f'  processed {index}/{len(records)} | last case: {record.stem}')

    if case_count > 0 and prob_sums is not None:
        prob_sums /= float(case_count)
        for i in range(0, num_classes):
            save_nifti(prob_sums[0, i].cpu().detach().numpy(), atlas_affine, output_dir / 'prob_atlas' / f'seg_atlas_class{i}.nii.gz')
        if enhance_ribs:
            for i in range(93, 117):
                prob_sums[:, i, :] = prob_sums[:, i, :] * 6
        hard_atlas = torch.argmax(prob_sums, dim=1, keepdim=True)
        save_nifti(hard_atlas[0, 0].cpu().detach().numpy().astype(np.uint16), atlas_affine, output_dir / f'seg_atlas_{num_classes}lbls.nii.gz')

    write_summary(
        output_dir / 'atlas_generation_summary.txt',
        [
            f'title={title}',
            f'model_type={model_type}',
            f'checkpoint={checkpoint_path}',
            f'num_classes={num_classes}',
            'aggregation=mean_warped_one_hot_labels',
            'dynamic_label_accumulation=1',
            f'case_count={case_count}',
            f'skipped_case_count={len(skipped_cases)}',
            f'standardize_orientation={int(standardize_orientation)}',
            *[f'skipped_case={stem}\t{reason}' for stem, reason in skipped_cases],
        ],
    )
    return case_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate whole-body segmentation atlases using the STEP1 wholebody SITReg/VFA model: '
            'sex-specific JHU FDG CT atlases from external CT segmentations and a JHU PSMA SUV atlas with lesion labels ignored.'
        )
    )
    parser.add_argument('--model-type', default='SITReg', choices=['SITReg', 'VFA'])
    parser.add_argument('--step1-root', default=str(DEFAULT_STEP1_ROOT), help='STEP1 output root containing atlas/ and experiments/.')
    parser.add_argument('--checkpoint', default=None, help='Optional checkpoint path. Defaults to the latest .pth.tar under <step1-root>/experiments/.')
    parser.add_argument('--fdg-batch7-dir', default=str(DEFAULT_JHU_FDG_BATCH7_DIR))
    parser.add_argument('--fdg-batch8-dir', default=str(DEFAULT_JHU_FDG_BATCH8_DIR))
    parser.add_argument('--fdg-batch7-seg-dir', default=str(DEFAULT_JHU_FDG_BATCH7_SEG_DIR))
    parser.add_argument('--fdg-batch8-seg-dir', default=str(DEFAULT_JHU_FDG_BATCH8_SEG_DIR))
    parser.add_argument('--fdg-clinical-csv', default=str(DEFAULT_JHU_FDG_CLINICAL))
    parser.add_argument('--psma-dir', default=str(DEFAULT_PSMA_DIR))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fdg-ct-num-classes', type=int, default=124)
    parser.add_argument('--psma-pet-num-classes', type=int, default=15)
    parser.add_argument(
        '--dataset-flips',
        default='jhu_fdg:none,jhu_psma:1',
        help='Dataset-specific voxel flips to align volumes to the JHU FDG convention. Format: dataset:axis|axis,dataset:none',
    )
    parser.add_argument(
        '--disable-orientation-standardization',
        action='store_true',
        help='Disable canonical reorientation plus dataset-specific flips. Use only for debugging if standardized orientation looks wrong.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step1_root = Path(args.step1_root)
    experiments_dir = step1_root / 'experiments'
    atlas_ct_root = step1_root / 'atlas' / 'ct'
    atlas_pet_root = step1_root / 'atlas' / 'pet'
    output_root = step1_root / 'atlas' / 'seg'
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if ('cuda' in args.device and torch.cuda.is_available()) else 'cpu')
    dataset_flip_axes = parse_dataset_flip_config(args.dataset_flips)
    standardize_orientation = not args.disable_orientation_standardization
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint_file(experiments_dir)

    sex_map = load_jhu_fdg_sex_map(Path(args.fdg_clinical_csv))
    fdg_records = build_fdg_ct_records(
        [Path(args.fdg_batch7_dir), Path(args.fdg_batch8_dir)],
        [Path(args.fdg_batch7_seg_dir), Path(args.fdg_batch8_seg_dir)],
        sex_map,
    )
    psma_records = build_psma_pet_records(Path(args.psma_dir))

    print(f'Using device: {device}')
    print(f'Model type: {args.model_type}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'STEP1 root: {step1_root}')
    print(f'JHU FDG CT cases with external seg: {len(fdg_records)}')
    print(f'JHU PSMA SUV cases with SUV seg: {len(psma_records)}')

    summary_lines = [
        f'model_type={args.model_type}',
        f'checkpoint={checkpoint_path}',
        f'step1_root={step1_root}',
        f'fdg_total_cases={len(fdg_records)}',
        f'psma_total_cases={len(psma_records)}',
        f'standardize_orientation={int(standardize_orientation)}',
    ]
    '''
    for sex in ('men', 'women'):
        atlas_ct_path = resolve_atlas_file(atlas_ct_root / sex)
        atlas_ct_nib = nib.load(str(atlas_ct_path))
        atlas_ct_np = atlas_ct_nib.get_fdata().astype(np.float32)
        atlas_ct_tensor = torch.from_numpy(atlas_ct_np[None, None]).to(device).float()
        sex_records = [record for record in fdg_records if record.sex == sex]
        case_count = run_atlas_generation(
            title=f'JHU FDG CT segmentation atlas ({sex})',
            records=sex_records,
            atlas_tensor=atlas_ct_tensor,
            atlas_affine=atlas_ct_nib.affine,
            output_dir=output_root / 'ct' / sex,
            checkpoint_path=checkpoint_path,
            model_type=args.model_type,
            device=device,
            dataset_flip_axes=dataset_flip_axes,
            standardize_orientation=standardize_orientation,
            num_classes=args.fdg_ct_num_classes,
            label_transform=remap_ct_lbls,
            enhance_ribs=True,
        )
        summary_lines.append(f'fdg_ct_{sex}_cases={case_count}')
        summary_lines.append(f'fdg_ct_{sex}_atlas={atlas_ct_path}')
    '''
    psma_ct_atlas_path = resolve_atlas_file(atlas_ct_root / 'men')
    psma_ct_atlas_nib = nib.load(str(psma_ct_atlas_path))
    psma_ct_atlas_tensor = torch.from_numpy(psma_ct_atlas_nib.get_fdata().astype(np.float32)[None, None]).to(device).float()
    psma_pet_atlas_path = resolve_atlas_file(atlas_pet_root / 'psma_men')
    psma_pet_atlas_nib = nib.load(str(psma_pet_atlas_path))
    psma_case_count = run_atlas_generation(
        title='JHU PSMA SUV segmentation atlas (lesions ignored)',
        records=psma_records,
        atlas_tensor=psma_ct_atlas_tensor,
        atlas_affine=psma_ct_atlas_nib.affine,
        output_dir=output_root / 'pet' / 'psma_men',
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        device=device,
        dataset_flip_axes=dataset_flip_axes,
        standardize_orientation=standardize_orientation,
        num_classes=args.psma_pet_num_classes,
        label_transform=remap_psma_suv_lbl,
        enhance_ribs=False,
    )
    summary_lines.append(f'psma_pet_cases={psma_case_count}')
    summary_lines.append(f'psma_ct_atlas={psma_ct_atlas_path}')
    summary_lines.append(f'psma_pet_atlas={psma_pet_atlas_path}')

    write_summary(output_root / 'atlas_generation_summary.txt', summary_lines)
    print(f'Saved outputs under: {output_root}')


if __name__ == '__main__':
    gpu_count = torch.cuda.device_count()
    print('Number of GPU: ' + str(gpu_count))
    for gpu_idx in range(gpu_count):
        print('     GPU #' + str(gpu_idx) + ': ' + torch.cuda.get_device_name(gpu_idx))
    if gpu_count > 0:
        torch.cuda.set_device(0)
        print('Currently using: ' + torch.cuda.get_device_name(0))
    print('If the GPU is available? ' + str(torch.cuda.is_available()))
    torch.manual_seed(42)
    np.random.seed(42)
    main()