import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MIR.models import TemplateCreation, SpatialTransformer, VFA
from MIR.models import EncoderFeatureExtractor, SITReg
from MIR.models.SITReg import GroupNormalizerFactory, ReLUFactory
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.models.SITReg.deformation_inversion_layer.fixed_point_iteration import (
    AndersonSolver,
    AndersonSolverArguments,
    MaxElementWiseAbsStopCriterion,
    RelativeL2ErrorStopCriterion,
)


INPUT_SHAPE = (192, 192, 144)
DEFAULT_PREPROCESSED_DIR = '/scratch2/jchen/DATA/HECKTOR25/preprocessed/'
DEFAULT_CTSEG_NUM_CLASSES = 118
DEFAULT_LESION_NUM_CLASSES = 3


class Hecktor25Dataset(Dataset):
    def __init__(self, data_dir, case_ids):
        self.data_dir = data_dir
        self.case_ids = list(case_ids)

    def __len__(self):
        return len(self.case_ids)

    def norm_ct(self, img):
        x = img.copy()
        x[x < -300] = -300
        x[x > 300] = 300
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        ct_path = os.path.join(self.data_dir, f'{case_id}_CT.nii.gz')
        ctseg_path = os.path.join(self.data_dir, f'{case_id}_CTSeg.nii.gz')
        seg_path = os.path.join(self.data_dir, f'{case_id}_SEG.nii.gz')

        ct = nib.load(ct_path).get_fdata().astype(np.float32)
        ct = self.norm_ct(ct)

        has_ctseg = os.path.exists(ctseg_path)
        has_seg = os.path.exists(seg_path)

        if has_ctseg:
            ctseg = np.rint(nib.load(ctseg_path).get_fdata()).astype(np.int16)
        else:
            ctseg = np.zeros_like(ct, dtype=np.int16)

        if has_seg:
            seg = np.rint(nib.load(seg_path).get_fdata()).astype(np.int16)
        else:
            seg = np.zeros_like(ct, dtype=np.int16)

        return {
            'case_id': case_id,
            'ct': torch.from_numpy(np.ascontiguousarray(ct[None, ...])).float(),
            'ctseg': torch.from_numpy(np.ascontiguousarray(ctseg[None, ...])).float(),
            'seg': torch.from_numpy(np.ascontiguousarray(seg[None, ...])).float(),
            'has_ctseg': torch.tensor(int(has_ctseg), dtype=torch.int64),
            'has_seg': torch.tensor(int(has_seg), dtype=torch.int64),
        }


def create_sitreg_model(input_shape, device):
    feature_extractor = EncoderFeatureExtractor(
        n_input_channels=1,
        activation_factory=ReLUFactory(),
        n_features_per_resolution=[12, 16, 32, 64, 128, 128],
        n_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
        input_shape=input_shape,
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
        input_shape=input_shape,
        transformation_downsampling_factor=(1.0, 1.0, 1.0),
        forward_fixed_point_solver=anderson_solver_forward,
        backward_fixed_point_solver=anderson_solver_backward,
        activation_factory=ReLUFactory(),
        normalizer_factory=GroupNormalizerFactory(4),
    ).to(device)


def build_registration_model(model_type, input_shape, device):
    if model_type == 'VFA':
        config = CONFIGS_VFA.get_VFA_default_config()
        config.img_size = input_shape
        base_model = VFA(config, device=str(device), SVF=True, return_full=True).to(device)
        model = TemplateCreation(base_model, input_shape).to(device)
    elif model_type == 'SITReg':
        base_model = create_sitreg_model(input_shape, device)
        model = TemplateCreation(base_model, input_shape, use_sitreg=True).to(device)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')
    return model


def latest_file_in_dir(folder, suffix):
    files = natsorted(glob.glob(os.path.join(folder, f'*{suffix}')))
    if not files:
        raise FileNotFoundError(f'No files ending with {suffix!r} found in {folder}')
    return files[-1]


def resolve_experiment_name(model_type, experiment_name=None):
    if experiment_name:
        return experiment_name
    pattern = os.path.join(THIS_DIR, 'experiments', f'{model_type}Atlas_*')
    matches = natsorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No experiment folders found for model type {model_type}')
    return os.path.basename(matches[-1])


def collect_case_ids(data_dir):
    case_ids = []
    for ct_path in natsorted(glob.glob(os.path.join(data_dir, '*_CT.nii.gz'))):
        case_ids.append(os.path.basename(ct_path)[:-len('_CT.nii.gz')])
    if not case_ids:
        raise RuntimeError(f'No HECKTOR25 CT volumes found in {data_dir}')
    return case_ids


def accumulate_label_counts(label_counts, warped_labels, voxel_indices):
    num_classes = label_counts.shape[0]
    voxels_per_class = int(np.prod(label_counts.shape[1:]))
    labels_flat = warped_labels.reshape(-1).astype(np.int64)
    valid = (labels_flat >= 0) & (labels_flat < num_classes)
    flat_indices = labels_flat[valid] * voxels_per_class + voxel_indices[valid]
    np.add.at(label_counts.reshape(-1), flat_indices, 1)


def compute_hard_atlas_from_counts(label_counts, chunk_size=131072):
    num_classes = label_counts.shape[0]
    flat_counts = label_counts.reshape(num_classes, -1)
    flat_out = np.zeros(flat_counts.shape[1], dtype=np.uint16)
    for start in range(0, flat_counts.shape[1], chunk_size):
        end = min(start + chunk_size, flat_counts.shape[1])
        flat_out[start:end] = np.argmax(flat_counts[:, start:end], axis=0).astype(np.uint16)
    return flat_out.reshape(label_counts.shape[1:])


def save_nifti(array, affine, out_path):
    nib.save(nib.Nifti1Image(array, affine), out_path)


def save_npz(array, out_path):
    np.savez_compressed(out_path, array=array)


def make_progress(iterable, *, total, desc):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        return checkpoint
    raise TypeError(f'Unsupported checkpoint format: {type(checkpoint)!r}')


def labels_to_one_hot(label_tensor, num_classes):
    label_long = torch.round(label_tensor.cpu()).long()
    label_long = torch.clamp(label_long, min=0, max=num_classes - 1)
    one_hot = nn.functional.one_hot(label_long, num_classes=num_classes)
    one_hot = torch.squeeze(one_hot, 1)
    return one_hot.permute(0, 4, 1, 2, 3).contiguous().float()


def compute_hard_atlas_from_probabilities(prob_sums, chunk_size=131072):
    num_classes = prob_sums.shape[0]
    flat_probs = prob_sums.reshape(num_classes, -1)
    flat_out = np.zeros(flat_probs.shape[1], dtype=np.uint16)
    for start in range(0, flat_probs.shape[1], chunk_size):
        end = min(start + chunk_size, flat_probs.shape[1])
        flat_out[start:end] = np.argmax(flat_probs[:, start:end], axis=0).astype(np.uint16)
    return flat_out.reshape(prob_sums.shape[1:])


def parse_args():
    parser = argparse.ArgumentParser(description='Generate HECKTOR25 segmentation atlases from a trained atlas-building model.')
    parser.add_argument('--model-type', default='SITReg', choices=['SITReg', 'VFA'])
    parser.add_argument('--experiment-name', default=None, help='Optional experiment folder name under experiments/.')
    parser.add_argument('--data-dir', default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ctseg-num-classes', type=int, default=DEFAULT_CTSEG_NUM_CLASSES)
    parser.add_argument('--lesion-num-classes', type=int, default=DEFAULT_LESION_NUM_CLASSES)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if ('cuda' in args.device and torch.cuda.is_available()) else 'cpu')
    experiment_name = resolve_experiment_name(args.model_type, args.experiment_name)

    checkpoint_dir = os.path.join(THIS_DIR, 'experiments', experiment_name)
    atlas_ct_dir = os.path.join(THIS_DIR, 'atlas', 'ct', experiment_name)
    output_dir = os.path.join(THIS_DIR, 'atlas', 'seg', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = latest_file_in_dir(checkpoint_dir, '.pth.tar')
    atlas_ct_path = latest_file_in_dir(atlas_ct_dir, '.nii.gz')

    print(f'Using device: {device}')
    print(f'Model type: {args.model_type}')
    print(f'Experiment: {experiment_name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Atlas CT: {atlas_ct_path}')
    print(f'Preprocessed dir: {args.data_dir}')

    atlas_ct_nib = nib.load(atlas_ct_path)
    atlas_ct_np = atlas_ct_nib.get_fdata().astype(np.float32)
    atlas_ct_t = torch.from_numpy(atlas_ct_np[None, None, ...]).to(device).float()

    case_ids = collect_case_ids(args.data_dir)
    dataset = Hecktor25Dataset(args.data_dir, case_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    model = build_registration_model(args.model_type, INPUT_SHAPE, device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    label_spatial_transform = getattr(model.reg_model, 'spatial_trans', None)
    if label_spatial_transform is None:
        label_spatial_transform = SpatialTransformer(INPUT_SHAPE).to(device)

    ctseg_prob_sums = np.zeros((args.ctseg_num_classes,) + INPUT_SHAPE, dtype=np.float32)
    lesion_prob_sums = np.zeros((args.lesion_num_classes,) + INPUT_SHAPE, dtype=np.float32)

    ctseg_case_count = 0
    lesion_case_count = 0

    with torch.no_grad():
        progress = make_progress(loader, total=len(loader), desc='Generate HECKTOR25 atlases')
        for batch_idx, batch in enumerate(progress, start=1):
            case_id = batch['case_id'][0] if isinstance(batch['case_id'], (list, tuple)) else batch['case_id']
            ct = batch['ct'].to(device).float()
            atlas_ct_batch = atlas_ct_t.repeat(ct.shape[0], 1, 1, 1, 1)
            outputs = model((atlas_ct_batch, ct))
            neg_flow = outputs[3]

            if int(batch['has_ctseg'][0].item()) == 1:
                ctseg = batch['ctseg'].to(device).float()
                ctseg_oh = labels_to_one_hot(ctseg, args.ctseg_num_classes)
                ctseg_def = []
                for class_index in range(args.ctseg_num_classes):
                    warped = label_spatial_transform(ctseg_oh[:, class_index:class_index + 1, ...].to(device).float(), neg_flow.float())
                    ctseg_def.append(warped.cpu())
                ctseg_warp = torch.cat(ctseg_def, dim=1)
                ctseg_prob_sums += ctseg_warp.detach().cpu().numpy().sum(axis=0).astype(np.float32)
                ctseg_case_count += 1

            if int(batch['has_seg'][0].item()) == 1:
                seg = batch['seg'].to(device).float()
                seg_oh = labels_to_one_hot(seg, args.lesion_num_classes)
                seg_def = []
                for class_index in range(args.lesion_num_classes):
                    warped = label_spatial_transform(seg_oh[:, class_index:class_index + 1, ...].to(device).float(), neg_flow.float())
                    seg_def.append(warped.cpu())
                seg_warp = torch.cat(seg_def, dim=1)
                lesion_prob_sums += seg_warp.detach().cpu().numpy().sum(axis=0).astype(np.float32)
                lesion_case_count += 1

            if tqdm is not None:
                progress.set_postfix_str(str(case_id))
            elif batch_idx % 25 == 0 or batch_idx == len(loader):
                print(
                    f'Processed {batch_idx}/{len(loader)} cases | '
                    f'ctseg cases: {ctseg_case_count} | lesion cases: {lesion_case_count} | '
                    f'last case: {case_id}'
                )

    if ctseg_case_count > 0:
        ctseg_prob_sums /= float(ctseg_case_count)
        ctseg_prob_dir = os.path.join(output_dir, 'prob_atlas', 'ctseg')
        os.makedirs(ctseg_prob_dir, exist_ok=True)
        for class_index in range(args.ctseg_num_classes):
            save_nifti(
                ctseg_prob_sums[class_index].astype(np.float32),
                atlas_ct_nib.affine,
                os.path.join(ctseg_prob_dir, f'ctseg_class{class_index}.nii.gz'),
            )
        save_npz(ctseg_prob_sums.astype(np.float32), os.path.join(ctseg_prob_dir, 'ctseg_prob_atlas.npz'))
        ctseg_hard = compute_hard_atlas_from_probabilities(ctseg_prob_sums)
        save_nifti(ctseg_hard.astype(np.uint16), atlas_ct_nib.affine, os.path.join(output_dir, f'ctseg_atlas_{args.ctseg_num_classes}lbls.nii.gz'))

    if lesion_case_count > 0:
        lesion_prob_sums /= float(lesion_case_count)
        lesion_prob_dir = os.path.join(output_dir, 'prob_atlas', 'lesion')
        os.makedirs(lesion_prob_dir, exist_ok=True)
        for class_index in range(args.lesion_num_classes):
            save_nifti(
                lesion_prob_sums[class_index].astype(np.float32),
                atlas_ct_nib.affine,
                os.path.join(lesion_prob_dir, f'lesion_class{class_index}.nii.gz'),
            )
        save_npz(lesion_prob_sums.astype(np.float32), os.path.join(lesion_prob_dir, 'lesion_prob_atlas.npz'))
        lesion_hard = compute_hard_atlas_from_probabilities(lesion_prob_sums)
        lesion_foreground_prob = lesion_prob_sums[1:, ...].sum(axis=0).astype(np.float32)
        save_nifti(lesion_hard.astype(np.uint16), atlas_ct_nib.affine, os.path.join(output_dir, f'lesion_seg_atlas_{args.lesion_num_classes}lbls.nii.gz'))
        save_nifti(lesion_foreground_prob.astype(np.float32), atlas_ct_nib.affine, os.path.join(output_dir, 'lesion_foreground_prob_atlas.nii.gz'))

    summary_path = os.path.join(output_dir, 'atlas_generation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'model_type={args.model_type}\n')
        f.write(f'experiment_name={experiment_name}\n')
        f.write(f'checkpoint={checkpoint_path}\n')
        f.write(f'atlas_ct={atlas_ct_path}\n')
        f.write(f'data_dir={args.data_dir}\n')
        f.write(f'ctseg_num_classes={args.ctseg_num_classes}\n')
        f.write(f'lesion_num_classes={args.lesion_num_classes}\n')
        f.write('aggregation=mean_warped_one_hot_labels\n')
        f.write('saved_probability_atlases=1\n')
        f.write(f'ctseg_case_count={ctseg_case_count}\n')
        f.write(f'lesion_case_count={lesion_case_count}\n')

    print('Saved outputs to:', output_dir)


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