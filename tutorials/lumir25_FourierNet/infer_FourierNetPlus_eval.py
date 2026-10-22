import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import glob
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from natsort import natsorted
from MIR.accuracy_measures import calc_jac_dets, get_identity_grid, calc_measurements
from MIR.models import SpatialTransformer, FourierNetPlus
import MIR.models.configs_FourierNet as configs_FourierNet
import nibabel as nib

# --------------------------------------------------------------------------- #
# Brain structure label lookup (133 entries including background)
# --------------------------------------------------------------------------- #
label_lookup = [
    0.,   4.,  11.,  23.,  30.,  31.,  32.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,
   44.,  45.,  47.,  48.,  49.,  50.,  51.,  52.,  55.,  56.,  57.,  58.,  59.,  60.,
   61.,  62.,  71.,  72.,  73.,  75.,  76., 100., 101., 102., 103., 104., 105., 106.,
  107., 108., 109., 112., 113., 114., 115., 116., 117., 118., 119., 120., 121., 122.,
  123., 124., 125., 128., 129., 132., 133., 134., 135., 136., 137., 138., 139., 140.,
  141., 142., 143., 144., 145., 146., 147., 148., 149., 150., 151., 152., 153., 154.,
  155., 156., 157., 160., 161., 162., 163., 164., 165., 166., 167., 168., 169., 170.,
  171., 172., 173., 174., 175., 176., 177., 178., 179., 180., 181., 182., 183., 184.,
  185., 186., 187., 190., 191., 192., 193., 194., 195., 196., 197., 198., 199., 200.,
  201., 202., 203., 204., 205., 206., 207.,
]
NUM_CLUS = len(label_lookup)  # 133

BRAIN_LABEL_NAMES_PATH = '/scratch/jchen/python_projects/LUMIR25/BrainStructureNames.txt'
with open(BRAIN_LABEL_NAMES_PATH) as _f:
    brain_label_names = [line.rstrip().split(' ')[1] for line in _f]

lm_names = [
    'anterior-commissure-AC', 'posterior-commissure-PC', 'infracollicular-sulcus',
    'pontomesencephalic-junction', 'superior-interpeduncular-fossa',
    'Right-superior-lateral-mesencephalic-sulcus', 'Left-superior-lateral-mesencephalic-sulcus',
    'Right-inferior-lateral-mesencephalic-sulcus', 'Left-inferior-lateral-mesencephalic-sulcus',
    'culmen', 'intermammillary-sulcus', 'Right-mammillary-body', 'Left-mammillary-body',
    'pineal-gland', 'Right-lateral-aspect-of-frontal-horn-at-AC',
    'Left-lateral-aspect-of-frontal-horn-at-AC', 'Right-lateral-aspect-of-frontal-horn-at-PC',
    'Left-lateral-aspect-of-frontal-horn-at-PC', 'genu-of-corpus-callosum',
    'splenium-of-the-corpus-callosum', 'Right-anterolateral-temporal-horn',
    'Left-anterolateral-temporal-horn', 'Right-superior-anteromedial-temporal-horn',
    'Left-superior-anteromedial-temporal-horn', 'Right-inferior-anteromedial-temporal-horn',
    'Left-inferior-anteromedial-temporal-horn', 'Right-indusium-griseum-origin',
    'Left-indusium-griseum-origin', 'Right-ventral-occipital-horn',
    'Left-ventral-occipital-horn', 'Right-olfactory-sulcal-fundus',
    'Left-olfactory-sulcal-fundus',
]

# --------------------------------------------------------------------------- #
# Dataset configurations
# --------------------------------------------------------------------------- #
DATASET_CONFIGS = [
    {
        'name': 'LUMIR_L2R24',
        'json': '/scratch/jchen/python_projects/LUMIR25/LUMIR_dataset_test.json',
        'base_dir': '/scratch2/jchen/DATA/LUMIR/',
        'label_dir': '/scratch2/jchen/DATA/LUMIR/labelsTs/',
        'lm_dir': '/scratch2/jchen/DATA/LUMIR/landmarksTs/',
    },
    {
        'name': 'LUMIR25_MM',
        'json': '/scratch2/jchen/DATA/LUMIR_multimodal/LUMIR25MultiModal_dataset.json',
        'base_dir': '/scratch2/jchen/DATA/LUMIR_multimodal/',
        'label_dir': '/scratch2/jchen/DATA/LUMIR/zero_shot/NIMH/lbl/',
        'lm_dir': None,
    },
    {
        'name': 'NIMH_t1w',
        'json': '/scratch2/jchen/DATA/LUMIR/zero_shot/LUMIR_zeroshot_intersubject_NIMH_t1w.json',
        'base_dir': '/scratch2/jchen/DATA/LUMIR/zero_shot/',
        'label_dir': '/scratch2/jchen/DATA/LUMIR/zero_shot/NIMH/lbl/',
        'lm_dir': None,
    },
    {
        'name': 'ADNI1_1.5T',
        'json': '/scratch2/jchen/DATA/LUMIR/zero_shot/LUMIR_zeroshot_intersubject_ADNI1_1.5T.json',
        'base_dir': '/scratch2/jchen/DATA/LUMIR/zero_shot/',
        'label_dir': '/scratch2/jchen/DATA/LUMIR/zero_shot/ADNI1_1.5T/lbl/',
        'lm_dir': None,
    },
    {
        'name': 'ADNI1_3T',
        'json': '/scratch2/jchen/DATA/LUMIR/zero_shot/LUMIR_zeroshot_intersubject_ADNI1_3T.json',
        'base_dir': '/scratch2/jchen/DATA/LUMIR/zero_shot/',
        'label_dir': '/scratch2/jchen/DATA/LUMIR/zero_shot/ADNI1_3T/lbl/',
        'lm_dir': None,
    },
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def csv_writter(line, name):
    with open(name + '.csv', 'a') as file:
        file.write(line)
        file.write('\n')


def load_best_checkpoint(model, exp_dir):
    checkpoints = natsorted(glob.glob(exp_dir + '*.pth.tar'))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found in {exp_dir}')
    best = checkpoints[-1]
    model.load_state_dict(torch.load(best, weights_only=False)['state_dict'])
    print(f'  Loaded: {best}')
    return model


def label_reassign_gpu(label_arr):
    ref = torch.from_numpy(label_arr).cuda()
    out = torch.zeros_like(ref)
    for i, v in enumerate(label_lookup):
        out[ref == v] = i
    return out


def dice_substruct_cpu(def_lbl, y_lbl, pair_id, num_classes):
    """Per-structure DSC without one_hot — avoids GPU OOM on large class counts.

    Returns (csv_line, mean_dsc).
    """
    pred = def_lbl.cpu().squeeze().long()
    true = y_lbl.cpu().squeeze().long()
    line = str(pair_id)
    total = 0.0
    for c in range(num_classes):
        pred_c = (pred == c)
        true_c = (true == c)
        inter = (pred_c & true_c).sum().item()
        denom = pred_c.sum().item() + true_c.sum().item()
        dsc = (2.0 * inter) / denom if denom > 0 else 1.0
        line += f',{dsc}'
        total += dsc
    return line, total / num_classes


def calc_TRE(x_lm_t, def_lm_t, y_lm_t, sub_idx):
    x_lm   = x_lm_t[0, 0]
    def_lm = def_lm_t[0, 0]
    y_lm   = y_lm_t[0, 0]
    zz = torch.linspace(0, x_lm.shape[0] - 1, x_lm.shape[0]).cuda()
    yy = torch.linspace(0, x_lm.shape[1] - 1, x_lm.shape[1]).cuda()
    xx = torch.linspace(0, x_lm.shape[2] - 1, x_lm.shape[2]).cuda()
    zv, yv, xv = torch.meshgrid(zz, yy, xx, indexing='ij')
    unique = torch.unique(x_lm).cuda()
    dfm_pos = torch.zeros((len(unique) - 1, 3))
    fx_pos  = torch.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        lbl_d = (def_lm == unique[i]).float()
        lbl_f = (y_lm   == unique[i]).float()
        dfm_pos[i - 1] = torch.stack([
            torch.sum(lbl_d * xv) / (lbl_d.sum() + 1e-8),
            torch.sum(lbl_d * yv) / (lbl_d.sum() + 1e-8),
            torch.sum(lbl_d * zv) / (lbl_d.sum() + 1e-8),
        ])
        fx_pos[i - 1] = torch.stack([
            torch.sum(lbl_f * xv) / (lbl_f.sum() + 1e-8),
            torch.sum(lbl_f * yv) / (lbl_f.sum() + 1e-8),
            torch.sum(lbl_f * zv) / (lbl_f.sum() + 1e-8),
        ])
    errors = torch.sqrt(torch.sum((dfm_pos - fx_pos) ** 2, dim=1))
    line = str(sub_idx)
    for err in errors:
        line += ',' + str(err.item())
    return line


def get_label_path(img_json_path, label_dir):
    basename = os.path.basename(img_json_path)
    for old in ['_0001.nii', '_0002.nii', '_0003.nii']:
        basename = basename.replace(old, '_0000.nii')
    return os.path.join(label_dir, basename)


class L2RLUMIRJSONDataset(Dataset):
    def __init__(self, base_dir, json_path):
        with open(json_path) as f:
            d = json.load(f)
        self.imgs = d.get('validation', d.get('test', []))
        self.base_dir = base_dir

    def __getitem__(self, index):
        img_dict = self.imgs[index]
        x = nib.load(self.base_dir + img_dict['moving']).get_fdata() / 255.
        y = nib.load(self.base_dir + img_dict['fixed']).get_fdata() / 255.
        x = torch.from_numpy(np.ascontiguousarray(x[None])).float()
        y = torch.from_numpy(np.ascontiguousarray(y[None])).float()
        return x, y

    def __len__(self):
        return len(self.imgs)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    results_dir = 'results/'
    os.makedirs(results_dir, exist_ok=True)

    H, W, D = 160, 224, 192
    config = configs_FourierNet.get_FourierNetPlus_default_config()
    config.img_size = (H, W, D)
    config.low_img_size = (H // 2, W // 2, D // 2)

    spatial_trans_nn = SpatialTransformer(size=(H, W, D), mode='nearest').cuda()

    wts_mono_dir  = 'experiments/FourierNetPlus_wo_aug_LUMIR25_ncc_1.0_diffusion_5.0/'
    wts_multi_dir = 'experiments/FourierNetPlus_w_aug_LUMIR25_ncc_1.0_diffusion_5.0/'

    for model_tag, wts_dir in [('FourierNetPlus_mono', wts_mono_dir), ('FourierNetPlus_multi', wts_multi_dir)]:
        print(f'\nLoading {model_tag} ...')
        model = FourierNetPlus(config).cuda()
        model = load_best_checkpoint(model, wts_dir)
        model.eval()

        for cfg in DATASET_CONFIGS:
            ds_name  = cfg['name']
            csv_base = results_dir + f'{model_tag}_{ds_name}'

            print(f'\n====== {ds_name} ({model_tag}) ======')

            ndv_csv = csv_base + '_ndv'
            dsc_csv = (csv_base + '_dsc') if cfg['label_dir'] else None
            tre_csv = (csv_base + '_tre') if cfg['lm_dir'] else None
            expected = [p for p in [ndv_csv, dsc_csv, tre_csv] if p]

            if all(os.path.exists(p + '.csv') for p in expected):
                print('  Skipping — all results already exist.')
                continue

            for p in expected:
                if os.path.exists(p + '.csv'):
                    os.remove(p + '.csv')

            csv_writter('pat_idx,ndv,ndp', ndv_csv)
            if dsc_csv:
                csv_writter('pat_idx,' + ','.join(brain_label_names), dsc_csv)
            if tre_csv:
                csv_writter('pat_idx,' + ','.join(lm_names), tre_csv)

            with open(cfg['json']) as f:
                d = json.load(f)
            imgs = d.get('validation', d.get('test', []))
            dataset = L2RLUMIRJSONDataset(cfg['base_dir'], cfg['json'])
            loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

            for i, (x, y) in enumerate(loader):
                mov_path = imgs[i]['moving']
                fix_path = imgs[i]['fixed']
                mv_id = os.path.basename(mov_path).split('_')[-2]
                fx_id = os.path.basename(fix_path).split('_')[-2]
                pair_id = f'fxid_{fx_id}_mvid_{mv_id}'
                print(f'Processing {pair_id} ... ', end='', flush=True)

                with torch.no_grad():
                    x = x.cuda()
                    y = y.cuda()
                    flow = model((x, y))

                    # --- NDV / NDP ---
                    mask = x.detach().cpu().numpy()[0, 0, 1:-1, 1:-1, 1:-1] > 0
                    disp = flow.cpu().detach().numpy()[0]
                    trans_ = disp + get_identity_grid(disp)
                    jac_dets = calc_jac_dets(trans_)
                    non_diff_v, _, non_diff_vol, _ = calc_measurements(jac_dets, mask)
                    total_v = np.sum(mask)
                    ndv = non_diff_vol / total_v * 100
                    ndp = non_diff_v   / total_v * 100
                    csv_writter(f'{pair_id},{ndv},{ndp}', ndv_csv)

                    # --- DSC ---
                    mean_dsc = float('nan')
                    if dsc_csv:
                        x_lbl_path = get_label_path(mov_path, cfg['label_dir'])
                        y_lbl_path = get_label_path(fix_path, cfg['label_dir'])
                        if os.path.exists(x_lbl_path) and os.path.exists(y_lbl_path):
                            x_lbl_t = label_reassign_gpu(nib.load(x_lbl_path).get_fdata())[None, None].float()
                            y_lbl_t = label_reassign_gpu(nib.load(y_lbl_path).get_fdata())[None, None]
                            def_lbl = spatial_trans_nn(x_lbl_t, flow).long()
                            dsc_line, mean_dsc = dice_substruct_cpu(def_lbl, y_lbl_t, pair_id, NUM_CLUS)
                            csv_writter(dsc_line, dsc_csv)

                    # --- TRE ---
                    if tre_csv:
                        x_lm_path = os.path.join(cfg['lm_dir'], os.path.basename(mov_path))
                        y_lm_path = os.path.join(cfg['lm_dir'], os.path.basename(fix_path))
                        if os.path.exists(x_lm_path) and os.path.exists(y_lm_path):
                            x_lm = torch.from_numpy(nib.load(x_lm_path).get_fdata()[None, None]).cuda()
                            y_lm = torch.from_numpy(nib.load(y_lm_path).get_fdata()[None, None]).cuda()
                            def_lm = spatial_trans_nn(x_lm.float(), flow)
                            tre_line = calc_TRE(x_lm, def_lm, y_lm, pair_id)
                            csv_writter(tre_line, tre_csv)

                msg = f'[{ds_name}] {i+1}/{len(imgs)} {pair_id} | NDV={ndv:.2f}% NDP={ndp:.2f}%'
                if not np.isnan(mean_dsc):
                    msg += f' | DSC={mean_dsc:.4f}'
                print(msg)

            print(f'Saved: {ndv_csv}.csv'
                  + (f', {dsc_csv}.csv' if dsc_csv else '')
                  + (f', {tre_csv}.csv' if tre_csv else ''))

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.manual_seed(0)
    print('Currently using:', torch.cuda.get_device_name(0))
    main()
