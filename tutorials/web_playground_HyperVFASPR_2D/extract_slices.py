"""Extract 2D axial slices from the preprocessed IXI volumes (160x192x224 pkl) for the browser demo.
IXI (CC BY-SA 3.0) preprocessed as in TransMorph: https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md
Axis 1 is the axial axis; we crop axis 2 to [16, 208) and transpose so slices are (192, 160) with
anterior at the top. Output: data/<split>.npz with img [N,192,160] float32 in [0,1], seg [N,192,160] uint8,
subject index and slice index arrays.
"""
import pickle, numpy as np, glob, os, sys, argparse
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--ixi_dir', default='IXI', help='folder with Train/ Val/ Test/ subject_*.pkl (the preprocessed IXI set released with TransMorph)')
ap.add_argument('--out', default='data', help='output folder for train/val/test.npz')
args = ap.parse_args()
root = os.path.join(args.ixi_dir, '')
SLICES = [84, 90, 96, 102, 108]
os.makedirs(args.out, exist_ok=True)
for split in ['Train', 'Val', 'Test']:
    files = sorted(glob.glob(root + split + '/*.pkl'), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    imgs, segs, subj, sidx = [], [], [], []
    for i, f in enumerate(files):
        img, seg = pickle.load(open(f, 'rb'))
        for k in SLICES:
            sl = img[:, k, 16:208].T[::-1].copy()          # (192,160), flip so anterior is up
            sg = seg[:, k, 16:208].T[::-1].copy()
            imgs.append(sl.astype(np.float32)); segs.append(sg.astype(np.uint8)); subj.append(i); sidx.append(k)
    imgs = np.stack(imgs); segs = np.stack(segs)
    print(split, imgs.shape, 'intensity max %.3f mean %.3f' % (imgs.max(), imgs.mean()), 'labels', len(np.unique(segs)))
    np.savez_compressed(os.path.join(args.out, f'{split.lower()}.npz'), img=imgs, seg=segs, subj=np.array(subj), slice=np.array(sidx))
# preview montage of a few test slices
from PIL import Image
d = np.load(os.path.join(args.out, 'test.npz')); sel = d['img'][2::5][:6]
sheet = np.concatenate([(np.clip(s / 0.7, 0, 1) * 255).astype(np.uint8) for s in sel], axis=1)
Image.fromarray(sheet).save(os.path.join(args.out, 'preview_test.png')); print('preview', sheet.shape)
