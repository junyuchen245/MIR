#!/usr/bin/env python3.8
"""Pick demo subjects from the IXI *test* split and write PNG slices + label maps + manifest for the web page."""
import numpy as np, json, os, argparse
from PIL import Image
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data_dir', default='data'); ap.add_argument('--out', default='web/samples')
args = ap.parse_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(args.data_dir, 'test.npz')); img, seg, subj, sl = d['img'], d['seg'], d['subj'], d['slice']
SLICE = 96
cand = np.where(sl == SLICE)[0]
# choose 8 subjects spread over the test set, skipping any with unusually low brain area
areas = [(seg[i] > 0).mean() for i in cand]
order = [cand[i] for i in np.argsort(areas)[::-1]]           # largest brain area first
pick = [order[k] for k in np.linspace(0, min(len(order) - 1, 60), 8).astype(int)]
manifest = []
for n, i in enumerate(pick):
    im = img[i] / max(img[i].max(), 1e-6)
    Image.fromarray((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)).save(f'{OUT}/s{n}.png', optimize=True)
    Image.fromarray(seg[i].astype(np.uint8)).save(f'{OUT}/s{n}_seg.png', optimize=True)
    manifest.append({'id': f's{n}', 'subject': int(subj[i]), 'slice': int(sl[i]), 'img': f's{n}.png', 'seg': f's{n}_seg.png'})
    print(n, 'test subject', subj[i], 'labels', len(np.unique(seg[i])))
json.dump({'size': [int(img.shape[1]), int(img.shape[2])], 'source': 'IXI dataset (CC BY-SA 3.0), test split, axial slice %d' % SLICE,
           'samples': manifest}, open(f'{OUT}/manifest.json', 'w'), indent=1)
print('total KB', sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) // 1024)
