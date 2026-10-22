"""SITReg + SPR + LDDMM on LUMIR25.

Thin front-end over the shared implementation in ``tutorials/lumir25_TM_LDDMM`` (train_TM_LDDMM.py,
backbones.py) with this tutorial's defaults; every command-line flag of the shared script can
still be overridden, e.g. ``python train_SITReg_SPR_LDDMM.py --reg_weight 0.005 --tag run2``.
Outputs go to ``experiments/`` and ``logs/`` in this folder.
"""
import os
import sys

SHARED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lumir25_TM_LDDMM')
sys.path.insert(0, SHARED_DIR)
from train_TM_LDDMM import main  # noqa: E402

DEFAULTS = {'backbone': 'sitreg', 'param': 'shooting', 'time_steps': 7, 'spr': 'kernel', 'sigmas': [2.0, 4.0, 6.0, 8.0], 'beta_weight': 0.1, 'reg_weight': 0.01, 'init_from': ''}
DEFAULTS['pretrained_dir'] = os.path.join(SHARED_DIR, 'pretrained_wts/')  # share the downloaded LUMIR24 weights

if __name__ == '__main__':
    main(DEFAULTS)
