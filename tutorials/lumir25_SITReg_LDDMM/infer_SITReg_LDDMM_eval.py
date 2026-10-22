"""Evaluate a SITReg + LDDMM checkpoint on the LUMIR test splits.

Thin front-end over ``tutorials/lumir25_TM_LDDMM/infer_TM_LDDMM_eval.py``; results go to
``results/`` in this folder. Example::

    python infer_SITReg_LDDMM_eval.py --ckpt experiments/<run>/dsc0.7xxx.pth.tar
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lumir25_TM_LDDMM'))
from infer_TM_LDDMM_eval import main  # noqa: E402

if __name__ == '__main__':
    main({'model_tag': 'SITReg_LDDMM'})
