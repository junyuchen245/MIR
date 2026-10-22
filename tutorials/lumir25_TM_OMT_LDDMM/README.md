# LUMIR25 TransMorph + region-specific kernel (OMT) + LDDMM

TransMorph-LDDMM with the RDMM-style spatially varying kernel (Shen, Vialard, Niethammer,
NeurIPS 2019): a multi-Gaussian kernel whose per-voxel mixture weights are predicted by a head on
the decoder features (softmax over `--sigmas`), advected with the flow, and regularized by the
optimal-mass-transport penalty `omt` (`--omt_weight`) that charges for moving weight away from the
largest σ. Unlike the SPR folder there is no single λ map and no Beta prior: the weights are free
mixtures over the kernel scales. Encoder/decoder are initialized from the LUMIR24 TransMorph-TVF
weights.

The model code lives in the MIR package (`MIR.models.lddmm`, `MIR.models.TransMorphLDDMM`) and the
training / evaluation logic in [`tutorials/lumir25_TM_LDDMM`](../lumir25_TM_LDDMM) (see its README
for the mathematics, all flags, and the calibration table). This folder only fixes the defaults:

```
--backbone transmorph_lddmm --param shooting --time_steps 7 --sigmas 1.0 2.0 3.0 4.0 --omt_weight 0.05 --reg_weight 0.01 --init_from lumir24
```

## Contents
- `train_TM_OMT_LDDMM.py`: training with the defaults above (any flag of the shared script can be overridden).
- `infer_TM_OMT_LDDMM_eval.py`: evaluation on the LUMIR test splits (NDV/NDP, Dice, TRE CSVs, tag `TM_OMT_LDDMM`).
- `experiments/`, `logs/`, `results/`: created by the scripts.

## Quick start
```bash
cd tutorials/lumir25_TM_OMT_LDDMM
python train_TM_OMT_LDDMM.py                      # full training
python train_TM_OMT_LDDMM.py --iters_per_epoch 300 --val_subset 8 --max_epoch 1 --tag calib   # quick calibration
python infer_TM_OMT_LDDMM_eval.py --ckpt experiments/<run>/dsc0.7xxx.pth.tar
```

## Calibration
300 iterations, 8 validation pairs, LUMIR24-TVF encoder init:

| val Dice | neg. Jacobian | inverse residual | energy drift | effective σ |
|---|---|---|---|---|
| 0.651 | 0.000 % | 0.007 vox | +1.4 % | 3.93 vox (of σ ∈ {1, 2, 3, 4}) |

Identity is ≈0.55; the stationary-kernel TransMorph-LDDMM reaches 0.668 at the same budget. The
weight head starts at ≈87 % on the largest σ and barely moves in 300 iterations (OMT term ≈0.0005),
so the kernel is still almost uniform; `--omt_weight` and the σ range need tuning with full runs.
`--sigmas 1 2 4 8` (FWHM 19 voxels at half resolution) was too smooth in an earlier calibration
(Dice 0.621).
