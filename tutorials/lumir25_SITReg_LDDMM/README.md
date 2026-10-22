# LUMIR25 SITReg + LDDMM

SITReg (built exactly as in `tutorials/lumir25_SITReg`, full resolution, no LUMIR24 pretrained
weights) used as the initial-momentum predictor through the model-agnostic `LDDMMWrapper`
(`adapter='sitreg'`): its level-0 forward displacement is read as the momentum, smoothed by the
Helmholtz kernel and integrated by geodesic shooting. Because SITReg runs at full resolution the
kernel is given `--alpha 8` (= 2 × 2², the same physical length scale as `alpha 2` on the
half-resolution TransMorph grid, FWHM ≈ 6 full-res voxels).

Note that SITReg's own symmetric/inverse-consistent construction is replaced by the LDDMM
inverse (`inv_flow`), and one training step needs ≈30 GB and ≈4.5 s at 160×224×192.

The model code lives in the MIR package (`MIR.models.lddmm`, `MIR.models.TransMorphLDDMM`) and the
training / evaluation logic in [`tutorials/lumir25_TM_LDDMM`](../lumir25_TM_LDDMM) (see its README
for the mathematics, all flags, and the calibration table). This folder only fixes the defaults:

```
--backbone sitreg --param shooting --time_steps 7 --alpha 8.0 --reg_weight 0.01 --init_from ''
```

## Contents
- `train_SITReg_LDDMM.py`: training with the defaults above (any flag of the shared script can be overridden).
- `infer_SITReg_LDDMM_eval.py`: evaluation on the LUMIR test splits (NDV/NDP, Dice, TRE CSVs, tag `SITReg_LDDMM`).
- `experiments/`, `logs/`, `results/`: created by the scripts.

## Quick start
```bash
cd tutorials/lumir25_SITReg_LDDMM
python train_SITReg_LDDMM.py                      # full training
python train_SITReg_LDDMM.py --iters_per_epoch 100 --val_subset 8 --max_epoch 1 --tag calib   # quick calibration
python infer_SITReg_LDDMM_eval.py --ckpt experiments/<run>/dsc0.7xxx.pth.tar
```

## Calibration
100 iterations, 8 validation pairs, random init (no LUMIR24 weights for SITReg), ≈4 s per step:

| val Dice | neg. Jacobian | inverse residual | energy drift |
|---|---|---|---|
| 0.570 | 0.000 % | 0.001 vox | −0.4 % |

Identity is ≈0.55 and the displacements are still below one voxel after 100 iterations, so this
only shows that the SITReg → momentum → shooting pipeline trains stably; SITReg needs its usual
full-length schedule (and ≈30 GB per step at full resolution).
