# LUMIR25 SITReg + SPR + LDDMM

SITReg as the momentum predictor (as in `lumir25_SITReg_LDDMM`) plus spatially varying
regularization: SITReg has no SPR head in MIR, so the model-agnostic `SPRHead` (a small CNN on
`cat(mov, fix, momentum)`, sigmoid output λ ∈ (0, 1]) predicts the regularization map, which sets
the kernel scale per voxel (`--spr kernel`, `--sigmas 2 4 6 8` on the full-resolution grid, i.e.
the same physical scales as `1 2 3 4` at half resolution) under the Beta prior
`logBeta(λ, 1 + beta_weight)`.

`--spr energy` is the cheaper alternative that weights the kinetic-energy integrand by λ with a
stationary kernel (`--alpha 8`).

The model code lives in the MIR package (`MIR.models.lddmm`, `MIR.models.TransMorphLDDMM`) and the
training / evaluation logic in [`tutorials/lumir25_TM_LDDMM`](../lumir25_TM_LDDMM) (see its README
for the mathematics, all flags, and the calibration table). This folder only fixes the defaults:

```
--backbone sitreg --param shooting --time_steps 7 --spr kernel --sigmas 2.0 4.0 6.0 8.0 --beta_weight 0.1 --reg_weight 0.01 --init_from ''
```

## Contents
- `train_SITReg_SPR_LDDMM.py`: training with the defaults above (any flag of the shared script can be overridden).
- `infer_SITReg_SPR_LDDMM_eval.py`: evaluation on the LUMIR test splits (NDV/NDP, Dice, TRE CSVs, tag `SITReg_SPR_LDDMM`).
- `experiments/`, `logs/`, `results/`: created by the scripts.

## Quick start
```bash
cd tutorials/lumir25_SITReg_SPR_LDDMM
python train_SITReg_SPR_LDDMM.py                      # full training
python train_SITReg_SPR_LDDMM.py --iters_per_epoch 100 --val_subset 8 --max_epoch 1 --tag calib   # quick calibration
python infer_SITReg_SPR_LDDMM_eval.py --ckpt experiments/<run>/dsc0.7xxx.pth.tar
```

## Calibration
100 iterations, 8 validation pairs, random init, ≈4.3 s per step:

| val Dice | neg. Jacobian | inverse residual | energy drift | mean λ / effective σ |
|---|---|---|---|---|
| 0.565 | 0.000 % | 0.000 vox | −0.1 % | λ = 0.986, σ = 7.92 vox |

As in the plain SITReg-LDDMM folder the network has barely started to deform after 100 iterations;
the Beta prior already pushes λ toward 1, so `--beta_weight` needs the same tuning as in the
TransMorph SPR folder.
