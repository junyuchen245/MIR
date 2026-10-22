# LUMIR25 TransMorph + SPR + LDDMM

TransMorph-LDDMM with the TransMorph-SPR head: the network predicts the initial momentum
and a spatially varying regularization map λ ∈ (0, 1]; λ sets the LDDMM kernel scale per voxel
(λ = 1 → smoothest Gaussian, λ = 0 → sharpest, `--spr kernel`), the kinetic energy under that
kernel is the regularizer and the Beta prior `logBeta(λ, 1 + beta_weight)` keeps λ near 1 unless
the data demand otherwise. Encoder/decoder are initialized from the LUMIR24 TransMorph-TVF weights.

Alternative: `--backbone transmorph_spr` uses your actual TransMorph-SPR network
(`TransMorphTVFSPR(SVF=False)`) as the momentum/λ predictor through the model-agnostic wrapper;
`--spr energy` weights the kinetic-energy integrand by λ instead (literal SPR loss, stationary kernel).

The model code lives in the MIR package (`MIR.models.lddmm`, `MIR.models.TransMorphLDDMM`) and the
training / evaluation logic in [`tutorials/lumir25_TM_LDDMM`](../lumir25_TM_LDDMM) (see its README
for the mathematics, all flags, and the calibration table). This folder only fixes the defaults:

```
--backbone transmorph_lddmm --param shooting --time_steps 7 --spr kernel --sigmas 1.0 2.0 3.0 4.0 --beta_weight 0.1 --reg_weight 0.01 --init_from lumir24
```

## Contents
- `train_TM_SPR_LDDMM.py`: training with the defaults above (any flag of the shared script can be overridden).
- `infer_TM_SPR_LDDMM_eval.py`: evaluation on the LUMIR test splits (NDV/NDP, Dice, TRE CSVs, tag `TM_SPR_LDDMM`).
- `experiments/`, `logs/`, `results/`: created by the scripts.

## Quick start
```bash
cd tutorials/lumir25_TM_SPR_LDDMM
python train_TM_SPR_LDDMM.py                      # full training
python train_TM_SPR_LDDMM.py --iters_per_epoch 300 --val_subset 8 --max_epoch 1 --tag calib   # quick calibration
python infer_TM_SPR_LDDMM_eval.py --ckpt experiments/<run>/dsc0.7xxx.pth.tar
```

## Calibration
300 iterations, 8 validation pairs, LUMIR24-TVF encoder init (`--iters_per_epoch 300 --val_subset 8 --max_epoch 1`):

| val Dice | neg. Jacobian | inverse residual | energy drift | mean λ / effective σ |
|---|---|---|---|---|
| 0.645 | 0.000 % | 0.006 vox | +1.5 % | λ = 0.997, σ = 3.99 vox |

Identity is ≈0.55; the stationary-kernel TransMorph-LDDMM reaches 0.668 and the fully trained
TransMorph-TVF baseline 0.725. The Beta prior at weight 0.1 drives λ to ≈1 (the smoothest kernel
everywhere) within a few hundred iterations, so `--beta_weight` must be tuned against the data term
exactly as `wt_logBeta` is in the SPR tutorials (0.01–0.2); whether λ then becomes spatially
informative can only be judged after full-length training.
