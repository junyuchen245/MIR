# LUMIR25 LDDMM Tutorial (TransMorph-LDDMM and model-agnostic wrapper)

Deep registration networks with a **time-varying velocity field** (LDDMM) instead of a
stationary velocity field (SVF). The network predicts the LDDMM parameters and the flow is
integrated inside the model, so the whole pipeline (feature extraction → velocity field →
diffeomorphism → image similarity) is trained end to end. There is no separate "network then
LDDMM refinement" stage.

Model code lives in the package:
- `MIR.models.lddmm` ([src/MIR/models/lddmm.py](../../src/MIR/models/lddmm.py)): `FourierKernel`,
  `LDDMMIntegrator` and the model-agnostic `LDDMMWrapper` that turns **any** dense registration
  backbone (TransMorph, VFA, VoxelMorph, LessNet, Fourier-Net+, SITReg, ...) into an LDDMM model.
- `MIR.models.TransMorphLDDMM` ([src/MIR/models/TransMorphLDDMM.py](../../src/MIR/models/TransMorphLDDMM.py)):
  the dedicated TransMorph variant (shares the encoder/decoder across time steps in `tvf` mode).

## Contents
- `train_TM_LDDMM.py`: training on LUMIR25 (NCC + LDDMM kinetic energy), with validation Dice,
  folding, inverse-consistency and energy-conservation diagnostics. `--backbone` selects the network.
- `backbones.py`: builds each backbone exactly as in its own `lumir25_*` tutorial (with SVF off)
  and wraps it; also handles LUMIR24 pretrained initialization.
- `infer_TM_LDDMM_eval.py`: evaluation on the LUMIR test splits (NDV/NDP, per-structure Dice, TRE);
  produces the same CSV layout as `tutorials/lumir25_TransMorph/infer_TransMorph_eval.py`.
- `experiments/`, `logs/`, `results/`, `pretrained_wts/`: created by the scripts.

## The transformation model

The deformation is the end point of the flow

```
d phi_t / dt = v_t o phi_t,    phi_0 = id,    t in [0, 1]
```

where every velocity `v_t` lies in the Hilbert space `V` induced by a smoothing kernel
`K = L^{-1}` (periodic FFT implementation, `MIR.models.FourierKernel`):

| kernel | operator | parameters |
|---|---|---|
| `helmholtz` (default) | `L = (-alpha * Laplacian + gamma)^power` | `alpha`, `gamma`, `power` (`sqrt(alpha/gamma)` ≈ characteristic length in voxels) |
| `gaussian` | `K` = (sum of) Gaussian(s) | `sigma` (one or several std-devs in voxels) |

Two parameterizations are available (`--param`):

| `--param` | what the network predicts | regularizer | notes |
|---|---|---|---|
| `shooting` (default) | the initial momentum `m_0` (3 channels, one head) | `<m_0, K m_0>` = ‖v_0‖²_V | geodesic shooting. `m_0` is transported with the conservation law `m_t = |Dψ_t⁻¹| (Dψ_t⁻¹)ᵀ m_0∘ψ_t⁻¹`, `v_t = K m_t`. Kinetic energy is constant along the geodesic (monitored as "E drift"). |
| `tvf` | one pre-velocity `p_t` per time step (per-step heads conditioned on the warped moving image, like TransMorph-TVF) | `Σ_t dt <p_t, K p_t>` | relaxation formulation of Beg et al. 2005. |

Forward and inverse maps are integrated jointly with explicit Euler composition
(`--time_steps` steps): `u ← u + dt·v_t∘(id+u)`, `w ← −dt·v_t + w∘(id − dt·v_t)`.
`model((mov, fix), return_dict=True)` returns `flow` (`u`), `inv_flow` (`w`), the
differentiable `energy`, the per-step `energies`, and `momentum` / `pre_velocities`.

With `--int_scale 2` (shooting only) the momentum is upsampled and the flow is integrated on
the full-resolution grid instead of the half-resolution network grid (≈9 GB instead of ≈2.4 GB
for a training step at 160×224×192).

## Spatially varying kernel (region-specific LDDMM, the SPR counterpart)

`--sigmas 1 2 4 8` replaces the stationary kernel by a multi-Gaussian kernel with **per-voxel
mixture weights**, following RDMM (Shen, Vialard, Niethammer, NeurIPS 2019):

```
K_w m = Σ_i √w_i · G_σi * (√w_i m),     w_i(x) ≥ 0,  Σ_i w_i(x) = 1
```

Where the weights favour a large σ the deformation is smooth, where they favour a small σ it can
be sharp, so the regularization strength varies in space exactly as in TransMorph-SPR, but it is
built into the metric: the kinetic energy `<m_0, K_w m_0>` is the regularizer, the weights are
advected with the flow (`w_t = w_0∘ψ_t⁻¹`) and the momentum conservation law is unchanged, so
geodesic shooting still conserves energy ("E drift" stays meaningful). The weights are predicted
end to end:

- `LDDMMWrapper(..., sigmas=[...])` uses a small model-agnostic CNN (`KernelWeightHead`) on
  `cat(mov, fix, field)`; any callable `(mov, fix, field) -> logits` can be passed as
  `weight_head`.
- `TransMorphLDDMM(..., sigmas=[...])` predicts the weights from its decoder features, like the
  SPR head of TransMorph-SPR.

Both start with ~87 % of the weight on the smoothest Gaussian. The output dict gains
`kernel_weights` and the optimal-mass-transport penalty `omt` ∈ [0, 1] (cost of moving weight
away from the largest σ; `--omt_weight`, default 0.1), which plays the role of the Beta prior in
SPR: without it the network would simply pick the sharpest kernel everywhere. The weight maps are
smoothed with a Gaussian of `--weight_smooth_sigma` voxels, and the validation log and TensorBoard
report the mean effective σ (`Σ_i w_i σ_i`) and its map.

```bash
python train_TM_LDDMM.py --backbone transmorph_lddmm --param shooting --sigmas 1 2 4 8 --omt_weight 0.1
python train_TM_LDDMM.py --backbone vfa --param shooting --sigmas 1 2 4 8
```

Peak training-step memory with four Gaussians (shooting, T=7): TransMorph 3.4 GB (half res),
VoxelMorph 21 GB, SITReg 38 GB, VFA 49 GB (full res).

## Integrating TransMorph-SPR / VFA-SPR into LDDMM (`--spr`)

Your SPR produces a single weight map `λ ∈ (0, 1]` (1 = full regularization) with a Beta prior
`logBeta(λ, 1 + w)` pushing it toward 1 and a λ-weighted diffusion penalty on the flow. Two ways
to use that map in the LDDMM model, both end to end:

| `--spr` | what λ does | kernel | prior |
|---|---|---|---|
| `kernel` (recommended) | sets the **kernel scale per voxel**: `λ = 1` → smoothest Gaussian, `λ = 0` → sharpest, in between linear interpolation along `--sigmas` (`SpatiallyVaryingKernel.weights_from_spr`); with two sigmas the weights are simply `[1−λ, λ]` | spatially varying, advected with the flow, energy conserved | `logBeta` (`--beta_weight`) |
| `energy` | weights the **kinetic-energy integrand** `λ(x)·⟨m, Km⟩(x)`, exactly the role λ plays in `LocalGrad3d` | stationary (`--kernel`) | `logBeta` (`--beta_weight`) |

`kernel` mode keeps the metric interpretation (the SPR map *is* the metric); `energy` mode is the
literal transcription of the SPR loss and is cheaper, but the shooting still follows the stationary
kernel, so the weighted energy is a regularizer rather than the geodesic metric.

Where λ comes from (`--spr_source`, default `auto`):

- `head`: a model-agnostic `SPRHead` (small CNN on `cat(mov, fix, field)`, sigmoid, initialized at
  λ = 0.88) for any backbone; `TransMorphLDDMM` instead uses the TransMorph-SPR head
  (`sigma_0..2` convolutions on its decoder features).
- `backbone`: your SPR networks themselves — `--backbone transmorph_spr` (`TransMorphTVFSPR(SVF=False)`)
  or `--backbone vfa_spr` (`VFASPR(SVF=False)`) return `(field, λ)` and the wrapper uses both.

The output dict carries `spr_weight` (λ); the OMT penalty is not used in SPR modes because the
Beta prior takes its place. Validation logs the mean λ (and the effective σ in `kernel` mode) and
TensorBoard shows the map.

```bash
# TransMorph-LDDMM with the TransMorph-SPR head, SPR map -> kernel scale
python train_TM_LDDMM.py --backbone transmorph_lddmm --spr kernel --sigmas 1 2 3 4 --beta_weight 0.1
# your TransMorph-SPR network as the backbone (its own weight map)
python train_TM_LDDMM.py --backbone transmorph_spr --spr kernel --sigmas 1 2 3 4
# VFA-SPR, literal SPR loss on the kinetic energy
python train_TM_LDDMM.py --backbone vfa_spr --spr energy
# any backbone with the model-agnostic SPR head
python train_TM_LDDMM.py --backbone voxelmorph --spr kernel --sigmas 1 2 3 4 --init_from ''
```

In code:

```python
model = LDDMMWrapper(TransMorphTVFSPR(config, SVF=False), spr='kernel', spr_source='backbone', sigmas=[1, 2, 3, 4])
out = model((mov, fix), return_dict=True)
loss = ncc(warp(mov, out['flow']), fix) + 0.01 * out['energy'] + logBeta()(out['spr_weight'], 1.1)
```

## Using the wrapper with any backbone

```python
from MIR.models import VFA, LDDMMWrapper, load_backbone_weights
backbone = VFA(config, device='cuda:0')                   # SVF=False: the raw field is the momentum
model = LDDMMWrapper(backbone, param='shooting', time_steps=7,
                     kernel='helmholtz', alpha=2.0, gamma=1.0, power=2).cuda()
load_backbone_weights(model.backbone, torch.load('VFA_LUMIR24.pth')['model_state_dict'])  # optional
out = model((mov, fix), return_dict=True)                # flow, inv_flow, energy, energies, momentum
loss = ncc(warp(mov, out['flow']), fix) + 0.01 * out['energy']
```

The backbone's output is **not** used as a displacement: it is the initial momentum (`shooting`)
or the per-step pre-velocity (`tvf`), smoothed by the kernel and integrated to a diffeomorphism.
Anything that returns a `[B, 3, D, H, W]` field works; if the backbone returns a tuple (e.g.
`(warped, flow)` with `gen_output=True`) the last element is used. SITReg needs
`adapter='sitreg'` (it returns mapping objects); any other calling convention can be handled
with `adapter=callable(backbone, mov, fix) -> field`. In `tvf` mode the backbone is re-applied
at every time step to the warped moving image with shared weights, which costs `time_steps`
backbone passes.

Backbones available in the tutorial (`--backbone`), with the resolution of their own tutorial
and the measured cost of one training step at 160×224×192 (shooting, T=7, batch 1):

| `--backbone` | network res. | step time | peak memory | notes |
|---|---|---|---|---|
| `transmorph_lddmm` (default) | 1/2 | 0.45 s | 2.4 GB | dedicated class, per-step heads in `tvf` (8.3 GB) |
| `transmorph` | 1/2 | 0.16 s | 2.4 GB | plain `TransMorph(SVF=False)` through the wrapper |
| `vfa` | 1/1 | 1.5 s | 40 GB | `tvf` at full res does not fit in 80 GB; use `--scale_factor 2` |
| `voxelmorph` | 1/1 | 0.95 s | 13 GB | `tvf`: 40 GB |
| `lessnet` | 1/1 | 0.9 s | 10 GB | `tvf`: 20 GB |
| `fouriernet` | 1/1 | 0.9 s | 8 GB | Fourier-Net+ |
| `sitreg` | 1/1 | 4.3 s | 30 GB | `adapter='sitreg'`; no LUMIR24 weights, use `--init_from ''` |

`--scale_factor` overrides the network resolution (the flow is upsampled to full resolution for
the loss, as in the TransMorph tutorial).

## Prerequisites
- MIR installed from the repository root (editable install), CUDA PyTorch.
- LUMIR24 training images and the LUMIR25 validation set at the paths in `train_TM_LDDMM.py`
  (`--train_dir`, `--val_dir`).
- The LUMIR24 TransMorph-TVF weights are downloaded automatically (`--init_from lumir24`) and
  used to initialize the Swin encoder and the shared decoder; the momentum/velocity heads
  start from scratch.

## Quick start

```bash
cd tutorials/lumir25_TM_LDDMM
# geodesic shooting (default)
python train_TM_LDDMM.py --param shooting --time_steps 7 --reg_weight 0.01
# relaxation / time-varying field
python train_TM_LDDMM.py --param tvf --time_steps 7 --reg_weight 0.01
# evaluation on the test splits
python infer_TM_LDDMM_eval.py --ckpt experiments/<run>/dsc0.7xxx.pth.tar --model_tag TM_LDDMM_shooting
```

Other backbones through the wrapper:

```bash
python train_TM_LDDMM.py --backbone vfa --param shooting
python train_TM_LDDMM.py --backbone sitreg --param shooting --init_from ''
python train_TM_LDDMM.py --backbone voxelmorph --param tvf --time_steps 5 --init_from ''
```

Useful flags: `--backbone`, `--scale_factor`, `--kernel gaussian --sigma 2 4`, `--alpha/--gamma/--power`, `--int_scale 2`,
`--grad_weight` (extra diffusion penalty on the final displacement, off by default),
`--iters_per_epoch N --val_subset M --max_epoch 1 --tag calib` for quick calibration runs,
`--cont_training` to resume from `experiments/<run>/`.

The experiment name encodes the configuration, e.g.
`LDDMM_transmorph_lddmm_shooting_T7_helmholtz_a2_g1_p2_int1_ncc_1_reg_0.01/`. The checkpoint
stores the arguments, so `infer_TM_LDDMM_eval.py` rebuilds the right backbone by itself.

## Outputs
- `experiments/<run>/dsc<val Dice>.pth.tar`: checkpoints (`state_dict`, `optimizer`, `args`,
  `epoch`, `best_dsc`). Two most recent are kept.
- `logs/<run>/logfile.log` and TensorBoard scalars: train loss (total / similarity / energy),
  validation Dice, percentage of non-positive Jacobian determinants, inverse-consistency residual
  (`|u + w∘(id+u)|`, voxels) and kinetic-energy drift along the geodesic.
- `results/<model_tag>_<dataset>_{ndv,dsc,tre}.csv` from the evaluation script.

## Calibration (300 iterations, 8 validation pairs, LUMIR24-TVF encoder init)

| run | val Dice | neg. Jacobian | inverse residual | energy term / NCC term |
|---|---|---|---|---|
| identity (no registration) | ≈0.55 | – | – | – |
| `shooting`, `--reg_weight 0.002` | 0.677 | 0.000 % | 0.018 vox | ≈1 % |
| `shooting`, `--reg_weight 0.01` | 0.668 | 0.000 % | 0.012 vox | ≈3 % |
| `tvf`, `--reg_weight 0.01` | 0.680 | 0.000 % | 0.014 vox | ≈4 % |
| `shooting`, `--sigmas 1 2 4 8 --omt_weight 0.1` | 0.621 | 0.000 % | 0.003 vox | effective σ stays at 7.9 vox |
| `shooting`, `--sigmas 1 2 3 4 --omt_weight 0.05` | 0.655 | 0.000 % | 0.007 vox | effective σ stays at 3.9 vox |
| `shooting`, `--spr kernel --sigmas 1 2 3 4 --beta_weight 0.1` | 0.642 | 0.000 % | 0.006 vox | mean λ driven to 0.996 by the Beta prior |

For reference, the TransMorph-TVF baseline in `tutorials/lumir25_TransMorph` reaches 0.725 on
the full validation set after 130 epochs with its diffusion term at ≈10 % of the NCC term.
These numbers only show that training works and give the scale of the energy term; the
regularization weight still has to be tuned with full-length runs.
With the spatially varying kernel the weight maps have barely left their prior after 300
iterations (the OMT term is ≈0.0005), so the kernel is still essentially the smoothest Gaussian
everywhere; σ = 8 voxels at half resolution (FWHM 19 voxels) is clearly too smooth for brains,
`1 2 3 4` is a better starting range there, and whether the weights specialize usefully can only
be judged after full-length training.
In the SPR run the Beta prior at weight 0.1 pushes λ to ≈1 within a few hundred iterations, i.e.
the smoothest kernel everywhere, exactly the behaviour the SPR tutorials counter by tuning
`wt_logBeta` (0.01–0.2) against the data term; `--beta_weight` needs the same tuning here.
Smoke runs with `transmorph_spr` and `vfa_spr` initialized from the LUMIR24 weights reach Dice
0.68 / 0.73 on two validation pairs after two iterations, so the pretrained SPR networks transfer
directly into the LDDMM model.

## Notes
- The regularizer is the LDDMM kinetic energy ‖v‖²_V, not a diffusion penalty on the
  displacement; its scale is different from the TransMorph baseline's `Grad3d` loss, so
  `--reg_weight` has to be retuned (0.01 is the default; see the calibration above).
- "E drift" (kinetic energy at the last step relative to the first) is a sanity check of the
  geodesic integration and is only reported for `--param shooting`; the velocities of the
  `tvf` variant are free to change over time.
- The kernel uses periodic boundary conditions (FFT). LUMIR brains are zero-padded so wrap-around
  has no visible effect.
- With a spatially varying kernel the momentum/pre-velocity is multiplied by `√w_i` before and
  after each Gaussian, so one FFT pair per Gaussian and time step is needed (4 Gaussians ≈ 4× the
  kernel cost, still small next to the backbone).
- Momentum transport samples `m_0` at `ψ_t⁻¹` with trilinear interpolation, which loses a few
  percent of the kinetic energy over the geodesic for rough momenta (the per-step energies are
  returned so this can be monitored). More time steps do not fix it; a smoother kernel does.
- `swin_type` other than `'swin'` is not supported by `TransMorphLDDMM`.
- Backbones whose heads are not zero-initialized (LessNet, SITReg) start with a non-trivial
  momentum; this is harmless but the first iterations have a large energy term.
- Reference LDDMM implementations consulted: Czolbe's
  [pyLDDMM](https://github.com/SteffenCzolbe/pyLDDMM) (Beg et al. relaxation, 2D NumPy),
  [lddmm-toolbox](https://github.com/TharindaDilshan/lddmm-toolbox) and
  [PatRyg99/lddmm](https://github.com/PatRyg99/lddmm) (PyTorch/KeOps geodesic shooting for
  surfaces and landmarks). The dense-image formulation here follows Beg 2005, Vialard 2012
  (momentum conservation) and FLASH (Fourier kernel).
