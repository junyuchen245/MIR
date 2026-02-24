# IXI Benchmarking Tutorial

This tutorial runs **inference benchmarking** (not training) for multiple deformable registration models on IXI. The canonical workflow is the notebook:

- `IXI_benchmarking.ipynb`

It benchmarks:

- `TransMorph`
- `TransMorphTVF`
- `VFA`
- `VFA-SPR`
- `SITReg`
- `SITReg-SPR`
- `ConvexAdam-MIND`
- `deedsBCV` (optional, Linux-only)

## References

- TransMorph:
  - Chen, Junyu, et al. "Transmorph: Transformer for unsupervised medical image registration." Medical Image Analysis 82 (2022): 102615.
- TransMorphTVF:
  - Chen, Junyu, Eric C. Frey, and Yong Du. "Unsupervised learning of diffeomorphic image registration via transmorph." WBIR, 2022.
- SPR:
  - Chen, Junyu, et al. "Unsupervised learning of spatially varying regularization for diffeomorphic image registration." Medical Image Analysis (2025): 103887.
- VFA:
  - Liu, Yihao, et al. "Vector field attention for deformable image registration." Journal of Medical Imaging 11.6 (2024): 064001.
- SITReg:
  - Honkamaa, Joel, and Pekka Marttinen. "SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration." MELBA (2024): 2148-2194.
- ConvexAdam:
  - Siebert, Hanna, et al. "Convexadam: Self-configuring dual-optimization-based 3d multitask medical image registration." IEEE TMI 44.2 (2024): 738-748.

## Environment

From MIR root:

```bash
python3.8 -m pip install -U pip
python3.8 -m pip install -e .
python3.8 -m pip install gdown nibabel scipy matplotlib pandas natsort
```

Recommended:

- CUDA-enabled PyTorch
- Single GPU with enough memory for 3D inference
- Linux if you plan to include `deedsBCV`

## Data and pretrained weights

The notebook automatically:

1. Downloads IXI benchmarking data (`IXI_data.zip`) if missing.
2. Extracts data to `./IXI_data/`.
3. Downloads missing pretrained checkpoints to `./pretrain_wts/`.

Expected paths in this tutorial directory:

- `./IXI_data/Test/*.pkl`
- `./IXI_data/atlas.pkl`
- `./pretrain_wts/*.pth.tar`

## Completed benchmark workflow

1. Open `IXI_benchmarking.ipynb`.
2. Run all cells top-to-bottom once.
3. Confirm the model registry (`models_dict`) contains the models you want to evaluate.
4. Run evaluation loop to generate one CSV per model.
5. Run the final analysis cell to summarize Dice and topology metrics and create boxplots.

## Outputs

Outputs are written to:

- `./results/<ModelName>.csv`

Each CSV contains:

- `study_idx`
- Organ-wise Dice columns
- `ndv` = non-diffeomorphic volume percentage
- `ndp` = non-diffeomorphic points percentage

The final analysis cell:

- Loads all model CSVs from `./results/`
- Aggregates structures
- Prints summary statistics
- Produces comparative boxplots (save manually if desired)

## Reproducibility notes

- Keep fixed image size: `(H, W, D) = (160, 192, 224)`.
- Keep `batch_size=1` for consistent memory use.
- Use the same pretrained checkpoint set across runs.
- If strict reproducibility is required, set random seeds and deterministic flags in PyTorch before model initialization.

## Troubleshooting

- **`Data directory not found and download failed.`**
  - Verify internet and Google Drive access; then rerun the data-download cell.
- **Missing `gdown` / `nibabel` / `scipy`**
  - Install missing packages in the active environment.
- **CUDA out-of-memory**
  - Close other GPU processes; keep `batch_size=1`; run one model at a time.
- **`deedsBCV` unavailable**
  - This backend is Linux-only and requires the executable to be discoverable.
- **Partial reruns after interruption**
  - Existing CSVs are detected and skipped by the evaluation loop.

## Legacy scripts

This directory may also include training scripts (`train_*.py`). They are not required for this completed IXI benchmarking workflow, which is fully handled by `IXI_benchmarking.ipynb`.
