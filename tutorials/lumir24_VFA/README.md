# LUMIR24 VFA Tutorial

Inference workflows for VFA on the LUMIR24 dataset.

## Contents
- `infer_VFA.py`: VFA inference using LUMIR24 weights.
- `infer_VFA_w_lumir25_wts.py`: VFA inference using LUMIR25 weights on LUMIR24.

## Prerequisites
- CUDA-enabled PyTorch.
- MIR installed from the repository root.
- LUMIR24 input data available at the paths configured in the scripts.
- Pretrained VFA weights downloaded and placed in the expected location.

## Quick start
1. Open the script you want to run and set dataset/output paths.
2. Run the script from this folder.

## Inputs
- LUMIR24 moving/fixed images as configured in the script.

## Outputs
- Registered images and flow fields saved to the output directory configured in the script.

## Notes
- `infer_VFA_w_lumir25_wts.py` is for cross-dataset testing with LUMIR25 weights.
