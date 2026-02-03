# LUMIR25 ConvexAdam Tutorial

Inference workflow for ConvexAdam on the LUMIR25 dataset.

## Contents
- `infer_ConvexAdam.py`: ConvexAdam inference script.

## Prerequisites
- CUDA-enabled PyTorch.
- MIR installed from the repository root.
- LUMIR25 input data available at the paths configured in the script.

## Quick start
1. Open `infer_ConvexAdam.py` and set dataset/output paths.
2. Run the script from this folder.

## Inputs
- LUMIR25 moving/fixed images as configured in the script.

## Outputs
- Registered images and flow fields saved to the output directory configured in the script.

## Notes
- ConvexAdam requires gradients; do not wrap its call in `no_grad`.
