# LUMIR24 ConvexAdam Tutorial

Inference workflows for ConvexAdam-based registration on the LUMIR24 dataset, including an optional VFA refinement stage.

## Contents
- `infer_ConvexAdam.py`: ConvexAdam-only inference.
- `infer_ConvexAdamVFA.py`: ConvexAdam followed by VFA refinement.

## Prerequisites
- CUDA-enabled PyTorch.
- MIR installed from the repository root.
- LUMIR24 input data available at the paths configured in the scripts.
- Pretrained weights downloaded (the scripts reference the expected filenames).

## Quick start
1. Open the script you want to run and set dataset and output paths.
2. Run the script from this folder.

## Inputs
- LUMIR24 moving/fixed images as configured in the script.

## Outputs
- Registered images and flow fields saved to the output directory configured in the script.

## Notes
- ConvexAdam requires gradients; do not wrap its call in `no_grad`.
- If you use the VFA refinement, ensure the VFA weights are present.
