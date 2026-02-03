# LUMIR24 TransMorph Tutorial

Inference workflow for TransMorph on the LUMIR24 dataset.

## Contents
- `infer_TransMorph.py`: TransMorph inference script.

## Prerequisites
- CUDA-enabled PyTorch.
- MIR installed from the repository root.
- LUMIR24 input data available at the paths configured in the script.
- Pretrained weights downloaded (the script references the expected filename).

## Quick start
1. Open `infer_TransMorph.py` and set dataset/output paths.
2. Run the script from this folder.

## Inputs
- LUMIR24 moving/fixed images as configured in the script.

## Outputs
- Registered images and flow fields saved to the output directory configured in the script.

## Notes
- The model expects voxel spacing and orientation that match the training configuration.
