# Brain Registration Tutorial

This tutorial demonstrates brain image registration using an affine pre-alignment followed by deformable registration. It includes example data and notebooks that walk through the full workflow.

## Contents
- `affine_registration.ipynb`: Affine-only registration walkthrough.
- `deformable_registration.ipynb`: Affine + deformable registration (VFA, TransMorphTVF, ConvexAdam-MIND).
- `sub-01_T1w.nii.gz`: Example moving image.
- `LUMIR_template.nii.gz`: Example template image.

## Prerequisites
- A CUDA-enabled PyTorch install.
- MIR installed (`pip install -e .` from the repository root).
- Internet access for first-time weight downloads (if not already cached).

## Quick start
1. Open `affine_registration.ipynb` and run all cells.
2. Open `deformable_registration.ipynb` and run all cells.

## Inputs
- Moving image: `sub-01_T1w.nii.gz`.
- Fixed/template image: `LUMIR_template.nii.gz`.

## Outputs
- Visualizations of the moving, fixed, and registered images.
- In-memory flow fields for affine and deformable stages.

## Notes
- The deformable notebook composes affine and deformable flows; do not wrap ConvexAdam in `no_grad`.
- If you replace input images, ensure they are reasonably preprocessed (orientation, spacing).
