# LUMIR25 VFA Tutorial

Inference and training utilities for VFA variants on the LUMIR25 dataset, including optional SynthSR preprocessing and a Dockerized example.

## Contents
- `infer_VFA.py`: VFA inference.
- `infer_VFA_w_SynthSR.py`: VFA inference with SynthSR preprocessing.
- `infer_HyperVFA_SPR.py`: HyperVFA-SPR inference.
- `infer_SynthSR.sh`: SynthSR preprocessing helper.
- `train_VFA_encoder.py`: VFA encoder training.
- `LUMIR25_Docker_Example/`: Dockerized example setup.

## Prerequisites
- CUDA-enabled PyTorch.
- MIR installed from the repository root (unless using the Docker example).
- LUMIR25 input data available at the paths configured in the scripts.
- Pretrained weights for VFA/HyperVFA as referenced by the scripts.
- SynthSR dependencies if you run the SynthSR pipeline.

## Quick start
1. Choose the script that matches your workflow (standard VFA, HyperVFA-SPR, or SynthSR-enabled).
2. Update dataset/output paths inside the script.
3. Run the script from this folder.

## Inputs
- LUMIR25 moving/fixed images as configured in the scripts.

## Outputs
- Registered images and flow fields saved to the output directory configured in the script.

## Notes
- The Docker example provides a self-contained environment for inference.
- SynthSR workflows require additional preprocessing time and dependencies.
