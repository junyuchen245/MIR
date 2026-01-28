# SITReg Tutorial (IXI)

This tutorial trains and evaluates the SITReg model on the IXI dataset using the script `train_SITReg.py`.

## What this tutorial does
- Builds the SITReg model and feature extractor.
- Trains with NCC similarity and diffusion regularization.
- Runs validation using Dice on VOI labels.
- Saves checkpoints and TensorBoard logs.

## Prerequisites
- Python environment with MIR installed and its dependencies available.
- GPU recommended.
- IXI dataset prepared as `.pkl` files (see paths below).

## Data layout
The script expects the following paths (update in `train_SITReg.py` if needed):
- `ATLAS_DIR`: `/scratch2/jchen/DATA/IXI/atlas.pkl`
- `TRAIN_DIR`: `/scratch2/jchen/DATA/IXI/Train/`
- `VAL_DIR`: `/scratch2/jchen/DATA/IXI/Val/`

## Outputs
- Checkpoints: `experiments/SITReg_IXI_ncc_<w1>_diffusion_<w2>/`
- Logs: `logs/SITReg_IXI_ncc_<w1>_diffusion_<w2>/`
- TensorBoard summaries inside the logs folder.

## How to run
From this directory:

`python3.8 -u train_SITReg.py`

## Notes
- Loss weights are configured near the top of `main()` as `weights = [1, 1]`.
- Input shape defaults to `(160, 192, 224)` in `INPUT_SHAPE`.
- If you want to resume training, set `cont_training = True` and update the checkpoint path in the script.
