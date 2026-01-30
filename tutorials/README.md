## Overview

This directory provides tutorials on how to run various models in MIR, including training, inference, and evaluation workflows. It also includes references to pretrained weights and dataset resources for reproducing published results.

## Tutorials at a glance

- **LUMIR24/25**: Training/inference pipelines for TransMorph and VFA on the Learn2Reg challenges.
- **IXI**: Brain registration pipelines for IXI (including SITReg-based training).
- **AutoPET**: PET/CT registration workflows and evaluation utilities.

Each tutorial folder contains a runnable script and, when applicable, a README with dataset paths, outputs, and notes.

## Tutorial summaries

- [IXI_SITReg](https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_SITReg): SITReg training and validation on IXI with NCC + diffusion regularization.
- [IXI_HyperTransMorph](https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_HyperTransMorph): HyperTransMorph-based IXI registration with hyperparameter conditioning.
- [IXI_HyperMorph](https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_HyperMorph): HyperMorph (VoxelMorph-style) IXI registration and evaluation.
- [IXI_TransMorphSPR](https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_TransMorphSPR): TransMorph with spatially varying regularization on IXI.
- [IXI_deedsBCV](https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_deedsBCV): deedsBCV validation on IXI (Linux-only, uses bundled binaries).
- [lumir24_TransMorph](https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir24_TransMorph): TransMorph pipelines for Learn2Reg 2024.
- [lumir24_VFA](https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir24_VFA): VFA pipelines for Learn2Reg 2024.
- [lumir24_ConvexAdam](https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir24_ConvexAdam): ConvexAdam baselines for Learn2Reg 2024.
- [lumir25_VFA](https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_VFA): VFA pipelines for Learn2Reg 2025.
- [lumir25_SITReg](https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_SITReg): SITReg pipelines for Learn2Reg 2025.
- [lumir25_ConvexAdam](https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_ConvexAdam): ConvexAdam baselines for Learn2Reg 2025.
- [autoPET_TransMorphSPR](https://github.com/junyuchen245/MIR/tree/main/tutorials/autoPET_TransMorphSPR): PET/CT registration with TransMorph SPR.
- [autoPET_ConvexAdamSPR](https://github.com/junyuchen245/MIR/tree/main/tutorials/autoPET_ConvexAdamSPR): PET/CT registration with ConvexAdam SPR.
- [autoPET_FireANTs_SPR](https://github.com/junyuchen245/MIR/tree/main/tutorials/autoPET_FireANTs_SPR): PET/CT registration with FireANTs SPR baselines.
- [VFA_image_seg](https://github.com/junyuchen245/MIR/tree/main/tutorials/VFA_image_seg): VFA-based feature extraction for segmentation tasks.
- [VFA_image_synthesis](https://github.com/junyuchen245/MIR/tree/main/tutorials/VFA_image_synthesis): VFA-based feature extraction for synthesis tasks.
- [brain_MRI_preprocessing](https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_MRI_preprocessing): MRI preprocessing utilities (bias correction, normalization, masking).
- [IPEN_registration](https://github.com/junyuchen245/MIR/tree/main/tutorials/IPEN_registration): Registration workflows for the IPEN dataset.
- [pretraining_registration_DNNs](https://github.com/junyuchen245/MIR/tree/main/tutorials/pretraining_registration_DNNs): Random-image pretraining pipelines for registration networks.

|Model/File|Google Drive ID |Dict Key|
|---|---|---|
|VFA (LUMIR24) - mono-contrast| `17XEfRYJbnrtCVhaBCOvQVOLkWhix9PAK` |`'model_state_dict'`|
|VFA (LUMIR25) - mono- & multi-contrast| `1cDY3isltI-uSCiivgP2zcx_5LeR8vIJ6` |`'state_dict'`|
|TransMorph (LUMIR24)- mono-contrast|`1SSqI88l1MdrPJgE4Rn8pqXnVfZNPxtry`|`'state_dict'`|
|MedIA-TM-SPR-Beta-autoPET|`1NCdoK4khv4j8JAjlgeJo6EB4dSQr83r6`|`'state_dict'`|
|MedIA-TM-SPR-Gaussian-autoPET|`1WqiR5YB8ypx-NUvYU_kQMb3edRVOLi09`|`'state_dict'`|
|LUMIR_dataset.json|`1b0hyH7ggjCysJG-VGvo38XVE8bFVRMxb`|-|
|LUMIR25_dataset.json|`164Flc1C6oufONGimvpKlrNtq5t3obXEo`|-|

## Datasets

|Dataset|Link|
|---|---|
|IXI| https://github.com/junyuchen245/Preprocessed_IXI_Dataset|
|LUMIR24|https://learn2reg.grand-challenge.org/learn2reg-2024/|
|LUMIR25|https://learn2reg.grand-challenge.org/learn2reg-2025/|

## How to run a tutorial

1. **Locate the folder** for the target tutorial (e.g., `IXI_SITReg`, `lumir25_VFA`).
2. **Open the script** and set dataset paths at the top of the file.
3. **Run the script** from inside the tutorial directory.
4. **Check outputs** in the `experiments/` and `logs/` folders created by the script.

## Outputs

Most training scripts write:
- **Checkpoints**: under `experiments/`.
- **Logs**: under `logs/`, often compatible with TensorBoard.
- **Figures**: optional visualizations saved per epoch.

## Notes

- Some tutorials rely on external datasets and large pretrained models.
- GPU is recommended for training-based scripts.
- If you add a new tutorial, include a short README in its folder.