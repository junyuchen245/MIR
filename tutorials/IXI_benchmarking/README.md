# IXI Benchmarking Tutorial

This tutorial benchmarks multiple registration models on the IXI dataset, including TransMorph, TransMorphTVF, VFA, VFA with SPR, SITReg, SITReg with SPR, and ConvexAdam.

- TransMorph:
  - Chen, Junyu, et al. "Transmorph: Transformer for unsupervised medical image registration." Medical image analysis 82 (2022): 102615.
- TransMorphTVF:
  - Chen, Junyu, Eric C. Frey, and Yong Du. "Unsupervised learning of diffeomorphic image registration via transmorph." International Workshop on Biomedical Image Registration. Cham: Springer International Publishing, 2022.
- SPR (a special type of regularization for image registration):
  - Chen, Junyu, et al. "Unsupervised learning of spatially varying regularization for diffeomorphic image registration." Medical image analysis (2025): 103887.
- VFA:
  - Liu, Yihao, et al. "Vector field attention for deformable image registration." Journal of Medical Imaging 11.6 (2024): 064001-064001.
- SITReg:
  - Honkamaa, Joel, and Pekka Marttinen. "SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration." The Journal of Machine Learning for Biomedical Imaging 2 (2024): 2148-2194.
- ConvexAdam:
  - Siebert, Hanna, et al. "Convexadam: Self-configuring dual-optimization-based 3d multitask medical image registration." IEEE Transactions on Medical Imaging 44.2 (2024): 738-748.


## What’s included
- Training scripts:
  - `train_TransMorph.py`
  - `train_TransMorphTVF.py`
  - `train_SITReg.py`
  - `train_SITReg_SPR.py`
  - `train_VFA.py`
  - `train_VFA_SPR.py`
- Notebook: `IXI_benchmarking.ipynb` (model setup and dataset download helpers)
- Data utilities in `data/`

## Dataset
The notebook downloads a zipped IXI test set to `test_dir` and expects it to be extracted there.

The notebook also downloads pretrained weights to `pretrained_wts`.
## Outputs
Each script writes:
- Checkpoints: `experiments/<run_name>/`
- Logs: `logs/<run_name>/`

## How to run
From this directory:

- `python3.8 -u train_TransMorph.py`
- `python3.8 -u train_TransMorphTVF.py`
- `python3.8 -u train_SITReg.py`
- `python3.8 -u train_SITReg_SPR.py`
- `python3.8 -u train_VFA.py`
- `python3.8 -u train_VFA_SPR.py`

## Notes
- Update dataset paths inside each script as needed.
- GPU recommended for training.
- Use TensorBoard to inspect logs in `logs/`.
