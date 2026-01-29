# IXI Benchmarking Tutorial

This tutorial benchmarks multiple registration models on the IXI dataset, including TransMorph, TransMorphTVF, SITReg, and SITReg with SPR.

## What’s included
- Training scripts:
  - `train_TransMorph.py`
  - `train_TransMorphTVF.py`
  - `train_SITReg.py`
  - `train_SITReg_SPR.py`
- Notebook: `IXI_benchmarking.ipynb` (model setup and dataset download helpers)
- Data utilities in `data/`

## Dataset
The notebook downloads a zipped IXI test set to `test_dir` and expects it to be extracted there.

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

## Notes
- Update dataset paths inside each script as needed.
- GPU recommended for training.
- Use TensorBoard to inspect logs in `logs/`.
