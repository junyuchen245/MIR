# Brain Template Building (LUMIR24)

This example builds a brain template from the LUMIR24 dataset using either **TransMorphTVF** or **VFA**. The script is self‑contained **except** for the dataset itself: you must download the LUMIR24 data in advance.

## Requirements

- LUMIR24 dataset downloaded locally.
- A CUDA‑enabled PyTorch environment (recommended for speed).
- Internet access on first run to download the pretrained weights and the LUMIR JSON file.

## Dataset prerequisite (required)

Download LUMIR24 and set the base directory in the script:

- `LUMIR_BASE_DIR`: root folder that contains the LUMIR24 images referenced by `LUMIR_dataset.json`.

The script will download `LUMIR_dataset.json` automatically if it is missing, but **the images themselves must be present**.

## Configure the script

Open `build_template.py` and edit these settings near the top:

- `MODEL_TYPE`: `"TransMorphTVF"` or `"VFA"`
- `LUMIR_BASE_DIR`: path to your local LUMIR24 data
- `WEIGHTS_PATH`: folder for pretrained weights (auto‑download)
- `OUT_DIR`: output folder for NIfTI templates
- `NUM_ITERS`: number of template refinement iterations
- `SHAPE_AVG_LOGDOMAIN`: log‑domain (velocity) averaging vs. flow averaging

## Run

From this folder:

```bash
python3.8 -u build_template.py
```

## Outputs

Generated templates are saved as NIfTI files:

```
template_outputs/template_iter_00.nii.gz
template_outputs/template_iter_01.nii.gz
...
```

## Notes

- The script downloads pretrained weights and the LUMIR dataset JSON automatically.
- If `MODEL_TYPE="TransMorphTVF"`, the model runs on downsampled inputs and the flow is upsampled for warping.
- If `MODEL_TYPE="VFA"`, the model runs at full resolution.
- For shape averaging, log‑domain averaging (`SHAPE_AVG_LOGDOMAIN=True`) is recommended for diffeomorphic consistency.
