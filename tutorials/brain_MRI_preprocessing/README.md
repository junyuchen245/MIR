# Brain MRI preprocessing pipeline

## STEP 1 Skull stripping with SynthStrip

Visit the [official website](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) of SynthStrip to download Docker image. Then run:

`python3.8 -u run_synthstrip.py`

## STEP 2 MRI preprocessing

`python3.8 preprocess.py sub-01_T1w.nii.gz template_img.nii.gz output.nii.gz`