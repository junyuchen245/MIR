# Brain MRI preprocessing pipeline

## STEP 1 Skull stripping with SynthStrip

Visit the [official website](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) of SynthStrip to download Docker image. Then run:

`python3.8 -u run_synthstrip.py`

## STEP 2 MRI preprocessing
Affine align to the MNI template\
`python3.8 preprocess.py sub-01_T1w.nii.gz mni_icbm152_2009c_t1_1mm_masked_img.nii.gz output.nii.gz`

Affine alignment to the LUMIR template for consistency with the LUMIR dataset\
`python3.8 preprocess.py sub-01_T1w.nii.gz LUMIR_template.nii.gz output.nii.gz`