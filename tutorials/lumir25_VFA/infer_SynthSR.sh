#!/usr/bin/env bash

docker pull jchen245/synthsr_build:synthsr_cpu_v0

docker run --rm \
        --ipc=host \
        --memory 256g \
        --mount type=bind,source=/scratch/jchen/DATA/LUMIR/LUMIR25/imagesVal,target=/input \
        --mount type=bind,source=/scratch/jchen/DATA/LUMIR/LUMIR25/imagesSynthSR,target=/output \
        jchen245/synthsr_build:synthsr_cpu_v0

rename '_SynthSR.nii.gz' '.nii.gz' /scratch/jchen/DATA/LUMIR/LUMIR25/imagesSynthSR/LUMIRMRI*.gz

cp /scratch/jchen/DATA/LUMIR/LUMIR25/imagesVal/LUMIRMRI_3*.gz /scratch/jchen/DATA/LUMIR/LUMIR25/imagesSynthSR/
cp /scratch/jchen/DATA/LUMIR/LUMIR25/imagesVal/LUMIRMRI_{4014..4023}_0000*.gz /scratch/jchen/DATA/LUMIR/LUMIR25/imagesSynthSR/
