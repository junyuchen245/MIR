#!/usr/bin/env bash
bash ./build.sh
#docker load --input reg_model.tar.gz

docker run --rm \
    --ipc=host \
    --memory 256g \
    --gpus "device=0" \
    --user $(id -u):$(id -g) \
    --network=none \
    --mount type=bind,source=/scratch/jchen/python_projects/custom_packages/MIR/tutorials/lumir25_VFA/LUMIR25_Docker_Example/LUMIR25_dataset.json,target=/app/LUMIR25_dataset.json \
    --mount type=bind,source=/scratch/jchen/DATA/LUMIR/LUMIR25,target=/app/input \
    --mount type=bind,source=/scratch/jchen/python_projects/custom_packages/MIR/tutorials/lumir25_VFA/LUMIR25_Docker_Example/LUMIR_VFAlumir25_TestPhase,target=/app/output \
    vfa_lumir25
