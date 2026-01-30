Brain template example (LUMIR24)
===============================

This example builds a brain template from the LUMIR24 dataset using either **TransMorphTVF** or **VFA**. The script is self‑contained **except** for the dataset itself: you must download the LUMIR24 data in advance.

Prerequisites
-------------

- LUMIR24 dataset downloaded locally.
- CUDA‑enabled PyTorch (recommended).
- Internet access on first run to download pretrained weights and the LUMIR JSON file.

Dataset requirement
-------------------

Download LUMIR24 and set the base directory in the script:

- ``LUMIR_BASE_DIR``: root folder that contains the LUMIR24 images referenced by ``LUMIR_dataset.json``.

The script will download ``LUMIR_dataset.json`` automatically if it is missing, but **the images themselves must be present**.

Run the example
---------------

From ``tutorials/build_brain_template``:

.. code-block:: bash

   python3.8 -u build_template.py

Configuration
-------------

Edit these settings near the top of ``build_template.py``:

- ``MODEL_TYPE``: ``"TransMorphTVF"`` or ``"VFA"``
- ``LUMIR_BASE_DIR``: path to your local LUMIR24 data
- ``WEIGHTS_PATH``: folder for pretrained weights (auto‑download)
- ``OUT_DIR``: output folder for NIfTI templates
- ``NUM_ITERS``: number of template refinement iterations
- ``SHAPE_AVG_LOGDOMAIN``: log‑domain (velocity) averaging vs. flow averaging

Outputs
-------

Templates are saved as NIfTI files in ``template_outputs``:

- ``template_outputs/template_iter_00.nii.gz``
- ``template_outputs/template_iter_01.nii.gz``
- ...
