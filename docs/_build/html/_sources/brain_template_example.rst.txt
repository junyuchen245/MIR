Brain template example
================================

This example builds a population brain template from the LUMIR24 dataset using
**TransMorphTVF**, **VFA**, or **ConvexAdam**. The script is self‑contained
**except** for the dataset itself: you must download LUMIR24 dataset in advance.

Pipeline overview
-----------------
1. Initialize the model (TransMorphTVF, VFA, or ConvexAdam).
2. For a set number of iterations:
    - Register all subjects to the current template.
    - Average the deformed images in either log‑domain (velocity) or
      displacement field domain to form a new template.
3. Save the intermediate templates as NIfTI files.

Prerequisites
-------------

- LUMIR24 dataset downloaded locally.
- CUDA‑enabled PyTorch (recommended).
- Internet access on first run to download pretrained weights (TransMorphTVF/VFA) and the LUMIR JSON file.

Dataset requirement
-------------------

Download LUMIR24 and set the base directory in the script:

- ``LUMIR_BASE_DIR``: root folder that contains the LUMIR24 images referenced by ``LUMIR_dataset.json``.

The script will download ``LUMIR_dataset.json`` automatically if it is missing, but **the images themselves must be present**.

Run the example
---------------

From `tutorials/build_brain_template <https://github.com/junyuchen245/MIR/tree/main/tutorials/build_brain_template>`_:

.. code-block:: bash

   python3.8 -u build_template.py

Configuration
-------------

Edit these settings near the top of ``build_template.py`` to match your environment and
method choice:

- ``MODEL_TYPE``: ``"TransMorphTVF"``, ``"VFA"``, or ``"ConvexAdam"``
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

Notes
-----

- Pretrained weights are downloaded only for ``TransMorphTVF`` and ``VFA``.
- ``ConvexAdam`` does not require pretrained weights.
