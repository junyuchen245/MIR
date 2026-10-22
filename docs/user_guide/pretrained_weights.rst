Pretrained weights
==================

Checkpoints are hosted on Google Drive and registered by name in ``MIR.ModelWeights``. Each entry gives
the Drive file id (``wts``) and the key under which the state dict is stored (``wts_key``).

Loading a checkpoint
--------------------

.. code-block:: python

   import gdown, torch
   from MIR import ModelWeights

   entry = ModelWeights['TransMorphTVF-LUMIR24-MonoModal']
   gdown.download(f"https://drive.google.com/uc?id={entry['wts']}", 'TransMorphTVF_LUMIR24.pth.tar')
   model.load_state_dict(torch.load('TransMorphTVF_LUMIR24.pth.tar')[entry['wts_key']])

To initialize a modified architecture from a checkpoint, load only the tensors whose names and shapes
match:

.. code-block:: python

   from MIR.utils import load_partial_weights
   load_partial_weights(model, 'TransMorphTVF_LUMIR24.pth.tar', weights_key='state_dict')

Available checkpoints
---------------------

.. list-table::
   :header-rows: 1
   :widths: 36 22 42

   * - ``ModelWeights`` key
     - Model
     - Trained on
   * - ``TransMorphTVF-LUMIR24-MonoModal``
     - TransMorph-TVF
     - LUMIR24 brain T1w MRI, inter-subject (Learn2Reg 2024)
   * - ``VFA-LUMIR24-MonoModal``
     - VFA
     - LUMIR24 brain T1w MRI (key ``model_state_dict``)
   * - ``VFA-LUMIR25-MultiModal``
     - VFA
     - LUMIR25 brain MRI, multiple contrasts (Learn2Reg 2025)
   * - ``TransMorph-IXI-Atlas2Subject``
     - TransMorph
     - IXI brain MRI, atlas-to-subject
   * - ``TransMorphTVF-IXI-Atlas2Subject``
     - TransMorph-TVF
     - IXI brain MRI, atlas-to-subject
   * - ``VFA-IXI-Atlas2Subject``
     - VFA
     - IXI brain MRI, atlas-to-subject
   * - ``VFA-SPR-IXI-Atlas2Subject``
     - VFA-SPR
     - IXI brain MRI, atlas-to-subject
   * - ``SITReg-IXI-Atlas2Subject``
     - SITReg
     - IXI brain MRI, atlas-to-subject
   * - ``SITReg-SPR-IXI-Atlas2Subject``
     - SITReg-SPR
     - IXI brain MRI, atlas-to-subject
   * - ``MedIA-TM-SPR-Beta-autoPET``
     - TransMorph-TVF-SPR, Beta prior
     - autoPET whole-body PET/CT, atlas-to-subject (MedIA 2025)
   * - ``MedIA-TM-SPR-Gaussian-autoPET``
     - TransMorph-TVF-SPR, Gaussian prior
     - autoPET whole-body PET/CT, atlas-to-subject (MedIA 2025)
   * - ``VFA-SegHead``
     - VFA segmentation head
     - VFA features for segmentation
   * - ``VFA-SynthHead``
     - VFA synthesis head
     - VFA features for image synthesis

Datasets and atlases
--------------------

``MIR.DatasetJSONs`` holds the dataset split files of LUMIR24 and LUMIR25, ``MIR.ValEvalModules`` the
official validation-evaluation modules, and ``MIR.pretrained_wts.FDG_atlas_autoPET`` an FDG PET/CT atlas
built from autoPET (CT, SUV, organ probability and label atlases, 2.8 x 2.8 x 3.8 mm, sex-specific
variants).
