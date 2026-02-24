IXI benchmarking quick start
============================

Use the IXI benchmarking workflow in
`tutorials/IXI_benchmarking <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_benchmarking>`_
to run quick comparisons across registration models on IXI.
The notebook
`tutorials/IXI_benchmarking/IXI_benchmarking.ipynb <https://github.com/junyuchen245/MIR/blob/main/tutorials/IXI_benchmarking/IXI_benchmarking.ipynb>`_
is the canonical end-to-end benchmark pipeline.

Models covered
--------------

- ``TransMorph``
- ``TransMorphTVF``
- ``VFA``
- ``VFA-SPR``
- ``SITReg``
- ``SITReg-SPR``
- ``ConvexAdam-MIND``

Model references
----------------

- ``TransMorph``:
    `TransMorph: Transformer for Unsupervised Medical Image Registration (MedIA 2022) <https://arxiv.org/abs/2111.10480>`_
- ``TransMorphTVF``:
    `Unsupervised Learning of Diffeomorphic Image Registration via TransMorph (WBIR 2022) <https://link.springer.com/chapter/10.1007/978-3-031-11203-4_11>`_
- ``VFA``:
    `Vector Field Attention for Deformable Image Registration (JMI 2024) <https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-11/issue-6/064001/Vector-field-attention-for-deformable-image-registration/10.1117/1.JMI.11.6.064001.full>`_
- ``VFA-SPR``:
    Uses the VFA backbone with SPR regularization from
    `Unsupervised Learning of Spatially Varying Regularization for Diffeomorphic Image Registration (MedIA 2025) <https://arxiv.org/abs/2412.17982>`_.
- ``SITReg``:
    `SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration (MELBA 2024) <https://www.melba-journal.org/papers/2024:026.html>`_
- ``SITReg-SPR``:
    Uses the SITReg backbone with SPR regularization from
    `Unsupervised Learning of Spatially Varying Regularization for Diffeomorphic Image Registration (MedIA 2025) <https://arxiv.org/abs/2412.17982>`_.
- ``ConvexAdam-MIND``:
    `ConvexAdam: Self-configuring dual-optimization-based 3D medical image registration (IEEE TMI 2024) <https://ieeexplore.ieee.org/abstract/document/10681158>`_

Environment setup
-----------------

.. code-block:: bash

    python3.8 -m pip install -U pip
    python3.8 -m pip install -e .
    python3.8 -m pip install gdown nibabel scipy matplotlib pandas natsort

Run benchmark (notebook)
------------------------

1. Open ``IXI_benchmarking.ipynb``.
2. Run all cells from top to bottom.
3. Ensure ``models_dict`` includes the exact model set you want.
4. Run the evaluation loop and analysis cells.

Data and pretrained weights
---------------------------

The notebook automatically downloads:

- IXI test data zip (if missing), then extracts to ``./IXI_data/``
- Pretrained model checkpoints to ``./pretrain_wts/``

Expected files:

- ``./IXI_data/Test/*.pkl``
- ``./IXI_data/atlas.pkl``
- ``./pretrain_wts/*.pth.tar``

Outputs
-------

Per-model CSVs are written to ``./results``:

- ``./results/TransMorph.csv``
- ``./results/TransMorphTVF.csv``
- ``./results/VFA.csv``
- ``./results/VFA-SPR.csv``
- ``./results/SITReg.csv``
- ``./results/SITReg-SPR.csv``
- ``./results/ConvexAdam-MIND.csv``

Each CSV includes organ-wise Dice plus:

- ``ndv``: non-diffeomorphic volume percentage
- ``ndp``: non-diffeomorphic points percentage

The final analysis cell aggregates Dice, prints summary statistics, and creates comparative boxplots.

Reproducibility checklist
-------------------------

- Keep image size fixed at ``(160, 192, 224)``.
- Keep ``batch_size=1`` for stable memory usage.
- Use the same pretrained checkpoint files across all models.
- For strict reproducibility, set random seeds and deterministic PyTorch options before model creation.

Troubleshooting
---------------

- If data download fails, verify Google Drive access and rerun the download cell.
- If package import fails, install missing dependencies in the active Python environment.
- If CUDA memory is insufficient, run one model at a time and free other GPU processes.
- ``deedsBCV`` is only supported on Linux and requires the executable in path/discovery.

Notebook snippets
-----------------

Imports and setup
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torch.utils.tensorboard import SummaryWriter
   import os
   import sys
   import glob

   from torch.utils.data import DataLoader
   import numpy as np
   import torch
   from torch import optim
   import matplotlib.pyplot as plt
   from natsort import natsorted
   from torchvision import transforms

   from MIR.models import SpatialTransformer, EncoderFeatureExtractor, SITReg, VFA, TransMorphTVF, TransMorph, convex_adam_MIND
   from MIR.models.SITReg import ReLUFactory, GroupNormalizerFactory
   from MIR.models.SITReg.composable_mapping import DataFormat
   from MIR.models.SITReg.deformation_inversion_layer.fixed_point_iteration import (
       AndersonSolver,
       AndersonSolverArguments,
       MaxElementWiseAbsStopCriterion,
       RelativeL2ErrorStopCriterion,
   )
   import MIR.models.convexAdam.configs_ConvexAdam_MIND as CONFIGS_CVXAdam
   import MIR.models.configs_TransMorph as configs_TransMorph
   import MIR.models.configs_VFA as CONFIGS_VFA
   from data import datasets, trans
   import torch.nn.functional as F

Dataset paths
^^^^^^^^^^^^^

.. code-block:: python

   H, W, D = 160, 192, 224
   data_dir = './IXI_data/'

Model factory functions
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def transmorph_model():
       scale_factor = 1
       config = configs_TransMorph.get_3DTransMorph3Lvl_config()
       config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
       config.window_size = (H // 64, W // 64, D // 64)
       config.out_chan = 3
       TM_model = TransMorph(config).cuda('cuda:0')
       return TM_model

   def transmorphTVF_model():
       scale_factor = 2
       config = configs_TransMorph.get_3DTransMorph3Lvl_config()
       config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
       config.window_size = (H // 64, W // 64, D // 64)
       config.out_chan = 3
       TMTVF_model = TransMorphTVF(config, time_steps=7).cuda('cuda:0')
       return TMTVF_model

   def vfa_model():
       scale_factor = 1
       config = CONFIGS_VFA.get_VFA_default_config()
       config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
       VFA_model = VFA(config, device='cuda:0')
       return VFA_model

   def create_model(INPUT_SHAPE) -> SITReg:
       feature_extractor = EncoderFeatureExtractor(
           n_input_channels=1,
           activation_factory=ReLUFactory(),
           n_features_per_resolution=[12, 16, 32, 64, 128, 128],
           n_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
           input_shape=INPUT_SHAPE,
           normalizer_factory=GroupNormalizerFactory(2),
       ).cuda()
       AndersonSolver_forward = AndersonSolver(
           MaxElementWiseAbsStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
           AndersonSolverArguments(memory_length=4),
       )
       AndersonSolver_backward = AndersonSolver(
           RelativeL2ErrorStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
           AndersonSolverArguments(memory_length=4),
       )
       network = SITReg(
           feature_extractor=feature_extractor,
           n_transformation_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
           n_transformation_features_per_resolution=[12, 64, 128, 256, 256, 256],
           max_control_point_multiplier=0.99,
           affine_transformation_type=None,
           input_voxel_size=(1.0, 1.0, 1.0),
           input_shape=INPUT_SHAPE,
           transformation_downsampling_factor=(1.0, 1.0, 1.0),
           forward_fixed_point_solver=AndersonSolver_forward,
           backward_fixed_point_solver=AndersonSolver_backward,
           activation_factory=ReLUFactory(),
           normalizer_factory=GroupNormalizerFactory(4),
       ).cuda()
       return network

   def sitreg_model():
       return create_model((H, W, D)).cuda('cuda:0')

   def convexadam_model():
       config = CONFIGS_CVXAdam.get_ConvexAdam_MIND_brain_default_config()
       model = convex_adam_MIND
       return {'model': model, 'config': config}

   models_dict = {
       'TransMorph': transmorph_model,
       'TransMorphTVF': transmorphTVF_model,
       'VFA': vfa_model,
       'SITReg': sitreg_model,
       'ConvexAdam-MIND': convexadam_model,
   }

Data download and extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   if not os.path.exists(data_dir):
       import gdown
       file_id = '1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol'
       url = f"https://drive.google.com/uc?id={file_id}"
       gdown.download(url, 'IXI_data.zip', quiet=False)
       import zipfile
       with zipfile.ZipFile('IXI_data.zip', 'r') as zip_ref:
           zip_ref.extractall('./')
       os.remove('IXI_data.zip')

   if not os.path.exists(data_dir):
       raise ValueError('Data directory not found and download failed.')

Inference loop
^^^^^^^^^^^^^^

.. code-block:: python

   spatial_trans = SpatialTransformer((H, W, D)).cuda()

   def inference(model_name, model, moving, fixed):
       if model_name == "TransMorph" or model_name == 'VFA':
           with torch.no_grad():
               model.eval()
               moving = moving.cuda()
               fixed = fixed.cuda()
               flow = model((moving, fixed))
       elif model_name == "TransMorphTVF":
           with torch.no_grad():
               model.eval()
               moving = F.avg_pool3d(moving, 2).cuda()
               fixed = F.avg_pool3d(fixed, 2).cuda()
               flow = model((moving, fixed))
               flow = F.interpolate(flow, size=(H, W, D), mode='trilinear', align_corners=True) * 2.0
       elif model_name == "SITReg":
           with torch.no_grad():
               model.eval()
               moving = moving.cuda()
               fixed = fixed.cuda()
               mapping_pair = model(moving, fixed, mappings_for_levels=((0, False),))[0]
               flow = mapping_pair.forward_mapping.sample(data_format=DataFormat.voxel_displacements()).generate_values()
       elif model_name == "ConvexAdam":
           moving = moving.cuda()
           fixed = fixed.cuda()
           convexadam = model()['model']
           config = model()['config']
           flow = convexadam(moving, fixed, config)
       return flow
