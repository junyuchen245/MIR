Installation
============

Requirements
------------

- Python 3.8 or newer (the tutorials in the repository are run with ``python3.8``).
- A CUDA-enabled PyTorch build (``torch >= 2.0.1``). Install it first, following the
  `PyTorch instructions <https://pytorch.org/get-started/locally/>`_ for your CUDA version;
  MIR does not pin a CUDA build.
- An NVIDIA GPU is strongly recommended. Most models run on 3-D volumes at full resolution
  and need tens of gigabytes of GPU memory for training; inference needs much less.

Installing MIR
--------------

Editable install from a clone of the repository (recommended, keeps the tutorials next to
the package):

.. code-block:: bash

   git clone https://github.com/junyuchen245/MIR.git
   cd MIR
   python3.8 -m pip install -U pip
   python3.8 -m pip install -e .

The core dependencies (``numpy``, ``scipy``, ``nibabel``, ``SimpleITK``, ``einops``,
``timm``, ``ml_collections``, ``antspyx``, ``pymedio``, ``matplotlib``, ``seaborn``,
``statsmodels``, ``natsort``, ``pytest``) are installed automatically.

Optional dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Needed for
   * - ``gdown``
     - Automatic download of pretrained weights and dataset JSON files in the tutorials.
   * - ``pydicom`` (``pip install -e .[dicom]``)
     - DICOM utilities (``mir-dicom-info`` command line tool, ``MIR.utils.dicom_utils``).
   * - ``tensorboard``
     - Training curves written by the tutorial training scripts.
   * - ``black`` (``pip install -e .[dev]``)
     - Code formatting for contributors.

Platform notes
--------------

- **deedsBCV** is integrated through a compiled Linux binary; it is unavailable on Windows
  and macOS. Everything else in MIR is platform independent.
- The FireANTs wrapper is imported lazily; the rest of ``MIR.models`` works without it.
- Mixed precision is not used by the tutorials. The LDDMM components perform their FFTs in
  float32 regardless of the autocast context.

Verifying the installation
--------------------------

.. code-block:: bash

   python3.8 -c "import MIR, torch; print(MIR.__file__, torch.cuda.is_available())"
   python3.8 -m pytest tests -q      # unit tests (CPU, a few minutes)

Pretrained weights
------------------

Pretrained checkpoints are hosted on Google Drive and referenced by name in
``MIR.ModelWeights`` (``MIR/pretrained_wts.py``). Each entry gives the Drive file id and the
key under which the state dict is stored:

.. code-block:: python

   import gdown, torch
   from MIR import ModelWeights

   entry = ModelWeights['TransMorphTVF-LUMIR24-MonoModal']
   gdown.download(f"https://drive.google.com/uc?id={entry['wts']}", 'TransMorphTVF_LUMIR24.pth.tar')
   state_dict = torch.load('TransMorphTVF_LUMIR24.pth.tar')[entry['wts_key']]

Available entries include LUMIR24 / LUMIR25 brain models (TransMorph-TVF, VFA), IXI
atlas-to-subject models (TransMorph, TransMorph-TVF, VFA, VFA-SPR, SITReg, SITReg-SPR),
PET/CT atlas models (TransMorph-SPR with Beta and Gaussian priors) and VFA segmentation /
synthesis heads. The tutorial scripts download what they need on first run.

Building the documentation
--------------------------

.. code-block:: bash

   python3.8 -m pip install sphinx sphinx-rtd-theme
   sphinx-build -b html docs docs/_build/html

The published documentation is rebuilt automatically from the ``main`` branch.
