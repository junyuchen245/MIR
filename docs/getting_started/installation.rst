Installation
============

.. note::
   MIR is a research toolkit installed from source. Install PyTorch with CUDA support **before** MIR.

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
``statsmodels``, ``natsort``) are installed automatically. Test and developer tools are an
optional extra (see below).

Optional dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Needed for
   * - ``gdown`` (``pip install -e .[download]``)
     - Automatic download of pretrained weights and dataset JSON files in the tutorials.
   * - ``pydicom`` (``pip install -e .[dicom]``)
     - DICOM utilities (``mir-dicom-info`` command line tool, ``MIR.utils.dicom_utils``).
   * - ``tensorboard``
     - Training curves written by the tutorial training scripts.
   * - ``pytest``, ``ruff``, ``pre-commit`` (``pip install -e .[dev]``)
     - Running the test suite, linting and formatting (see :doc:`../development/contributing`).
   * - Sphinx and theme (``pip install -e .[docs]``)
     - Building this documentation locally.

Platform notes
--------------

- **deedsBCV** is integrated through a compiled Linux binary; it is unavailable on Windows
  and macOS. Everything else in MIR is platform independent.
- The FireANTs wrapper is imported lazily; the rest of ``MIR.models`` works without it.
- Mixed precision (float16, bfloat16) and multi-GPU training are supported by every model but are opt-in; the
  tutorials run in float32 on one GPU. bfloat16 needs a GPU with compute capability 8.0+ (A100, H100, RTX 30xx/40xx)
  or the CPU. See :doc:`../user_guide/precision_multigpu`.
- ``torchrun`` belongs to the Python environment it was installed with; if MIR lives in another interpreter, launch
  distributed jobs with ``python3.8 -m torch.distributed.run --nproc_per_node=N script.py``.

Verifying the installation
--------------------------

.. code-block:: bash

   python3.8 -c "import MIR, torch; print(MIR.__version__, MIR.__file__, torch.cuda.is_available())"
   python3.8 -m pip install -e .[dev]
   python3.8 -m pytest tests -q      # unit tests (CPU, about a minute)

Pretrained weights
------------------

.. seealso:: :doc:`../user_guide/pretrained_weights` lists every available checkpoint.

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

   python3.8 -m pip install -e .[docs]          # or: pip install -r docs/requirements.txt
   python3.8 -m sphinx -b html docs docs/_build/html

The published documentation is rebuilt automatically from the ``main`` branch.
