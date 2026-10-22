MIR: Medical Image Registration toolkit
========================================

.. image:: https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square
   :alt: License: Apache 2.0
   :target: https://opensource.org/license/apache-2-0
.. image:: https://github.com/junyuchen245/MIR/actions/workflows/tests.yml/badge.svg
   :alt: Tests
   :target: https://github.com/junyuchen245/MIR/actions/workflows/tests.yml
.. image:: https://github.com/junyuchen245/MIR/actions/workflows/docs.yml/badge.svg
   :alt: Docs
   :target: https://github.com/junyuchen245/MIR/actions/workflows/docs.yml

**MIR** is a PyTorch research toolkit for deformable medical image registration. It bundles
the registration networks developed around *TransMorph* (TransMorph, VFA, their spatially
varying regularization and hypernetwork variants, and the new LDDMM models), ports of
third-party learning-based and classical methods behind one calling convention, the loss
functions and regularizers used to train them, evaluation metrics, and end-to-end tutorials
on public benchmarks (IXI, Learn2Reg LUMIR, autoPET) and on atlas building.

| Documentation: https://junyuchen.me/MIR
| Source code: https://github.com/junyuchen245/MIR

Highlights
----------

- **One convention for every model.** Each registration network takes a ``(moving, fixed)``
  pair of ``[B, 1, D, H, W]`` tensors and returns a dense displacement field in voxels that
  ``MIR.models.SpatialTransformer`` applies to any image or label map. Swapping architectures
  is a one-line change.
- **A family of transformation models.** Plain displacements, stationary velocity fields
  (scaling and squaring), time-varying compositions (TransMorph-TVF) and, new, **LDDMM**
  geodesic shooting with learned initial momentum and an exact inverse map
  (:doc:`models/lddmm`). The LDDMM layer wraps *any* backbone in the toolkit.
- **Spatially varying regularization (SPR)** as a learned per-voxel map, either weighting
  the regularization loss (TransMorph-SPR, VFA-SPR, SITReg-SPR, LessNet-SPR, Fourier-Net-SPR)
  or, in the LDDMM models, setting the kernel scale of the metric itself.
- **Hypernetwork conditioning** (HyperMorph, HyperTransMorph, HyperVFA) to train one model
  across a range of regularization strengths.
- **Classical and third-party baselines** behind the same interface: ConvexAdam, deedsBCV,
  FireANTs (greedy / SyN), SITReg, MultiMorph, and intensity-normalization utilities.
- **Everything needed to train and evaluate**: similarity losses (NCC variants, SSIM, MIND,
  mutual information, correlation ratio, Dice), regularizers (diffusion, local / spatially
  weighted, Beta priors, GradICON, KL), Jacobian-based folding metrics, Dice utilities,
  data augmentation, and a registry of pretrained weights.

Installation
------------

MIR requires Python 3.8 or newer and a CUDA-enabled PyTorch build. Install PyTorch first,
then install MIR in editable mode from the repository root:

.. code-block:: bash

   git clone https://github.com/junyuchen245/MIR.git
   cd MIR
   python3.8 -m pip install -U pip
   python3.8 -m pip install -e .

See :doc:`installation` for optional dependencies, platform notes and how to verify the
installation.

Sixty-second example
--------------------

Register a brain MRI to a template with a pretrained VFA model and warp the image:

.. code-block:: python

   import torch, nibabel as nib
   from MIR import ModelWeights
   from MIR.models import VFA, SpatialTransformer
   import MIR.models.configs_VFA as configs_VFA

   mov = torch.from_numpy(nib.load('subject.nii.gz').get_fdata()[None, None]).float().cuda()
   fix = torch.from_numpy(nib.load('template.nii.gz').get_fdata()[None, None]).float().cuda()

   config = configs_VFA.get_VFA_default_config()
   config.img_size = fix.shape[2:]
   model = VFA(config, device='cuda:0').cuda().eval()
   model.load_state_dict(torch.load('VFA_LUMIR24.pth')[ModelWeights['VFA-LUMIR24-MonoModal']['wts_key']])

   with torch.no_grad():
       flow = model((mov, fix))                      # [1, 3, D, H, W] displacement in voxels
   warped = SpatialTransformer(fix.shape[2:]).cuda()(mov, flow)

The same four lines work for TransMorph, SITReg, LessNet, Fourier-Net and VoxelMorph; the
LDDMM wrapper adds a diffeomorphic flow and its inverse around any of them:

.. code-block:: python

   from MIR.models import LDDMMWrapper
   model = LDDMMWrapper(backbone, param='shooting', time_steps=7, sigmas=[1, 2, 4, 8]).cuda()
   out = model((mov, fix), return_dict=True)       # flow, inv_flow, energy (regularizer), kernel weights

Where to go next
----------------

.. list-table::
   :widths: 28 72
   :header-rows: 0

   * - :doc:`installation`
     - Requirements, optional dependencies, platform notes, verifying the install.
   * - :doc:`concepts`
     - Tensor and displacement conventions, the transformation models (displacement, SVF,
       TVF, LDDMM), spatially varying regularization, losses and metrics.
   * - :doc:`quick_start`
     - Task-oriented starts: brain registration with pretrained weights, IXI benchmarking,
       template construction.
   * - :doc:`models/index`
     - One page per model family with variants, usage, pretrained weights and references.
   * - :doc:`tutorials`
     - Catalogue of the runnable tutorial folders (training, inference, evaluation).
   * - :doc:`api`
     - Auto-generated API reference.

Choosing a model
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 16 16 16

   * - Model
     - Architecture
     - Transformation
     - SPR variant
     - Page
   * - TransMorph / TVF
     - Swin-Transformer encoder, conv decoder
     - displacement, SVF, TVF composition
     - TransMorphTVFSPR
     - :doc:`models/transmorph`
   * - TransMorph-LDDMM
     - TransMorph + LDDMM shooting
     - geodesic (diffeomorphic, invertible)
     - SPR-scaled kernel
     - :doc:`models/lddmm`
   * - LDDMMWrapper
     - any backbone + LDDMM shooting
     - geodesic / relaxation
     - SPR-scaled kernel
     - :doc:`models/lddmm`
   * - VFA / TransVFA
     - vector-field attention
     - displacement, SVF
     - VFASPR
     - :doc:`models/vfa`
   * - HyperTransMorph / HyperVFA / HyperMorph
     - hypernetwork-conditioned
     - as the base model
     - Hyper*SPR
     - :doc:`models/hypertransmorph`, :doc:`models/hypermorph`
   * - VoxelMorph
     - U-Net
     - displacement
     - --
     - :doc:`models/voxelmorph`
   * - LessNet, Fourier-Net(+)
     - light-weight CNNs
     - displacement, SVF
     - LessNetSPR, FourierNet(Plus)SPR
     - :doc:`models/lessnet`, :doc:`models/fouriernet`
   * - SITReg
     - multi-resolution symmetric
     - inverse consistent, topology preserving
     - separate SPR head (tutorials)
     - :doc:`models/sitreg`
   * - ConvexAdam, deedsBCV, FireANTs
     - classical / optimization-based
     - discrete, SVF, greedy / SyN
     - ConvexAdam-SPR, FireANTs-SPR
     - :doc:`models/convexadam`, :doc:`models/deedsbcv`
   * - MultiMorph, TemplateCreation
     - group-wise / atlas building
     - --
     - --
     - :doc:`models/multimorph`

Citing MIR
----------

If MIR is useful in your work, please cite the paper that introduced the model you used
(listed on each model page and under :ref:`included-projects`) and link to the repository.

.. _included-projects:

Included projects
-----------------

1. **Pretraining Deformable Image Registration Networks with Random Images**
   *Chen, Junyu, et al.* ***MIDL Short Papers***, *2025.*
   `Paper <https://openreview.net/forum?id=NJANlZzxfi#discussion>`__ | `Repo <https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs>`__

2. **Correlation Ratio for Unsupervised Learning of Multi-modal Deformable Registration**
   *Chen, Xiaojian, et al.* ***SPIE Medical Imaging: Image Processing***, *2025.*
   `Paper <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13406/1340632/Correlation-ratio-for-unsupervised-learning-of-multi-modal-deformable-registration/10.1117/12.3047699.full>`__ | `Repo <https://github.com/junyuchen245/Correlation_Ratio>`__

3. **Unsupervised Learning of Spatially Varying Regularization for Diffeomorphic Image Registration**
   *Chen, Junyu, et al.* ***Medical Image Analysis***, *2025.*
   `Paper <https://arxiv.org/abs/2412.17982>`__ | `Repo <https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg>`__

4. **Vector Field Attention for Deformable Image Registration**
   *Liu, Yihao, et al.* ***Journal of Medical Imaging***, *2024.*
   `Paper <https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-11/issue-6/064001/Vector-field-attention-for-deformable-image-registration/10.1117/1.JMI.11.6.064001.full>`__ | `Repo <https://github.com/yihao6/vfa/>`__

5. **Unsupervised Learning of Multi-modal Affine Registration for PET/CT**
   *Chen, Junyu, et al.* ***IEEE NSS/MIC/RTSD***, *2024.*
   `Paper <https://arxiv.org/pdf/2409.13863>`__ | `Repo <https://github.com/junyuchen245/Correlation_Ratio>`__

6. **On Finite Difference Jacobian Computation in Deformable Image Registration**
   *Liu, Yihao, et al.* ***International Journal of Computer Vision***, *2024.*
   `Paper <https://link.springer.com/article/10.1007/s11263-024-02047-1>`__ | `Repo <https://github.com/yihao6/digital_diffeomorphism>`__

7. **Deformable Cross-Attention Transformer for Medical Image Registration**
   *Chen, Junyu, et al.* ***MLMI Workshop***, *MICCAI 2023.*
   `Paper <https://arxiv.org/abs/2303.06179>`__ | `Repo <https://github.com/junyuchen245/TransMorph_DCA>`__

8. **Unsupervised Learning of Diffeomorphic Image Registration via TransMorph**
   *Chen, Junyu, et al.* ***WBIR***, *2022.*
   `Paper <https://link.springer.com/chapter/10.1007/978-3-031-11203-4_11>`__ | `Repo <https://github.com/junyuchen245/TransMorph_TVF>`__

9. **TransMorph: Transformer for Unsupervised Medical Image Registration**
   *Chen, Junyu, et al.* ***Medical Image Analysis***, *2022.*
   `Paper <https://arxiv.org/abs/2111.10480>`__ | `Repo <https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration>`__

External packages wrapped
-------------------------

1. **Intensity Normalization Toolkit** -- *Reinhold, Jacob C., et al.* ***SPIE MI***, *2019.*
   `Paper <https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/>`__ | `Repo <https://github.com/jcreinhold/intensity-normalization>`__
2. **ConvexAdam: Dual-Optimization-Based 3D Registration** -- *Siebert, Hanna, et al.* ***IEEE TMI***, *2024.*
   `Paper <https://ieeexplore.ieee.org/abstract/document/10681158>`__ | `Repo <https://github.com/multimodallearning/convexAdam>`__
3. **MultiMorph: On-demand Atlas Construction** -- *Abulnaga, S. Mazdak, et al.* ***CVPR***, *2025.*
   `Paper <https://openaccess.thecvf.com/content/CVPR2025/html/Abulnaga_MultiMorph_On-demand_Atlas_Construction_CVPR_2025_paper.html>`__ | `Repo <https://github.com/mabulnaga/multimorph>`__
4. **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration** -- *Honkamaa, Joel, and Pekka Marttinen.* ***MELBA***, *2024.*
   `Paper <https://www.melba-journal.org/papers/2024:026.html>`__ | `Repo <https://github.com/honkamj/SITReg>`__
5. **FireANTs** -- GPU implementations of greedy and SyN registration. `Repo <https://github.com/rohitrango/fireants>`__
6. **deedsBCV** -- `Repo <https://github.com/mattiaspaul/deedsBCV>`__ (Linux only)
7. **LessNet** and **Fourier-Net / Fourier-Net+** -- *Jia, Xi, et al.* `LessNet <https://github.com/xi-jia/LessNet>`__, `Fourier-Net <https://github.com/xi-jia/Fourier-Net>`__

About the author
----------------

MIR is developed by `Junyu Chen <https://junyuchen245.github.io>`_ (Johns Hopkins University).

.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   installation
   concepts
   quick_start
   quick_start_brain_registration
   quick_start_ixi_benchmarking
   brain_template_example

.. toctree::
   :maxdepth: 2
   :caption: Models
   :hidden:

   models/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials and reference
   :hidden:

   tutorials
   api
