MIR Documentation
=================

.. image:: https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square
   :alt: License: Apache 2.0
   :target: https://opensource.org/license/apache-2-0

Overview
--------

MIR is a research toolkit for medical image registration, providing model implementations,
training/inference scripts, and supporting utilities. It also includes curated wrappers for
selected external packages to support **TransMorph‑style registration workflows**.

Documentation: https://junyuchen.me/MIR

Repository: https://github.com/junyuchen245/MIR

Installation
------------

**Prerequisite:** Install a CUDA-enabled PyTorch build before installing MIR.

Editable install (recommended)::

   git clone https://github.com/junyuchen245/MIR.git
   cd MIR
   python3.8 -m pip install -U pip
   python3.8 -m pip install -e .

Platform notes
--------------

- deedsBCV integration is **Linux-only**. Windows users can still use MIR, but deedsBCV functionality is unavailable.

Getting started
---------------

Quick start: IXI benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Benchmark multiple registration models on the IXI dataset using the scripts in
`tutorials/IXI_benchmarking <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_benchmarking>`_.

.. code-block:: bash

   cd tutorials/IXI_benchmarking
   python3.8 -u train_TransMorph.py
   python3.8 -u train_TransMorphTVF.py
   python3.8 -u train_SITReg.py
   python3.8 -u train_SITReg_SPR.py

Example: Brain template construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build a population template from LUMIR24 using TransMorphTVF, VFA, or ConvexAdam.
See :doc:`brain_template_example` for the full walkthrough.

Included Projects
-----------------

1. **Pretraining Deformable Image Registration Networks with Random Images**
   *Chen, Junyu, et al.* ***MIDL Short Papers***, *2025.*
   `Paper <https://openreview.net/forum?id=NJANlZzxfi#discussion>`_ | `Repo <https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs>`_

2. **Correlation Ratio for Unsupervised Learning of Multi-modal Deformable Registration**
   *Chen, Xiaojian, et al.* ***SPIE Medical Imaging: Image Processing***, *2025.*
   `Paper <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13406/1340632/Correlation-ratio-for-unsupervised-learning-of-multi-modal-deformable-registration/10.1117/12.3047699.full>`_ | `Repo <https://github.com/junyuchen245/Correlation_Ratio>`_

3. **Unsupervised Learning of Spatially Varying Regularization for Diffeomorphic Image Registration**
   *Chen, Junyu, et al.* ***Medical Image Analysis***, *2025.*
   `Paper <https://arxiv.org/abs/2412.17982>`_ | `Repo <https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg>`_

4. **Vector Field Attention for Deformable Image Registration**
   *Liu, Yihao, et al.* ***Journal of Medical Imaging***, *2024.*
   `Paper <https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-11/issue-6/064001/Vector-field-attention-for-deformable-image-registration/10.1117/1.JMI.11.6.064001.full>`_ | `Repo <https://github.com/yihao6/vfa/>`_

5. **Unsupervised Learning of Multi-modal Affine Registration for PET/CT**
   *Chen, Junyu, et al.* ***IEEE NSS/MIC/RTSD***, *2024.*
   `Paper <https://arxiv.org/pdf/2409.13863>`_ | `Repo <https://github.com/junyuchen245/Correlation_Ratio>`_

6. **On Finite Difference Jacobian Computation in Deformable Image Registration**
   *Liu, Yihao, et al.* ***International Journal of Computer Vision***, *2024.*
   `Paper <https://link.springer.com/article/10.1007/s11263-024-02047-1>`_ | `Repo <https://github.com/yihao6/digital_diffeomorphism>`_

7. **Deformable Cross-Attention Transformer for Medical Image Registration**
   *Chen, Junyu, et al.* ***MLMI Workshop***, *MICCAI 2023.*
   `Paper <https://arxiv.org/abs/2303.06179>`_ | `Repo <https://github.com/junyuchen245/TransMorph_DCA>`_

8. **Unsupervised Learning of Diffeomorphic Image Registration via TransMorph**
   *Chen, Junyu, et al.* ***WBIR***, *2022.*
   `Paper <https://link.springer.com/chapter/10.1007/978-3-031-11203-4_11>`_ | `Repo <https://github.com/junyuchen245/TransMorph_TVF>`_

9. **TransMorph: Transformer for Unsupervised Medical Image Registration**
   *Chen, Junyu, et al.* ***Medical Image Analysis***, *2022.*
   `Paper <https://arxiv.org/abs/2111.10480>`_ | `Repo <https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration>`_

External Wrappers Included
--------------------------

1. **Intensity Normalization Toolkit**
   *Reinhold, Jacob C., et al.* ***SPIE MI***, *2019.*
   `Paper <https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/>`_ | `Repo <https://github.com/jcreinhold/intensity-normalization>`_

2. **ConvexAdam: Dual-Optimization-Based 3D Registration**
   *Siebert, Hanna, et al.* ***IEEE TMI***, *2024.*
   `Paper <https://ieeexplore.ieee.org/abstract/document/10681158>`_ | `Repo <https://github.com/multimodallearning/convexAdam>`_

3. **MultiMorph: On-demand Atlas Construction**
   *Abulnaga, S. Mazdak, et al.* ***CVPR***, *2025.*
   `Paper <https://openaccess.thecvf.com/content/CVPR2025/html/Abulnaga_MultiMorph_On-demand_Atlas_Construction_CVPR_2025_paper.html>`_ | `Repo <https://github.com/mabulnaga/multimorph>`_

4. **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration.**
   *Honkamaa, Joel, and Pekka Marttinen.* ***MELBA***, *2024.*
   `Paper <https://www.melba-journal.org/papers/2024:026.html>`_ | `Repo <https://github.com/honkamj/SITReg>`_

5. **deedsBCV**
   `Repo <https://github.com/mattiaspaul/deedsBCV>`_

About Me
--------

`About Me <https://junyuchen245.github.io>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick_start
   quick_start_brain_registration
   quick_start_ixi_benchmarking
   brain_template_example
   models/index
   api
