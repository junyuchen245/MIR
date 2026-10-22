Tutorials
=========

Every tutorial is a self-contained folder under
`tutorials/ <https://github.com/junyuchen245/MIR/tree/main/tutorials>`__ with runnable training,
inference and evaluation scripts. Set the dataset paths at the top of a script, run it from inside its
folder, and find checkpoints (``experiments/``), logs (``logs/``) and result tables (``results/``) next to
it.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`zap;1.2em` New to MIR?
      :link: getting_started/brain_registration
      :link-type: doc

      Start with ``brain_registration``: affine + deformable registration with pretrained weights.

   .. grid-item-card:: :octicon:`flame;1.2em` Training your own model?
      :link: user_guide/training
      :link-type: doc

      ``lumir25_TransMorph`` is the most complete training template; the training guide explains it.

Brain MRI: Learn2Reg LUMIR
--------------------------

Inter-subject brain MRI registration on the LUMIR challenge data (Learn2Reg 2024 and 2025). Every
LUMIR25 tutorial uses the same protocol, so results are directly comparable: training with random
multi-contrast augmentation, then evaluation on the LUMIR test splits and zero-shot datasets (Dice,
landmark TRE, non-diffeomorphic volume).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Folder
     - Contents
   * - ``lumir25_TransMorph``, ``lumir24_TransMorph``
     - TransMorph-TVF training and evaluation.
   * - ``lumir25_VFA``, ``lumir24_VFA``
     - VFA training and inference, HyperVFA-SPR, SynthSR preprocessing, a Docker submission example.
   * - ``lumir25_VoxelMorph``, ``lumir25_LessNet``, ``lumir25_FourierNet``
     - The same protocol for the convolutional baselines.
   * - ``lumir25_ConvexAdam``, ``lumir24_ConvexAdam``
     - ConvexAdam baselines (no training).
   * - ``lumir25_TM_LDDMM`` and ``lumir25_*_LDDMM``
     - LDDMM training and evaluation for TransMorph and other backbones (:doc:`models/lddmm`).

Brain MRI: IXI and general use
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Folder
     - Contents
   * - ``brain_registration``
     - Affine + deformable registration with pretrained weights (:doc:`getting_started/brain_registration`).
   * - ``IXI_benchmarking``
     - Benchmark seven models on IXI from a notebook (:doc:`getting_started/ixi_benchmarking`).
   * - ``IXI_TransMorphSPR``, ``IXI_SITReg``
     - Atlas-to-subject training with spatially varying regularization, and SITReg.
   * - ``IXI_HyperTransMorph``, ``IXI_HyperMorph``
     - Hypernetwork-conditioned regularization.
   * - ``IXI_deedsBCV``
     - deedsBCV validation (Linux only).
   * - ``build_brain_template``
     - Population template construction (:doc:`getting_started/brain_template`).
   * - ``brain_MRI_preprocessing``
     - Bias-field correction, intensity normalization, skull stripping.
   * - ``pretraining_registration_DNNs``
     - Pretraining registration networks on random synthetic images.

Whole-body PET/CT
-----------------

Atlas-to-subject registration of FDG PET/CT from the autoPET challenge data, with spatially varying
regularization learned or optimized.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Folder
     - Contents
   * - ``autoPET_TransMorphSPR``
     - TransMorph-TVF-SPR training and inference (the MedIA 2025 SPR models).
   * - ``autoPET_ConvexAdamSPR``, ``autoPET_FireANTs_SPR``
     - Optimization-based registration with spatially varying regularization.
   * - ``AutoPET``
     - Dataset metadata summaries.

Feature learning
----------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Folder
     - Contents
   * - ``VFA_image_seg``, ``VFA_image_synthesis``
     - Segmentation and synthesis heads on pretrained VFA features.

Pretrained weights and dataset files used by the tutorials are listed in
:doc:`user_guide/pretrained_weights`.
