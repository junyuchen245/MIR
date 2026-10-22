Tutorials
=========

Every tutorial is a folder under
`tutorials/ <https://github.com/junyuchen245/MIR/tree/main/tutorials>`_ with runnable
training, inference and evaluation scripts and, where relevant, a README with dataset paths
and outputs. Scripts are run from inside their folder; outputs go to ``experiments/``,
``logs/`` and ``results/`` next to them. Dataset paths are set at the top of each script.

Brain MRI, Learn2Reg LUMIR (2024 / 2025)
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Folder
     - What it does
   * - ``lumir24_TransMorph``, ``lumir25_TransMorph``
     - TransMorph-TVF training with random multi-contrast augmentation, evaluation on the
       LUMIR test splits (Dice, TRE, NDV).
   * - ``lumir24_VFA``, ``lumir25_VFA``
     - VFA training and inference, HyperVFA-SPR, SynthSR preprocessing, Docker example.
   * - ``lumir25_SITReg``, ``lumir25_VoxelMorph``, ``lumir25_LessNet``, ``lumir25_FourierNet``
     - The same training / evaluation protocol for the other backbones.
   * - ``lumir24_ConvexAdam``, ``lumir25_ConvexAdam``
     - ConvexAdam baselines.
   * - ``lumir25_TM_LDDMM``
     - **Reference LDDMM tutorial**: TransMorph-LDDMM and the model-agnostic wrapper for any
       backbone, geodesic shooting or relaxation, stationary or spatially varying kernels,
       SPR integration; calibration tables.
   * - ``lumir25_TM_SPR_LDDMM``, ``lumir25_TM_OMT_LDDMM``, ``lumir25_SITReg_LDDMM``,
       ``lumir25_SITReg_SPR_LDDMM``
     - Thin front-ends over the reference LDDMM tutorial with fixed configurations.
   * - ``LUMIR24_validation_benchmarking``
     - Validation-phase evaluation utilities.
   * - ``build_brain_template``
     - Population template construction with TransMorph-TVF, VFA or ConvexAdam
       (:doc:`brain_template_example`).
   * - ``brain_registration``, ``brain_MRI_preprocessing``
     - Affine + deformable registration with pretrained weights
       (:doc:`quick_start_brain_registration`); bias correction, normalization, masking.

Brain MRI, IXI
--------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Folder
     - What it does
   * - ``IXI_benchmarking``
     - Notebook benchmarking TransMorph, TransMorph-TVF, VFA, VFA-SPR, SITReg, SITReg-SPR
       and ConvexAdam-MIND on IXI (:doc:`quick_start_ixi_benchmarking`).
   * - ``IXI_TransMorphSPR``, ``IXI_SITReg``, ``IXI_HyperTransMorph``, ``IXI_HyperMorph``
     - Training with spatially varying regularization or hypernetwork conditioning.
   * - ``IXI_deedsBCV``
     - deedsBCV validation (Linux only).
   * - ``pretraining_registration_DNNs``
     - Pretraining registration networks on random images.

PET/CT
------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Folder
     - What it does
   * - ``autoPET_TransMorphSPR``, ``autoPET_ConvexAdamSPR``, ``autoPET_FireANTs_SPR``
     - PET/CT registration with spatially varying regularization, learned and classical.
   * - ``PSMA_atlas_building``, ``Wholebody_PETCT_Atlas``, ``BICMAC_PETCT_Atlas``
     - Whole-body PET/CT atlas construction, atlas-to-subject registration (TransMorph-SPR,
       VFA-SPR, SITReg-SPR, TransMorph-SPR + LDDMM), instance optimization and the
       population analyses built on them.
   * - ``PSMAReg``, ``HECKTOR25``, ``JHU_FDG``, ``IPEN_registration``
     - Dataset-specific registration pipelines.

Other
-----

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Folder
     - What it does
   * - ``VFA_image_seg``, ``VFA_image_synthesis``
     - VFA features for segmentation and synthesis heads.
   * - ``PSMA_atlas_building_MultiMorph``
     - On-demand atlas construction with MultiMorph.

Pretrained weights and dataset files used by the tutorials are listed in
`tutorials/README.md <https://github.com/junyuchen245/MIR/blob/main/tutorials/README.md>`_.
