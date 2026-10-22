Atlas construction: MultiMorph and TemplateCreation
===================================================

Overview
--------

MIR contains two routes to population templates:

- ``MIR.models.atlas_builder.TemplateCreation`` wraps any pairwise registration network
  (TransMorph, VFA, SITReg) in the "register everyone to the current template, average, repeat"
  loop with log-domain or displacement averaging (``MeanStream``), used by the brain and PET/CT
  atlas tutorials.
- ``MIR.models.MultiMorph`` (``GroupNet3D``) is the group-wise, on-demand atlas construction
  network of Abulnaga et al., with its epoch-batch samplers (``make_epoch_batches``,
  ``ListBatchSampler``) and ``Warp3d``.

Key modules
-----------

- ``MIR.models.atlas_builder`` (``TemplateCreation``, ``MeanStream``)
- ``MIR.models.MultiMorph`` (``GroupNet3D``, ``Warp3d``)
- ``MIR.models.affine_reg`` (``AffineReg3D``, ``PreAffineToTemplate``) for pre-alignment

Related tutorials
-----------------

- `Brain template construction <https://github.com/junyuchen245/MIR/tree/main/tutorials/build_brain_template>`_

References
----------

- Abulnaga et al. MultiMorph: on-demand atlas construction. *CVPR* 2025.
  `Paper <https://openaccess.thecvf.com/content/CVPR2025/html/Abulnaga_MultiMorph_On-demand_Atlas_Construction_CVPR_2025_paper.html>`__ |
  `Repo <https://github.com/mabulnaga/multimorph>`__
