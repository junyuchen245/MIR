ConvexAdam and FireANTs
=======================

Overview
--------

Two optimization-based registration methods are wrapped behind the MIR interface for
baselines and for hybrid pipelines:

- **ConvexAdam** (Siebert et al.): discrete convex optimization on hand-crafted MIND features
  followed by Adam-based instance optimization. MIR provides the MIND variant, SVF and
  spatially-varying-regularization variants, and versions that optimize on learned features
  (nnU-Net segmentation features, VFA features).
- **FireANTs** (Jena et al.): GPU implementations of greedy and SyN diffeomorphic
  registration, with SPR variants, under ``MIR.models.fireants``.

Usage
-----

.. code-block:: python

   from MIR.models import convex_adam_MIND, convex_adam_MIND_SPR, get_ConvexAdam_MIND_brain_default_config

   config = get_ConvexAdam_MIND_brain_default_config()
   flow = convex_adam_MIND(mov, fix, config)                 # displacement in voxels

Other entry points: ``convex_adam_MIND_SVF``, ``convex_adam_features`` /
``convex_adam_features_svf`` (precomputed features), ``convex_adam_seg_features``,
``convex_adam_vfa``. FireANTs: ``MIR.models.fireants.registration.greedy`` / ``syn`` and their
``_SPR`` counterparts.

Key modules
-----------

- ``MIR.models.convexAdam`` (``convex_adam_MIND``, ``configs_ConvexAdam_MIND``, ``convex_adam_VFA``, ...)
- ``MIR.models.fireants``

Related tutorials
-----------------

- `LUMIR25 ConvexAdam <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_ConvexAdam>`_
- `LUMIR24 ConvexAdam <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir24_ConvexAdam>`_
- `autoPET ConvexAdam-SPR <https://github.com/junyuchen245/MIR/tree/main/tutorials/autoPET_ConvexAdamSPR>`_
- `autoPET FireANTs-SPR <https://github.com/junyuchen245/MIR/tree/main/tutorials/autoPET_FireANTs_SPR>`_
- `Brain registration (ConvexAdam-MIND) <https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_registration>`_

References
----------

- Siebert, Grossbroehmer, Hansen, Heinrich. ConvexAdam: self-configuring dual-optimization-based
  3D multitask medical image registration. *IEEE TMI* 2024.
  `Paper <https://ieeexplore.ieee.org/abstract/document/10681158>`__ |
  `Repo <https://github.com/multimodallearning/convexAdam>`__
- Jena et al. FireANTs: adaptive Riemannian optimization for multi-scale diffeomorphic
  registration. `Repo <https://github.com/rohitrango/fireants>`__
