SITReg
======

Overview
--------

SITReg (Honkamaa and Marttinen) is a multi-resolution architecture that is **symmetric**,
**inverse consistent** and **topology preserving** by construction: deformations are built from
per-level updates passed through deformation inversion layers, so the forward and inverse maps
are exact inverses of each other. MIR vendors the reference implementation under
``MIR.models.SITReg`` and uses it through ``TemplateCreation`` in the atlas-registration
tutorials.

Usage
-----

.. code-block:: python

   from MIR.models import SITReg, EncoderFeatureExtractor
   from MIR.models.SITReg import ReLUFactory, GroupNormalizerFactory
   from MIR.models.SITReg.composable_mapping import DataFormat

   features = EncoderFeatureExtractor(n_input_channels=1, activation_factory=ReLUFactory(),
                                      n_features_per_resolution=[12, 16, 32, 64, 128, 128],
                                      n_convolutions_per_resolution=[2] * 6, input_shape=shape,
                                      normalizer_factory=GroupNormalizerFactory(2))
   model = SITReg(feature_extractor=features, ...)              # see the LUMIR25 tutorial for the full call
   mapping = model(mov, fix, mappings_for_levels=((0, False),))[0]
   flow = mapping.forward_mapping.sample(data_format=DataFormat.voxel_displacements()).generate_values()

The LDDMM wrapper supports SITReg as a momentum predictor through ``adapter='sitreg'``
(:doc:`lddmm`). Spatially varying regularization for SITReg is provided by a separate SPR head
in the tutorials (``SPRHead`` in ``MIR.models.lddmm`` is the model-agnostic equivalent).

Pretrained weights
------------------

``ModelWeights['SITReg-IXI-Atlas2Subject']``, ``ModelWeights['SITReg-SPR-IXI-Atlas2Subject']``.

Key modules
-----------

- ``MIR.models.SITReg`` (``SITReg``, ``MappingPair``, ``EncoderFeatureExtractor``,
  ``ReLUFactory``, ``GroupNormalizerFactory``)
- ``MIR.models.atlas_builder.TemplateCreation`` (``use_sitreg=True``)

Related tutorials
-----------------

- `IXI SITReg <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_SITReg>`_
- `LUMIR25 SITReg <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_SITReg>`_
- `LUMIR25 SITReg + LDDMM <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_SITReg_LDDMM>`_
- `Whole-body PET/CT atlas (SITReg-SPR) <https://github.com/junyuchen245/MIR/tree/main/tutorials/Wholebody_PETCT_Atlas>`_

References
----------

- Honkamaa, Marttinen. SITReg: multi-resolution architecture for symmetric, inverse consistent,
  and topology preserving image registration. *MELBA* 2024.
  `Paper <https://www.melba-journal.org/papers/2024:026.html>`__ |
  `Repo <https://github.com/honkamj/SITReg>`__
