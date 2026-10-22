API reference
=============

The reference is generated from the docstrings and organized by subpackage. Most user code needs only
``MIR.models`` (networks and ``SpatialTransformer``), ``MIR.image_similarity`` (losses),
``MIR.deformation_regularizer`` (regularizers) and ``MIR.accuracy_measures`` (metrics).

Models
------

Registration networks, transformation models (``VecInt``, LDDMM), spatial transformers, configurations,
backbones and heads, classical methods.

.. autosummary::
   :toctree: api
   :recursive:

   MIR.models

Losses
------

Image similarity and label overlap losses.

.. autosummary::
   :toctree: api
   :recursive:

   MIR.image_similarity

Regularizers
------------

Global, spatially varying and inverse-consistency regularizers, priors.

.. autosummary::
   :toctree: api
   :recursive:

   MIR.deformation_regularizer

Metrics
-------

Dice and Jacobian-determinant (digital diffeomorphism) evaluation.

.. autosummary::
   :toctree: api
   :recursive:

   MIR.accuracy_measures

Utilities and data
------------------

Training utilities, augmentation, intensity normalization, synthetic images for pretraining, statistics,
Learn2Reg I/O, label references and the pretrained-weight registry.

.. autosummary::
   :toctree: api
   :recursive:

   MIR.utils
   MIR.data_augmentation
   MIR.intensity_normalization
   MIR.random_image_generation
   MIR.statistical_analysis
   MIR.l2r_io
   MIR.label_reference
   MIR.pretrained_wts
