HyperTransMorph
===============

Overview
--------

HyperTransMorph conditions TransMorph-TVF on the regularization weight through a
hypernetwork: the weight is an input at training and inference time, so a single network
spans a range of regularization strengths and the trade-off between accuracy and smoothness
can be chosen per case without retraining (the HyperMorph idea applied to TransMorph).

Variants
--------

- ``HyperTransMorphTVF``: hypernetwork-conditioned TransMorph-TVF.
- ``HyperTransMorphTVFSPR``: the same with the spatially varying regularization head.
- ``HyperVFA``, ``HyperVFASPR``: the VFA counterparts (:doc:`vfa`).

Usage
-----

.. code-block:: python

   from MIR.models import HyperTransMorphTVF
   model = HyperTransMorphTVF(config, time_steps=7).cuda()
   flow = model((mov, fix), hyper_weight)        # hyper_weight: regularization weight tensor

See the tutorial for the training loop that samples the weight per iteration.

Key modules
-----------

- ``MIR.models.HyperTransMorph``, ``MIR.models.HyperVFA``

Related tutorials
-----------------

- `IXI HyperTransMorph <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_HyperTransMorph>`_
- `LUMIR25 HyperVFA-SPR inference <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_VFA>`_

References
----------

- Hoopes, Hoffmann, Fischl, Guttag, Dalca. HyperMorph: amortized hyperparameter learning for
  image registration. *IPMI* 2021.
- Chen et al. TransMorph. *Medical Image Analysis* 2022.
