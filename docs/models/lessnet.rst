LessNet
=======

Overview
--------

LessNet is a light-weight convolutional registration network (0.1 M parameters) ported from
`xi-jia/LessNet <https://github.com/xi-jia/LessNet>`_. MIR keeps the backbone close to the
reference implementation and converts its normalized-grid prediction to a voxel displacement,
so the output plugs into every MIR utility.

Variants
--------

- ``LessNet``: base model; ``SVF=True`` integrates the field with ``VecInt``.
- ``LessNetSPR``: adds a spatially varying regularization head and follows the
  ``TransMorphTVFSPR`` return convention.
- ``LessNetUNet``: the bare backbone.

Default configuration
---------------------

``get_LessNet_default_config`` follows the upstream reproduction command:
``start_channel = 8``, ``NCC_mok`` similarity (``using_l2 = 2``), ``smth_labda = 5.0``,
``lr = 1e-4``. ``get_LessNet_SVF_config`` and ``get_LessNet_SPR_config`` enable the
respective variants.

Usage
-----

.. code-block:: python

   from MIR.models import LessNet, LessNetSPR
   from MIR.models.configs_LessNet import get_LessNet_default_config, get_LessNet_SVF_config, get_LessNet_SPR_config

   model = LessNet(get_LessNet_default_config())
   flow = model((mov, fix))

   svf_model = LessNet(get_LessNet_SVF_config(), gen_output=True)
   warped, flow = svf_model((mov, fix))

   spr_model = LessNetSPR(get_LessNet_SPR_config(), SVF=True)
   pos_flow, neg_flow, wts = spr_model((mov, fix))

Key modules
-----------

- ``MIR.models.LessNet``, ``MIR.models.configs_LessNet``

Related tutorials
-----------------

- `LUMIR25 LessNet <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_LessNet>`_

References
----------

- Jia, Bartlett, Zhang, Lu, Qiu, Duan. U-Net vs transformer: is U-Net outdated in medical image
  registration? *MLMI* 2022 (LessNet).
