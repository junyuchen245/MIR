Fourier-Net and Fourier-Net+
============================

Overview
--------

Fourier-Net predicts a band-limited deformation in the Fourier domain and recovers the
full-resolution displacement by zero-padding, which makes it very light and fast.
Fourier-Net+ additionally works on low-frequency inputs. Both are ported from
`xi-jia/Fourier-Net <https://github.com/xi-jia/Fourier-Net>`_ with the output converted to
MIR voxel displacements.

Variants
--------

- ``FourierNet``, ``FourierNetPlus``: base models; ``SVF=True`` integrates the field.
- ``FourierNetSPR``, ``FourierNetPlusSPR``: spatially varying regularization heads,
  ``TransMorphTVFSPR`` return convention.
- ``FourierNetBackbone``, ``FourierNetPlusBackbone``: bare backbones.

Default configuration
---------------------

``FourierNet``: ``start_channel = 8``, ``NCC_mok``, ``smth_labda = 1.0``, ``lr = 1e-4``.
``FourierNetPlus``: ``start_channel = 16``, ``low_img_size = (80, 96, 112)``,
``smth_labda = 5.0``. Set ``config.img_size`` and, for Fourier-Net+, ``config.low_img_size``
to half of it.

Usage
-----

.. code-block:: python

   from MIR.models import FourierNet, FourierNetPlus, FourierNetPlusSPR
   from MIR.models.configs_FourierNet import (get_FourierNet_default_config,
                                              get_FourierNetPlus_default_config, get_FourierNetPlus_SPR_config)

   model = FourierNet(get_FourierNet_default_config())
   flow = model((mov, fix))

   plus = FourierNetPlus(get_FourierNetPlus_default_config(), gen_output=True)
   warped, flow = plus((mov, fix))

   spr = FourierNetPlusSPR(get_FourierNetPlus_SPR_config())
   pos_flow, neg_flow, wts = spr((mov, fix))

Key modules
-----------

- ``MIR.models.FourierNet``, ``MIR.models.configs_FourierNet``

Related tutorials
-----------------

- `LUMIR25 Fourier-Net+ <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_FourierNet>`_

References
----------

- Jia, Bartlett, Zhang, Lu, Qiu, Duan. Fourier-Net: fast image registration with band-limited
  deformation. *AAAI* 2023.
- Jia, Lu, Qiu, Duan. Fourier-Net+: leveraging band-limited representation for efficient 3D
  medical image registration. 2023.
