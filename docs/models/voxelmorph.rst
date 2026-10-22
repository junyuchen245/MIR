VoxelMorph
==========

Overview
--------

``VxmDense`` is the U-Net registration baseline of Balakrishnan et al., included for
comparison and as a light-weight backbone (0.3 M parameters with the default feature
configuration). It predicts a displacement field; wrap it in ``LDDMMWrapper`` for a
diffeomorphic variant (:doc:`lddmm`).

Usage
-----

.. code-block:: python

   from MIR.models import VxmDense
   import MIR.models.configs_VoxelMorph as configs_VoxelMorph

   config = configs_VoxelMorph.get_VXM_default_config()     # also get_VXM_1_config, get_VXM_BJ_config
   config.img_size = fix.shape[2:]
   model = VxmDense(config, gen_output=False).cuda()
   flow = model((mov, fix))

Key modules
-----------

- ``MIR.models.VoxelMorph``, ``MIR.models.configs_VoxelMorph``

Related tutorials
-----------------

- `LUMIR25 VoxelMorph <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_VoxelMorph>`_

References
----------

- Balakrishnan, Zhao, Sabuncu, Guttag, Dalca. VoxelMorph: a learning framework for deformable
  medical image registration. *IEEE TMI* 2019.
