HyperMorph
==========

Overview
--------

HyperMorph is the hypernetwork-conditioned VoxelMorph: the regularization weight is an input
of the network, so the smoothness of the deformation is chosen at inference time. MIR's
``HyperVxmDense`` is the convolutional counterpart of :doc:`hypertransmorph`.

Usage
-----

.. code-block:: python

   from MIR.models import HyperVxmDense
   import MIR.models.configs_VoxelMorph as configs_VoxelMorph

   config = configs_VoxelMorph.get_VXM_default_config()
   config.img_size = fix.shape[2:]
   model = HyperVxmDense(config).cuda()
   flow = model((mov, fix), torch.tensor([[0.3]], device='cuda'))   # regularization weight

Key modules
-----------

- ``MIR.models.HyperMorph``, ``MIR.models.configs_VoxelMorph``

Related tutorials
-----------------

- `IXI HyperMorph <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_HyperMorph>`_

References
----------

- Hoopes, Hoffmann, Fischl, Guttag, Dalca. HyperMorph: amortized hyperparameter learning for
  image registration. *IPMI* 2021.
