Quick start
===========

These guides use pretrained weights that are downloaded on first run. All you need is a CUDA-enabled
PyTorch and an editable install of MIR (:doc:`installation`).

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`zap;1.2em` Register a brain MRI
      :link: brain_registration
      :link-type: doc

      Affine pre-alignment with ``AffineReg3D``, then deformable registration with VFA,
      TransMorph-TVF or ConvexAdam-MIND, and flow composition.

   .. grid-item-card:: :octicon:`graph;1.2em` Benchmark on IXI
      :link: ixi_benchmarking
      :link-type: doc

      Evaluate seven registration models from one notebook and compare Dice and folding.

   .. grid-item-card:: :octicon:`people;1.2em` Build a template
      :link: brain_template
      :link-type: doc

      Construct a population brain template from LUMIR24 by iterated registration and averaging.

The five-line pattern
---------------------

Every registration model in MIR follows the same pattern. Once you know it, every tutorial reads the
same way.

.. code-block:: python

   from MIR.models import TransMorphTVF, SpatialTransformer
   import MIR.models.configs_TransMorph as configs_TransMorph

   config = configs_TransMorph.get_3DTransMorph3Lvl_config()   # 1. pick a configuration
   config.img_size = (80, 112, 96)                            # 2. tell it the grid it will see
   model = TransMorphTVF(config, time_steps=7).cuda()         # 3. build the model
   flow = model((moving, fixed))                              # 4. predict a displacement in voxels
   warped = SpatialTransformer(config.img_size).cuda()(moving, flow)   # 5. warp anything with it

The :doc:`../user_guide/conventions` page explains tensor shapes, displacement units, resolution
changes and flow composition in detail.

Where to go next
----------------

- Train your own model: :doc:`../user_guide/training` walks through a complete training loop, and
  :doc:`../tutorials` lists the ready-made training scripts per dataset.
- Pick a model: :doc:`../models/index` compares the architectures and their variants.
- Look up a function: :doc:`../api`.
