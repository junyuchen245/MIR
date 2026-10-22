User guide
==========

The user guide explains the ideas shared by every model in MIR. Read :doc:`conventions` first; the other
pages can be read in any order.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`plug;1.2em` Conventions
      :link: conventions
      :link-type: doc

      Tensor shapes, displacement units, warping, composing flows, changing resolution, and the calling
      convention of each model type.

   .. grid-item-card:: :octicon:`git-merge;1.2em` Transformation models
      :link: transformations
      :link-type: doc

      How a network output becomes a deformation: displacement, SVF, time-varying fields, symmetric
      mappings, LDDMM. What each guarantees.

   .. grid-item-card:: :octicon:`meter;1.2em` Regularization
      :link: regularization
      :link-type: doc

      Global smoothness penalties, spatially varying regularization (SPR) and hypernetwork-conditioned
      regularization strength.

   .. grid-item-card:: :octicon:`tools;1.2em` Losses and metrics
      :link: losses_metrics
      :link-type: doc

      Mono- and multi-modal similarity, label supervision, regularizers, Dice and Jacobian-based folding
      metrics.

   .. grid-item-card:: :octicon:`flame;1.2em` Training a model
      :link: training
      :link-type: doc

      A complete, annotated training loop with the practices used across the tutorials.

   .. grid-item-card:: :octicon:`download;1.2em` Pretrained weights
      :link: pretrained_weights
      :link-type: doc

      Every published checkpoint, what it was trained on, and how to load it.
