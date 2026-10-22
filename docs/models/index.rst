Models
======

MIR implements its own registration networks and wraps third-party methods behind the same interface
(:doc:`../user_guide/conventions`). Each model page lists the variants, a usage snippet, pretrained
weights, the tutorials that train or evaluate it, and the paper to cite.

Registration networks
---------------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: TransMorph
      :link: transmorph
      :link-type: doc

      Swin-Transformer encoder with a convolutional decoder. The reference architecture of MIR.

      :bdg-primary-line:`SVF` :bdg-primary-line:`TVF` :bdg-success-line:`SPR` :bdg-secondary-line:`pretrained`

   .. grid-item-card:: VFA
      :link: vfa
      :link-type: doc

      Vector field attention: correspondences read directly from feature similarity. Light and
      contrast-robust.

      :bdg-primary-line:`SVF` :bdg-success-line:`SPR` :bdg-secondary-line:`pretrained`

   .. grid-item-card:: SITReg
      :link: sitreg
      :link-type: doc

      Multi-resolution, symmetric, inverse-consistent and topology-preserving by construction.

      :bdg-primary-line:`inverse consistent` :bdg-secondary-line:`pretrained`

   .. grid-item-card:: VoxelMorph
      :link: voxelmorph
      :link-type: doc

      The U-Net baseline of learning-based registration.

      :bdg-primary-line:`displacement`

   .. grid-item-card:: LessNet
      :link: lessnet
      :link-type: doc

      A very light convolutional network (0.1 M parameters).

      :bdg-primary-line:`SVF` :bdg-success-line:`SPR`

   .. grid-item-card:: Fourier-Net and Fourier-Net+
      :link: fouriernet
      :link-type: doc

      Band-limited deformations predicted in the Fourier domain; fast and compact.

      :bdg-primary-line:`SVF` :bdg-success-line:`SPR`

Extensions
----------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: HyperTransMorph and HyperVFA
      :link: hypertransmorph
      :link-type: doc

      Hypernetwork-conditioned regularization strength, chosen at inference time.

   .. grid-item-card:: HyperMorph
      :link: hypermorph
      :link-type: doc

      The hypernetwork-conditioned VoxelMorph.

   .. grid-item-card:: LDDMM
      :link: lddmm
      :link-type: doc

      Time-varying velocity fields with geodesic shooting, for TransMorph or any other backbone.

Optimization-based methods
--------------------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: ConvexAdam and FireANTs
      :link: convexadam
      :link-type: doc

      Discrete convex optimization with Adam refinement, and GPU greedy / SyN registration.

      :bdg-success-line:`SPR` :bdg-info-line:`no training`

   .. grid-item-card:: deedsBCV
      :link: deedsbcv
      :link-type: doc

      MRF-based discrete registration with self-similarity descriptors (Linux).

      :bdg-info-line:`no training`

Atlases and building blocks
---------------------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Atlas construction
      :link: multimorph
      :link-type: doc

      Template building with any registration network, and MultiMorph group-wise atlases.

   .. grid-item-card:: Backbones and heads
      :link: backbones_and_heads
      :link-type: doc

      Swin encoders, decoders, segmentation, synthesis and self-supervised heads.

Comparison
----------

Parameter counts are for the configurations used in the LUMIR25 tutorials (160 x 224 x 192 brain MRI;
TransMorph runs at half resolution).

.. list-table::
   :header-rows: 1
   :widths: 18 30 12 26 14

   * - Model
     - Architecture
     - Parameters
     - Transformation models
     - Weights
   * - TransMorph-TVF
     - Swin Transformer + conv decoder
     - 17.1 M
     - displacement, SVF, TVF
     - LUMIR, IXI, autoPET
   * - VFA
     - conv encoder + vector field attention
     - 2.0 M
     - displacement, SVF
     - LUMIR, IXI
   * - SITReg
     - multi-resolution symmetric
     - 12.9 M
     - inverse consistent
     - IXI
   * - VoxelMorph
     - U-Net
     - 0.3 M
     - displacement
     - --
   * - LessNet
     - light conv U-Net
     - 0.1 M
     - displacement, SVF
     - --
   * - Fourier-Net+
     - band-limited conv
     - 0.9 M
     - displacement, SVF
     - --
   * - ConvexAdam, deedsBCV, FireANTs
     - optimization
     - --
     - discrete, SVF, greedy / SyN
     - not needed

Every learned model above also accepts the LDDMM transformation model through
:doc:`LDDMMWrapper <lddmm>`.
