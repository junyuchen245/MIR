Models
======

Every model family in MIR has a page with the same layout: what the model is and when to use
it, the variants that are included, a minimal usage snippet, the modules to import, pretrained
weights where they exist, the tutorials that exercise it, and the reference to cite. All
models share the calling convention described in :doc:`../concepts`.

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Family
     - Summary
     - Variants
   * - :doc:`transmorph`
     - Swin-Transformer registration network; the reference architecture of the toolkit.
     - TransMorph, TransMorph-TVF, TransMorph-TVF-SPR, TransMorph-DCA, TransMorph-Affine
   * - :doc:`lddmm`
     - Time-varying velocity fields: geodesic shooting / relaxation, spatially varying
       kernels, SPR in the metric; model-agnostic wrapper.
     - TransMorphLDDMM, LDDMMWrapper
   * - :doc:`vfa`
     - Vector field attention: correspondence from feature similarity instead of a decoder.
     - VFA, VFA-SPR, TransVFA, HyperVFA(-SPR)
   * - :doc:`hypertransmorph`, :doc:`hypermorph`
     - Hypernetwork-conditioned regularization.
     - HyperTransMorphTVF(-SPR), HyperVxmDense
   * - :doc:`voxelmorph`
     - U-Net baseline.
     - VxmDense
   * - :doc:`lessnet`, :doc:`fouriernet`
     - Light-weight CNN baselines ported from their reference implementations.
     - LessNet(-SPR), Fourier-Net(-SPR), Fourier-Net+(-SPR)
   * - :doc:`sitreg`
     - Symmetric, inverse-consistent, topology-preserving multi-resolution registration.
     - SITReg (+ SPR head in tutorials)
   * - :doc:`convexadam`, :doc:`deedsbcv`
     - Classical and optimization-based baselines behind the MIR interface.
     - ConvexAdam-MIND(-SVF/-SPR), ConvexAdam with learned features, deedsBCV, FireANTs
   * - :doc:`multimorph`
     - Group-wise registration and atlas construction.
     - MultiMorph (GroupNet3D), TemplateCreation
   * - :doc:`backbones_and_heads`
     - Reusable encoders and task heads.
     - Swin / deformable Swin transformers, segmentation, synthesis and SSL heads

.. toctree::
   :maxdepth: 1

   transmorph
   lddmm
   vfa
   hypertransmorph
   hypermorph
   voxelmorph
   lessnet
   fouriernet
   sitreg
   convexadam
   deedsbcv
   multimorph
   backbones_and_heads
