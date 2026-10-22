Backbones and heads
===================

Overview
--------

The building blocks shared by the registration models can be used on their own, for example to
attach a segmentation or synthesis head to pretrained registration features.

Backbones
---------

- ``MIR.models.Swin_Transformer.SwinTransformer``: 3-D Swin Transformer encoder (TransMorph).
- ``MIR.models.Deformable_Swin_Transformer.DefSwinTransformer`` and
  ``Deformable_Swin_Transformer_v2.DefSwinTransformerV2``: deformable cross-attention encoders
  (TransMorph-DCA).
- ``MIR.models.TransMorph.Conv3dReLU``, ``DecoderBlock``, ``RegistrationHead``: decoder blocks
  and the zero-initialized registration head.
- ``MIR.models.VFA.Encoder`` / ``Decoder``, ``MIR.models.LessNet.LessNetUNet``,
  ``MIR.models.FourierNet.FourierNetBackbone``.

Heads
-----

- ``MIR.models.Segmentation_Heads.AdvancedDecoder3D``: segmentation decoder on registration
  features (used with VFA, see the ``VFA_image_seg`` tutorial).
- ``MIR.models.Synthesis_Heads``: ``SynthesisHead3D``, ``SynthesisHead3DAdvanced``,
  ``EfficientAdvancedSynthHead3D``, ``ConvNeXtSynthHead3D`` (``VFA_image_synthesis``).
- ``MIR.models.Selfsupervised_Learning_Heads``: ``SSLHeadNLvl``, ``SSLHead1Lvl`` for
  self-supervised pretraining of the encoders.
- ``MIR.models.lddmm.KernelWeightHead`` and ``SPRHead``: model-agnostic heads that predict
  kernel-mixture weights or an SPR map from the images and a field (:doc:`lddmm`).
