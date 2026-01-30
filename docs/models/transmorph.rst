TransMorph
==========

Overview
--------
TransMorph is a transformer-based deformable registration family in MIR. This
page covers TransMorph variants that share the same core architecture and
training utilities, including TransMorphTVF.

Included variants
-----------------
- TransMorph
- TransMorphTVF (time-varying velocity / log-domain diffeomorphic variant)
- TransMorph DCA (cross-attention variant)

Key modules
-----------
- ``MIR.models.TransMorph``
- ``MIR.models.configs_TransMorph``
- ``MIR.models.Deformable_Swin_Transformer``
- ``MIR.models.Deformable_Swin_Transformer_v2``

Related tutorials
-----------------
- `IXI benchmarking <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_benchmarking>`_
- `Build brain template (LUMIR24) <https://github.com/junyuchen245/MIR/tree/main/tutorials/build_brain_template>`_
- `LUMIR24 TransMorph <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir24_TransMorph>`_
