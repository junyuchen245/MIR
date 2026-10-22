VFA: Vector Field Attention
===========================

Overview
--------

Vector Field Attention replaces the decoder of a registration network by an attention
mechanism: correspondences are found by comparing the moving and fixed feature maps directly,
and the displacement is read off the attention weights. VFA is light (about 2 M parameters),
contrast-robust, and the model behind the LUMIR24 / LUMIR25 challenge entries.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 26 50 24

   * - Class
     - Description
     - Output
   * - ``VFA``
     - Base model; ``SVF=True`` integrates the field, ``return_full=True`` also returns the
       warped images and the inverse flow, ``return_all_flows=True`` the per-level flows.
     - ``flow``
   * - ``VFASPR``
     - VFA with a spatially varying regularization decoder on the matching features.
     - ``(flow, wts)`` or ``(pos_flow, neg_flow, wts)``
   * - ``TransVFA``
     - Transformer-augmented VFA encoder.
     - ``flow``
   * - ``HyperVFA``, ``HyperVFASPR``
     - Hypernetwork-conditioned regularization weight (see :doc:`hypertransmorph`).
     - as VFA / VFASPR

Usage
-----

.. code-block:: python

   from MIR.models import VFA, VFASPR
   import MIR.models.configs_VFA as configs_VFA

   config = configs_VFA.get_VFA_default_config()
   config.img_size = fix.shape[2:]
   model = VFA(config, device='cuda:0').cuda()
   flow = model((mov, fix))

   spr = VFASPR(config, device='cuda:0', SVF=True).cuda()
   pos_flow, neg_flow, wts = spr((mov, fix))

Pretrained weights
------------------

``ModelWeights['VFA-LUMIR24-MonoModal']`` (key ``model_state_dict``),
``ModelWeights['VFA-LUMIR25-MultiModal']``, ``ModelWeights['VFA-IXI-Atlas2Subject']``,
``ModelWeights['VFA-SPR-IXI-Atlas2Subject']``, segmentation and synthesis heads
``ModelWeights['VFA-SegHead']``, ``['VFA-SynthHead']``.

Key modules
-----------

- ``MIR.models.VFA``, ``MIR.models.configs_VFA``
- ``MIR.models.TransVFA``, ``MIR.models.HyperVFA``

Related tutorials
-----------------

- `LUMIR25 VFA <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_VFA>`_
- `LUMIR24 VFA <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir24_VFA>`_
- `Brain registration with pretrained VFA <https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_registration>`_
- `VFA features for segmentation <https://github.com/junyuchen245/MIR/tree/main/tutorials/VFA_image_seg>`_
- `VFA features for synthesis <https://github.com/junyuchen245/MIR/tree/main/tutorials/VFA_image_synthesis>`_

References
----------

- Liu, Chen, Zuo, Carass, Prince. Vector field attention for deformable image registration.
  *Journal of Medical Imaging* 2024.
  `Paper <https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-11/issue-6/064001/Vector-field-attention-for-deformable-image-registration/10.1117/1.JMI.11.6.064001.full>`__
