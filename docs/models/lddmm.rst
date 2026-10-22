LDDMM: time-varying velocity fields
===================================

Overview
--------

The LDDMM components replace the stationary velocity field of a registration network by a
**time-varying velocity field**. The deformation is the end point of the flow

.. math::

   \frac{d\phi_t}{dt} = v_t \circ \phi_t, \qquad \phi_0 = \mathrm{id}, \qquad t \in [0, 1],

where every velocity :math:`v_t` lives in the reproducing-kernel Hilbert space defined by a
smoothing kernel :math:`K = L^{-1}`. The network predicts the parameters of the flow and the
flow is integrated inside the model, so feature extraction, velocity field, diffeomorphism
and image similarity are trained **end to end**. There is no separate "network, then LDDMM
refinement" stage.

Two parameterizations are provided (``param``):

.. list-table::
   :header-rows: 1
   :widths: 14 30 26 30

   * - ``param``
     - The network predicts
     - Regularizer
     - Properties
   * - ``'shooting'`` (default)
     - the initial momentum :math:`m_0` (one 3-channel field)
     - :math:`\langle m_0, K m_0\rangle = \|v_0\|_V^2`
     - geodesic shooting: :math:`m_0` is transported with the conservation law
       :math:`m_t = |D\psi_t^{-1}| (D\psi_t^{-1})^{\top} m_0 \circ \psi_t^{-1}`,
       :math:`v_t = K m_t`; kinetic energy is constant along the path; compact
       representation of the whole deformation
   * - ``'tvf'``
     - one pre-velocity :math:`p_t` per time step
     - :math:`\sum_t \Delta t\, \langle p_t, K p_t\rangle`
     - relaxation formulation of Beg et al. 2005; more freedom, not a geodesic

Forward and inverse maps are integrated jointly with explicit Euler composition over
``time_steps`` steps, so every model returns both ``flow`` (the displacement that warps the
moving image) and ``inv_flow``.

Kernels
-------

``MIR.models.FourierKernel`` applies :math:`K` in the Fourier domain (periodic boundaries):

- ``kernel='helmholtz'`` (default): :math:`L = (-\alpha \Delta + \gamma)^{s}` with
  ``alpha``, ``gamma``, ``power``; :math:`\sqrt{\alpha/\gamma}` is the characteristic
  length in voxels.
- ``kernel='gaussian'``: a Gaussian or a sum of Gaussians (``sigma``, ``weights``).

``MIR.models.SpatiallyVaryingKernel`` is a multi-Gaussian kernel with **per-voxel mixture
weights** (Shen, Vialard, Niethammer, NeurIPS 2019):

.. math::

   K_w m = \sum_i \sqrt{w_i}\; G_{\sigma_i} * (\sqrt{w_i}\, m), \qquad w_i(x) \ge 0,\;
   \sum_i w_i(x) = 1 .

Where the weights favour a large :math:`\sigma` the deformation is smooth, where they favour
a small one it can be sharp. The weights are advected with the flow and, because the momentum
conservation law does not depend on the kernel, geodesic shooting still conserves energy.

Spatially varying regularization inside the metric
--------------------------------------------------

Pass ``sigmas=[...]`` to any LDDMM model to use the spatially varying kernel. The weights
come from one of three sources:

- **Learned mixture (``spr=None``).** A ``KernelWeightHead`` (or a user-supplied callable)
  predicts a softmax over the Gaussians per voxel; the ``omt`` penalty returned by the model
  (optimal mass transport, in ``[0, 1]``) charges for moving weight away from the largest
  :math:`\sigma` and plays the role of a prior. Add ``out['omt'] * omt_weight`` to the loss.
- **SPR map, kernel mode (``spr='kernel'``).** A TransMorph-SPR style map :math:`\lambda`
  in :math:`(0, 1]` selects the kernel scale: :math:`\lambda = 1` the smoothest Gaussian,
  :math:`\lambda = 0` the sharpest, in between linear interpolation along ``sigmas``.
  Use the same Beta prior as the loss-side SPR models: ``logBeta(out['spr_weight'], 1 + w)``.
- **SPR map, energy mode (``spr='energy'``).** The stationary kernel is kept and
  :math:`\lambda` weights the kinetic-energy integrand, the literal counterpart of
  ``LocalGrad3d(flow, lam)``.

The map :math:`\lambda` comes from a model-agnostic ``SPRHead`` (``spr_source='head'``) or
from the backbone itself when it is an SPR network returning ``(field, lam)`` such as
``TransMorphTVFSPR(SVF=False)`` or ``VFASPR(SVF=False)`` (``spr_source='backbone'``).

Models
------

``TransMorphLDDMM``
   The dedicated TransMorph variant: Swin encoder and decoder of TransMorph, one momentum
   head (``shooting``) or per-step heads on the shared decoder (``tvf``), an optional
   TransMorph-SPR head (``spr=...``) or kernel-weight head (``sigmas=...``) on the decoder
   features. Compatible with TransMorph / TransMorph-TVF checkpoints through
   ``load_backbone_weights`` (encoder and decoder are reused, heads start from scratch).

``LDDMMWrapper``
   Model-agnostic: wraps **any** backbone that maps ``(moving, fixed)`` to a dense
   3-channel field (TransMorph, VFA, VoxelMorph, LessNet, Fourier-Net, SITReg through
   ``adapter='sitreg'``, or a custom adapter). The backbone's field is the momentum
   (``shooting``) or the per-step pre-velocity (``tvf``, the backbone is re-applied at every
   step to the warped moving image). Build the backbone with its own SVF integration
   switched off. ``field_scale`` adds a learnable scalar on the backbone field, which is
   needed when warm-starting from a backbone trained to output displacements: a trained
   displacement field is far too large and rough to be a momentum, so start at 0.1 to 0.3
   and let training grow it. ``int_scale=2`` integrates the flow on a grid twice as fine
   as the backbone output.

Usage
-----

.. code-block:: python

   from MIR.models import TransMorphLDDMM, LDDMMWrapper, VFA, load_backbone_weights
   from MIR.deformation_regularizer import logBeta
   import MIR.models.configs_TransMorph as configs_TransMorph

   # dedicated TransMorph-LDDMM, stationary Helmholtz kernel, geodesic shooting
   config = configs_TransMorph.get_3DTransMorph3Lvl_config()
   config.img_size = (80, 112, 96)
   model = TransMorphLDDMM(config, param='shooting', time_steps=7, alpha=2.0, gamma=1.0, power=2).cuda()
   out = model((mov, fix), return_dict=True)
   loss = ncc(warp(mov, out['flow']), fix) + 0.01 * out['energy']

   # any backbone, spatially varying kernel with learned mixture weights
   model = LDDMMWrapper(VFA(vfa_config, device='cuda:0'), param='shooting', time_steps=7,
                        sigmas=[1, 2, 4, 8]).cuda()
   out = model((mov, fix), return_dict=True)
   loss = ncc(warp(mov, out['flow']), fix) + 0.01 * out['energy'] + 0.1 * out['omt']

   # your SPR network as the backbone, its weight map setting the kernel scale
   model = LDDMMWrapper(TransMorphTVFSPR(config, SVF=False), spr='kernel', spr_source='backbone',
                        sigmas=[1, 2, 3, 4], field_scale=0.2).cuda()
   out = model((mov, fix), return_dict=True)
   loss = ncc(warp(mov, out['flow']), fix) + 0.002 * out['energy'] + logBeta()(out['spr_weight'], 1.005)

The output dictionary always contains ``flow``, ``inv_flow``, ``energy`` (differentiable)
and ``energies`` (detached per-step kinetic energies: their constancy is the diagnostic that
the geodesic integration is accurate); shooting adds ``momentum``, ``tvf`` adds
``pre_velocities``, spatially varying kernels add ``kernel_weights`` and ``omt`` or
``spr_weight``. ``return_path=True`` on the wrapper also returns the intermediate ``flows``
and ``velocities`` for path visualizations.

Practical notes
---------------

- Kernel parameters are expressed in voxels of the grid the backbone outputs. For a network
  at half resolution ``sigmas=[1, 2, 3, 4]`` is a good starting range for brains; at full
  resolution double them.
- The kinetic energy is not on the scale of a diffusion penalty. Calibrate its weight so
  that the term is a few percent of the similarity term at convergence; the tutorials
  record the magnitudes used.
- Momentum transport samples :math:`m_0` with trilinear interpolation and loses a few
  percent of kinetic energy over the path for rough momenta; a smoother kernel helps, more
  time steps do not.
- The periodic boundary of the FFT is harmless for zero-padded images; crop or pad if the
  anatomy touches the volume border.

Key modules
-----------

- ``MIR.models.lddmm``: ``FourierKernel``, ``SpatiallyVaryingKernel``, ``KernelWeightHead``,
  ``SPRHead``, ``LDDMMIntegrator``, ``LDDMMWrapper``, ``jacobian_fd``, ``det3x3``,
  ``load_backbone_weights``
- ``MIR.models.TransMorphLDDMM``: ``TransMorphLDDMM``

Related tutorials
-----------------

- `LUMIR25 TransMorph-LDDMM <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_TM_LDDMM>`_
  (reference implementation: training, evaluation, calibration tables, all flags)
- `LUMIR25 TransMorph + SPR + LDDMM <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_TM_SPR_LDDMM>`_
- `LUMIR25 TransMorph + region-specific kernel (OMT) + LDDMM <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_TM_OMT_LDDMM>`_
- `LUMIR25 SITReg + LDDMM <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_SITReg_LDDMM>`_
- `LUMIR25 SITReg + SPR + LDDMM <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_SITReg_SPR_LDDMM>`_

References
----------

- Beg, Miller, Trouve, Younes. Computing large deformation metric mappings via geodesic flows
  of diffeomorphisms. *IJCV* 2005.
- Vialard, Risser, Rueckert, Cotter. Diffeomorphic 3D image registration via geodesic shooting
  using an efficient adjoint calculation. *IJCV* 2012.
- Zhang, Fletcher. Fast diffeomorphic image registration via Fourier-approximated Lie algebras
  (FLASH). *IJCV* 2019.
- Yang, Kwitt, Styner, Niethammer. Quicksilver: fast predictive image registration. *NeuroImage* 2017.
- Shen, Vialard, Niethammer. Region-specific diffeomorphic metric mapping. *NeurIPS* 2019.
- Chen et al. Unsupervised learning of spatially varying regularization for diffeomorphic image
  registration. *Medical Image Analysis* 2025.
