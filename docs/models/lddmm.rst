LDDMM
=====

Overview
--------

Large Deformation Diffeomorphic Metric Mapping models the deformation as the end point of a flow driven
by a time-varying velocity field,

.. math::

   \frac{d\phi_t}{dt} = v_t \circ \phi_t, \qquad \phi_0 = \mathrm{id}, \qquad t \in [0, 1],

where each velocity is smoothed by a kernel :math:`K`. In MIR the network predicts the parameters of the
flow and the integration happens inside the model, so the registration network is trained end to end
like any other. The result is diffeomorphic, comes with its inverse, and the kinetic energy of the path
is the regularizer. Use it when an explicit, invertible path between the images is useful (statistics
on deformations, geodesic interpolation, atlas building), or as an alternative to SVF integration.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 26 50 24

   * - Class
     - Description
     - Output
   * - ``TransMorphLDDMM``
     - TransMorph encoder and decoder with a momentum head (``param='shooting'``) or per-step heads
       (``param='tvf'``); compatible with TransMorph / TransMorph-TVF checkpoints.
     - ``flow`` or a dict
   * - ``LDDMMWrapper``
     - The same transformation model around **any** backbone (VFA, VoxelMorph, LessNet, Fourier-Net,
       SITReg through ``adapter='sitreg'``, or a custom adapter).
     - ``flow`` or a dict

Two parameterizations:

- ``param='shooting'`` (geodesic shooting): the network predicts the initial momentum :math:`m_0`, which
  is transported with the conservation law; the regularizer is :math:`\langle m_0, K m_0\rangle`, constant
  along the path.
- ``param='tvf'`` (relaxation, Beg et al. 2005): the network predicts one velocity per time step; the
  regularizer is the path energy.

Kernels
-------

- ``FourierKernel``: Helmholtz :math:`(-\alpha\Delta + \gamma)^{-s}` (default) or Gaussian, applied in
  the Fourier domain.
- ``SpatiallyVaryingKernel`` (``sigmas=[...]``): a mixture of Gaussians with per-voxel weights (Shen et al.,
  NeurIPS 2019), so the smoothness of the deformation varies in space. The weights come from a learned
  head (with an optimal-mass-transport penalty, ``out['omt']``) or from an SPR map
  (``spr='kernel'``, see :doc:`../user_guide/regularization`).

Usage
-----

.. tab-set::

   .. tab-item:: TransMorphLDDMM

      .. code-block:: python

         from MIR.models import TransMorphLDDMM
         import MIR.models.configs_TransMorph as configs_TransMorph

         config = configs_TransMorph.get_3DTransMorph3Lvl_config()
         config.img_size = (80, 112, 96)
         model = TransMorphLDDMM(config, param='shooting', time_steps=7).cuda()

         out = model((mov, fix), return_dict=True)
         loss = similarity(warp(mov, out['flow']), fix) + 0.01 * out['energy']

   .. tab-item:: Any backbone

      .. code-block:: python

         from MIR.models import LDDMMWrapper, VFA

         backbone = VFA(vfa_config, device='cuda:0')          # its own SVF integration off
         model = LDDMMWrapper(backbone, param='shooting', time_steps=7).cuda()

         out = model((mov, fix), return_dict=True)
         loss = similarity(warp(mov, out['flow']), fix) + 0.01 * out['energy']

   .. tab-item:: With an SPR map

      .. code-block:: python

         from MIR.models import LDDMMWrapper, TransMorphTVFSPR
         from MIR.deformation_regularizer import logBeta

         model = LDDMMWrapper(TransMorphTVFSPR(config, SVF=False), spr='kernel', spr_source='backbone',
                              sigmas=[1, 2, 3, 4], field_scale=0.2).cuda()
         out = model((mov, fix), return_dict=True)
         loss = (similarity(warp(mov, out['flow']), fix) + 0.002 * out['energy']
                 + logBeta()(out['spr_weight'], 1.005))

The output dict contains ``flow``, ``inv_flow``, ``energy`` (differentiable) and ``energies`` (per-step
kinetic energies, a diagnostic of integration accuracy), plus ``momentum`` or ``pre_velocities`` and,
with a spatially varying kernel, ``kernel_weights`` with ``omt`` or ``spr_weight``.

Practical notes
---------------

- Kernel sizes are in voxels of the grid the backbone outputs; ``sigmas=[1, 2, 3, 4]`` suits brain MRI
  at half resolution.
- Calibrate the energy weight so the term is a few percent of the similarity term at convergence.
- When warm-starting from a network trained to output displacements, pass ``field_scale=0.1``-``0.3``:
  a trained displacement field is too large to be used directly as a momentum.

Key modules
-----------

- ``MIR.models.lddmm``: ``LDDMMWrapper``, ``LDDMMIntegrator``, ``FourierKernel``,
  ``SpatiallyVaryingKernel``, ``KernelWeightHead``, ``SPRHead``, ``jacobian_fd``, ``det3x3``
- ``MIR.models.TransMorphLDDMM``

Related tutorials
-----------------

- `LUMIR25 LDDMM <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_TM_LDDMM>`__:
  training, evaluation and calibration for TransMorph and any other backbone, with ready-made
  configurations in ``lumir25_TM_SPR_LDDMM``, ``lumir25_TM_OMT_LDDMM``, ``lumir25_SITReg_LDDMM`` and
  ``lumir25_SITReg_SPR_LDDMM``.

References
----------

- Beg, Miller, Trouve, Younes. Computing large deformation metric mappings via geodesic flows of
  diffeomorphisms. *IJCV* 2005.
- Vialard, Risser, Rueckert, Cotter. Diffeomorphic 3D image registration via geodesic shooting using an
  efficient adjoint calculation. *IJCV* 2012.
- Yang, Kwitt, Styner, Niethammer. Quicksilver: fast predictive image registration. *NeuroImage* 2017.
- Shen, Vialard, Niethammer. Region-specific diffeomorphic metric mapping. *NeurIPS* 2019.
