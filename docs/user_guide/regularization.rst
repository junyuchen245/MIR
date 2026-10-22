Regularization
==============

Registration is ill-posed: many deformations match two images equally well. The regularizer encodes which
ones are plausible. MIR offers three levels of control.

Global smoothness
-----------------

A single weight trades accuracy against smoothness everywhere in the image.

.. code-block:: python

   from MIR.deformation_regularizer import Grad3d

   smoothness = Grad3d(penalty='l2')                     # diffusion; 'l1' for total variation
   loss = similarity(warp(moving, flow), fixed) + lam * smoothness(flow, fixed)

``GradICON3d`` and ``GradICONExact3d`` add an inverse-consistency penalty for bidirectional training,
``DisplacementRegularizer(energy_type)`` provides bending energy (``'bending'``) and L1 or L2 gradient norms (``'gradient-l1'``, ``'gradient-l2'``).

Spatially varying regularization (SPR)
--------------------------------------

Uniform regularization over-smooths where anatomy slides or changes abruptly and under-smooths elsewhere.
SPR models predict a per-voxel regularization-strength map ``wts`` in ``(0, 1]`` alongside the flow and
use it to weight a local smoothness penalty. A Beta prior keeps the map close to 1 (full regularization)
unless the data demand otherwise, so the network learns *where* to relax the regularization.

.. code-block:: python

   from MIR.models import TransMorphTVFSPR
   from MIR.deformation_regularizer import LocalGrad3d, logBeta

   model = TransMorphTVFSPR(config, SVF=True).cuda()
   pos_flow, neg_flow, wts = model((moving, fixed))

   local_smoothness, beta_prior = LocalGrad3d(penalty='l2'), logBeta()
   loss = (similarity(warp(moving, pos_flow), fixed)
           + w_grad * local_smoothness(pos_flow, wts)
           + beta_prior(wts, 1.0 + w_beta))              # Beta(alpha = 1 + w_beta, beta = 1)

SPR variants exist for TransMorph-TVF, VFA, LessNet, Fourier-Net(+) and the hypernetwork models; SITReg
and any other backbone can use the model-agnostic ``MIR.models.SPRHead``. ``logGaussian`` is an
alternative prior. The map is interpretable: low values mark regions where the deformation is allowed to
be irregular, such as sliding organ boundaries.

In the LDDMM models the same map can instead set the scale of the smoothing kernel per voxel
(``spr='kernel'``), so the spatially varying regularization becomes part of the metric; see
:doc:`../models/lddmm`.

Hypernetwork-conditioned strength
---------------------------------

Choosing the regularization weight usually needs a sweep. Hypernetwork models take the weight as an
input: one network is trained over a range of weights, and the trade-off is chosen at inference time,
per case if needed.

.. code-block:: python

   from MIR.models import HyperTransMorphTVF

   model = HyperTransMorphTVF(config, time_steps=7).cuda()
   reg_code = torch.tensor([[0.3]], device='cuda')       # sampled at random during training
   flow = model((moving, fixed), reg_code)

See :doc:`../models/hypertransmorph` and :doc:`../models/hypermorph`.

Practical guidance
------------------

- Start from the weights used by the tutorial closest to your data; they are calibrated for the
  similarity loss used there.
- Monitor folding (:doc:`losses_metrics`): a non-diffeomorphic volume above a fraction of a percent
  usually means the smoothness weight is too low or the transformation model too permissive.
- Label supervision (Dice) improves anatomical accuracy but can push the flow to fold at label borders;
  keep a regularizer in the loss when using it.
