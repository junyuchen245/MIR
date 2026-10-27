Transformation models
=====================

A registration network predicts a dense field. The *transformation model* decides how that field becomes
a deformation, and with it whether the deformation is guaranteed to be smooth and invertible
(diffeomorphic), whether its inverse is available, and how it should be regularized. In MIR the
transformation model is a constructor argument or a wrapper, independent of the architecture.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 38 14 14 14

   * - Model
     - The network output is ...
     - Diffeo-morphic
     - Inverse
     - Cost
   * - Displacement
     - the displacement itself
     - no
     - no
     - lowest
   * - Stationary velocity field (SVF)
     - a velocity ``v``; ``phi = exp(v)`` by scaling and squaring
     - in practice
     - ``exp(-v)``
     - low
   * - Time-varying field (TVF)
     - small displacements from per-step heads, composed
     - approximately
     - no
     - medium
   * - Symmetric, inverse consistent
     - per-level updates through deformation-inversion layers (SITReg)
     - yes
     - exact
     - high
   * - LDDMM
     - an initial momentum (geodesic shooting) or per-step velocities, smoothed by a kernel and integrated
     - yes
     - integrated
     - medium

Displacement
------------

The simplest choice: the output is used directly as ``u``. Fast and flexible, but nothing prevents
folding; rely on a smoothness penalty (:doc:`regularization`) and check folding with the Jacobian metrics
(:doc:`losses_metrics`). Use ``SVF=False`` on models that support both.

Stationary velocity fields
--------------------------

The output is treated as a velocity field and integrated by scaling and squaring with
``MIR.models.VecInt``. The result is diffeomorphic for reasonably smooth velocities and its inverse is
obtained by integrating ``-v``. This is the default for most diffeomorphic models in MIR.

.. code-block:: python

   model = TransMorph(config, SVF=True, SVF_steps=7)     # built in
   # or by hand, for any field:
   from MIR.models import VecInt
   integrate = VecInt(inshape=(D, H, W), nsteps=7)
   flow, inv_flow = integrate(v), integrate(-v)

To turn any displacement field into an SVF (for example to make a network's displacement output
diffeomorphic, or to average fields in the velocity domain), fit one with ``fit_warp_to_svf`` (or the
coarse-to-fine ``fit_warp_to_svf_fast``) and choose what to get back with ``output_type``:

.. code-block:: python

   from MIR.models import fit_warp_to_svf
   v = fit_warp_to_svf(disp, output_type="velocity")          # the fitted velocity field
   disp_diffeo = fit_warp_to_svf(disp, output_type="displacement")   # exp(v), a diffeomorphic displacement

.. note::
   The older names ``output_type="disp"`` (the default) and ``output_type="svf"`` still work but are inverted
   relative to what they return: ``"disp"`` returns the **velocity** and ``"svf"`` returns the **displacement**.
   Prefer the explicit ``"velocity"`` / ``"displacement"`` in new code.

Time-varying composition (TVF)
------------------------------

``TransMorphTVF`` predicts a sequence of small displacements with separate heads, each looking at the
moving image warped by the composition so far, and composes them. This progressive refinement is what
the LUMIR brain models use; it handles large deformations well. The composed result is not
guaranteed diffeomorphic but folds very little in practice.

Symmetric, inverse-consistent mappings
--------------------------------------

SITReg builds the deformation from per-level updates passed through deformation-inversion layers, so the
forward and inverse maps are exact inverses and swapping the inputs swaps the maps. Choose it when
inverse consistency is required, for example for longitudinal or atlas-based analyses. See
:doc:`../models/sitreg`.

LDDMM
-----

LDDMM models the deformation as the end point of a flow driven by a time-varying velocity field in a
reproducing-kernel Hilbert space. The network predicts the initial momentum (geodesic shooting) or a
velocity per time step, the kernel smooths it, and the flow and its inverse are integrated inside the
model, so training remains end to end. The kinetic energy of the path is the regularizer. MIR provides a
dedicated ``TransMorphLDDMM`` and a ``LDDMMWrapper`` that applies the same transformation model to any
backbone. See :doc:`../models/lddmm`.

Choosing
--------

- **Default**: SVF integration (``SVF=True``) on the architecture of your choice.
- **Large deformations** (inter-subject brain, whole body): TransMorph-TVF, or an SVF model with
  instance optimization.
- **Inverse consistency required**: SITReg, or an LDDMM model when an explicit geodesic path is useful.
- **Maximum speed**: displacement models with a smoothness penalty.
