Quick start
===========

Pick the start that matches your task. All three use pretrained weights that are downloaded
on first run and need only a CUDA-enabled PyTorch and an editable install of MIR
(:doc:`installation`).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Guide
     - You will
   * - :doc:`quick_start_brain_registration`
     - Register a brain MRI to a template: affine pre-alignment with ``AffineReg3D``, then
       deformable registration with VFA, TransMorph-TVF or ConvexAdam-MIND, and compose the
       flows.
   * - :doc:`quick_start_ixi_benchmarking`
     - Benchmark seven registration models on IXI from a notebook and get per-model Dice
       tables and box plots.
   * - :doc:`brain_template_example`
     - Build a population template from LUMIR24 by iterating registration and averaging.

To train a model, start from the tutorial that matches your data (:doc:`tutorials`); the
LUMIR25 TransMorph and LDDMM tutorials are the most complete training templates.
