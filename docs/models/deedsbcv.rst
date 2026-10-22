deedsBCV
========

Overview
--------

deedsBCV (Heinrich et al.) is a discrete, MRF-based registration method with a self-similarity
descriptor. MIR drives the compiled Linux binary from Python so that it can be evaluated with the
same scripts as the learned models.

Usage
-----

The wrapper in ``MIR.models.deedsBCV`` takes NIfTI paths and returns the displacement as a MIR
voxel field; ``MIR.utils.deedsbcv_binary`` locates the bundled binary or downloads it when
configured. See the IXI tutorial for an end-to-end example.

Notes
-----

- Linux only (compiled binary); the rest of MIR is unaffected on other platforms.
- Fields are produced on the image grid with scipy-based resampling; no autograd.

Key modules
-----------

- ``MIR.models.deedsBCV``, ``MIR.utils.deedsbcv_binary``

Related tutorials
-----------------

- `IXI deedsBCV validation <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_deedsBCV>`_

References
----------

- Heinrich, Jenkinson, Brady, Schnabel. MRF-based deformable registration and ventilation
  estimation of lung CT. *IEEE TMI* 2013. `Repo <https://github.com/mattiaspaul/deedsBCV>`__
