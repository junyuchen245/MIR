Changelog
=========

Unreleased
----------

A code-quality pass over the package. Public names, call signatures, numerical outputs and pretrained-weight
compatibility are unchanged, except for the bug fixes listed below; this was verified with the checks described
in :doc:`../development/contributing`.

Fixed: may change results
^^^^^^^^^^^^^^^^^^^^^^^^^

These fixes change numbers where the affected option is used. Re-running an analysis that used one of them can give
slightly different results; saved outputs and trained weights are not affected.

- ``VFA`` / ``VFASPR`` with ``SVF=True, return_full=True``: the inverse field was computed as ``exp(-exp(v))``
  (the integrated field integrated a second time). It is now the inverse ``exp(-v)``, which composes with the
  forward field to the identity 4-10x more accurately; the change is sub-voxel (about 0.02-0.4 voxels for
  velocities up to 10 voxels). The forward field and the warped moving image are unchanged.
- ``FastNCC`` used the window edge length instead of the window volume to compute local means; it now agrees
  with ``NCC_vxm``.
- ``jacobian_determinant`` (``MIR.models.registration_utils``) differentiated along the batch axis and returned the
  determinant of the displacement gradient. It now returns the Jacobian determinant of ``x + u(x)`` for any batch
  size. It is only used by ``fit_warp_to_svf(init="jacwt")``.
- ``GradICONExact3d`` sampled the field with ``align_corners=False`` while normalizing coordinates for
  ``align_corners=True`` (an error of about 0.5% in the sampled derivatives); point, channel and step-size
  conventions are now consistent.
- ``MultiVariateKL_divergence`` used ``exp(log_sigma)`` as the variance while ``KL_divergence`` used it as the
  standard deviation; both now treat ``log_sigma`` as a log standard deviation.
- ``MutualInformation`` and ``localMutualInformation`` clamped inputs to ``[0, maxval]``, ignoring ``minval``
  (identical results for the default ``minval=0``).
- ``AffineReg3D.optimize`` reported a ``"loss"`` that compared the raw moving image with the normalized fixed
  image; it now uses the raw intensities for both. The returned warped image and transform are unchanged.

Fixed: errors and crashes
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``MultiMorph``: ``ResizeTransform`` raised ``AttributeError`` when downsampling; ``FastMeanConv3d`` and
  ``FastMeanConv3dUp`` mixed up group summaries for batch sizes above 1 (unchanged for batch size 1);
  ``SimpleUNet`` returned ``(x, x)`` instead of ``x`` when ``bidir=False``.
- ``SwinTransformer``, ``DefSwinTransformer`` and ``DefSwinTransformerV2`` could not be built with their defaults
  (``pat_merg_rf`` and ``dwin_size``), and their ``train()`` returned ``None`` instead of the module.
- ``TransVFA``'s feature decoder could not be built with its default ``vfa_channels``.
- ``NCC_mok`` / ``NCC_mok2`` failed with ``win=None``; ``NCC_vfa`` failed on 1-D inputs; ``NCC_gauss`` now
  rejects non-3-D inputs with a clear message.
- ``DiceSegLoss`` / ``SegmentationLoss`` failed when ``ignore_index`` was outside ``[0, num_classes - 1]``
  (for example 255).
- ``fit_warp_to_svf`` raised ``NameError`` for an unknown ``objective``; it now raises ``ValueError``.
- ``PreAffineToTemplate`` only worked with exactly three scales.
- ``return_orginal=True`` on the VFA models now raises a clear ``NotImplementedError`` (the required helper is not
  part of MIR's VFA port) instead of ``AttributeError``.
- ``MeanStream`` turned its running mean and count into trainable parameters after the first update.
- The deedsBCV helper no longer picks up executables from the current directory when ``MIR_DEEDSBCV_DIR`` is unset.
- ``CenterCropPad3D`` ignored the ``padding_value`` given to its constructor.
- ``RandomMultiContrastRemap`` failed on unbatched ``(C, D, H, W)`` inputs.
- The self-supervised learning heads required a GPU for their sampling noise (same random numbers as before).

Fixed: checkpoints and devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Older checkpoints load with** ``strict=True`` **again.** Checkpoints saved before the spatial transformer's
  sampling grid became a non-persistent buffer (for example the public TransMorph-TVF LUMIR24 weights) contain
  ``spatial_trans.grid`` entries that made strict loading fail. ``SpatialTransformer`` now ignores those stale
  entries.
- ``HyperTransMorphTVFSPR`` failed in ``forward``: its regularization-weight head was called without the
  hyperparameter conditioning. It now runs; parameters are unchanged.
- Losses and models no longer require a GPU: ``CorrRatio``, ``LocalCorrRatio``, ``MutualInformation``,
  ``localMutualInformation``, ``NCC``, ``NCC_gauss``, ``NCC_vxm``, ``MIND_loss``, ``AffineTransformer``, the
  Swin/deformable-Swin positional encodings and windows, and ``invert_warp_via_velocity`` used hard-coded
  ``.cuda()`` calls. They now follow the input's device, so they also run on CPU and on GPUs other than
  ``cuda:0``. Results on the default GPU are unchanged.

Added
^^^^^

- ``MIR.__version__``.
- ``fit_warp_to_svf`` / ``fit_warp_to_svf_fast`` accept ``output_type="velocity"`` and ``"displacement"``. The
  older names keep their behaviour (``"disp"``, the default, returns the velocity; ``"svf"`` returns the
  displacement) because existing scripts depend on them; see :doc:`../user_guide/transformations`.
- Optional dependency groups: ``dev`` (pytest, ruff, pre-commit), ``docs`` (Sphinx toolchain) and
  ``download`` (gdown). ``pytest`` is no longer installed as a runtime dependency.
- Regression tests for the fixes above, a public-API stability test, a lint job in continuous integration
  and a pre-commit configuration.
- Compatibility tooling in ``tools/``: public-API snapshot, golden numerical outputs (including strict loading
  of the bundled pretrained checkpoints) and a syntax-tree equivalence check.
- :doc:`../development/contributing` guide.

Changed
^^^^^^^

- Defaults that could not work before: ``pat_merg_rf`` of the Swin/deformable-Swin encoders is now ``4`` (every
  configuration already passes 4), and ``CenterCropPad3D.forward(padding_value=None)`` uses the constructor's value
  (``0`` by default, as before).

- First-party code is linted and formatted with ruff (vendored code is excluded), unused imports and dead
  locals were removed, and docstrings were completed in the Google style, with type hints on public functions.
  None of this changes behaviour.
