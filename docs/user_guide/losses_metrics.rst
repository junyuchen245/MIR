Losses and metrics
==================

Similarity losses
-----------------

All similarity losses take ``(warped, fixed)`` and return a scalar to minimize. They live in
``MIR.image_similarity``.

.. list-table::
   :header-rows: 1
   :widths: 26 44 30

   * - Loss
     - Use it for
     - Notes
   * - ``NCC_vxm``, ``NCC_gauss``, ``NCC_mok``, ``NCC_vfa``
     - Mono-modal registration (same contrast)
     - Local normalized cross-correlation with box or Gaussian windows
   * - ``FastNCC``, ``NCC_fp16``
     - Large volumes, limited memory
     - Faster or half-precision NCC
   * - ``SSIM3D``
     - Mono-modal, structure-sensitive
     - Used by the PET/CT atlas models
   * - ``PCC``
     - Global intensity agreement
     - Pearson correlation
   * - ``MIND_loss``
     - Multi-modal (MR-CT, multi-contrast MRI)
     - Modality-independent neighbourhood descriptor
   * - ``MutualInformation``, ``localMutualInformation``, ``MattesMutualInformation``,
       ``NormalizedMutualInformation``
     - Multi-modal
     - Global or local MI, Parzen-window estimates
   * - ``CorrRatio``, ``LocalCorrRatio``
     - Multi-modal with a functional intensity relationship (PET-CT)
     - Correlation ratio
   * - ``NormalizedGradientFieldLoss``
     - Multi-modal edges
     - Normalized gradient fields

Label supervision
-----------------

When segmentations are available at training time, a Dice loss on warped labels improves anatomical
accuracy.

.. code-block:: python

   from MIR.image_similarity import DiceLoss, sparse_dice_from_int_labels

   dice = DiceLoss(num_class=K, one_hot=True)
   loss_dice = dice(warp(moving_onehot, flow), fixed_labels.long())      # bilinear warp keeps gradients

   # many labels (e.g. 100+ organs): sample a subset per iteration to save memory
   loss_dice = sparse_dice_from_int_labels(src_lbl_int=moving_labels, tgt_lbl_int=fixed_labels,
                                           flow_src2tgt=flow, warp_fn=warp, num_classes=K, K=36,
                                           include_bg=True, present_only=True, dice_loss=dice)

Warp one-hot or soft label maps with the bilinear transformer during training: the nearest-neighbour
transformer has no gradient.

Regularizers
------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Regularizer (``MIR.deformation_regularizer``)
     - Penalizes
   * - ``Grad3d``, ``Grad2D``, ``Grad3DiTV``
     - Spatial gradients of the flow (diffusion, total variation, isotropic TV)
   * - ``DisplacementRegularizer``
     - Bending energy or L1 / L2 gradient norm (``energy_type='bending' | 'gradient-l1' | 'gradient-l2'``)
   * - ``LocalGrad3d`` with ``logBeta`` / ``logGaussian``
     - Spatially weighted smoothness and a prior on the weight map (SPR)
   * - ``GradICON3d``, ``GradICONExact3d``
     - Inverse inconsistency of forward and backward flows
   * - ``KL_divergence``, ``MultiVariateKL_divergence``
     - Divergence from a prior for probabilistic registration

See :doc:`regularization` for how to combine them.

Evaluation metrics
------------------

``MIR.accuracy_measures`` provides overlap and deformation-quality metrics.

.. code-block:: python

   from MIR.accuracy_measures import (dice_val_VOI, dice_val_substruct, calc_jac_dets,
                                      get_identity_grid, calc_measurements)

   warped_labels = warp_nn(moving_labels.float(), flow).long()
   mean_dice = dice_val_VOI(warped_labels, fixed_labels.long(), num_clus=K)
   per_label_csv_line = dice_val_substruct(warped_labels, fixed_labels.long(), case_id, num_classes=K)

   disp = flow[0].cpu().numpy()
   jac = calc_jac_dets(disp + get_identity_grid(disp))          # several finite-difference schemes
   n_folded, _, folded_volume, _ = calc_measurements(jac, mask)  # non-diffeomorphic points / volume
   ndv_percent = folded_volume / mask.sum() * 100

The Jacobian metrics implement the *digital diffeomorphism* criteria of Liu et al. (IJCV 2024): a voxel
is folded if any of the finite-difference Jacobians around it is non-positive, and the non-diffeomorphic
volume (NDV) measures how much of the image is affected. This is the folding measure used by the
Learn2Reg LUMIR challenge. For a differentiable Jacobian on the GPU use
``MIR.models.lddmm.jacobian_fd`` and ``det3x3``.

Data augmentation
-----------------

``MIR.RandomMultiContrastRemap`` randomly remaps intensities inside a mask, which makes a model trained
on one MRI contrast robust to others; ``flip_aug`` and ``affine_aug`` provide geometric augmentation.

.. code-block:: python

   from MIR import RandomMultiContrastRemap

   augment = RandomMultiContrastRemap(p=0.85)
   moving_aug = augment(moving, moving > 0)
