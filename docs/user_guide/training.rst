Training a model
================

This page walks through a complete unsupervised training loop with the practices used across the MIR
tutorials. The ready-to-run version for brain MRI is
`tutorials/lumir25_TransMorph <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_TransMorph>`__.

1. Data
-------

Training pairs are drawn at random: each moving image is paired with a different, randomly chosen fixed
image every epoch, so a cohort of ``N`` images gives ``N(N-1)`` possible pairs.

.. code-block:: python

   class PairDataset(torch.utils.data.Dataset):
       def __init__(self, paths):
           self.paths = paths

       def __getitem__(self, i):
           j = random.choice([k for k in range(len(self.paths)) if k != i])
           load = lambda p: torch.from_numpy(nib.load(p).get_fdata()[None] / 255.).float()
           return load(self.paths[i]), load(self.paths[j])

       def __len__(self):
           return len(self.paths)

   loader = DataLoader(PairDataset(train_paths), batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

2. Model, losses, optimizer
---------------------------

.. code-block:: python

   from MIR.models import TransMorphTVF, SpatialTransformer
   from MIR.image_similarity import NCC_vxm
   from MIR.deformation_regularizer import Grad3d
   from MIR import RandomMultiContrastRemap
   import MIR.models.configs_TransMorph as configs_TransMorph

   H, W, D = 160, 224, 192
   config = configs_TransMorph.get_3DTransMorph3Lvl_config()
   config.img_size = (H // 2, W // 2, D // 2)          # the network runs at half resolution
   config.window_size = (H // 64, W // 64, D // 64)
   model = TransMorphTVF(config, time_steps=7).cuda()

   warp = SpatialTransformer((H, W, D)).cuda()
   similarity, smoothness = NCC_vxm(), Grad3d(penalty='l2')
   augment = RandomMultiContrastRemap(p=0.85)            # contrast-agnostic training
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)

3. One training step, in both directions
----------------------------------------

The network predicts at half resolution; the flow is upsampled and the loss is computed on the
full-resolution images. Each pair is used in both directions.

.. code-block:: python

   def step(mov, fix):
       mov_half, fix_half = F.avg_pool3d(augment(mov, mov > 0), 2), F.avg_pool3d(augment(fix, fix > 0), 2)
       flow = model((mov_half, fix_half))
       flow = F.interpolate(flow, scale_factor=2, mode='trilinear', align_corners=False) * 2
       loss = similarity(warp(mov, flow), fix) + 1.0 * smoothness(flow, fix)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       return loss.item()

   for epoch in range(max_epoch):
       for g in optimizer.param_groups:                  # polynomial learning-rate decay
           g['lr'] = 1e-4 * (1 - epoch / max_epoch) ** 0.9
       for mov, fix in loader:
           mov, fix = mov.cuda(), fix.cuda()
           step(mov, fix)
           step(fix, mov)

4. Validation and checkpoints
-----------------------------

Validate on labeled pairs with the nearest-neighbour transformer and keep the best checkpoints.

.. code-block:: python

   from MIR.accuracy_measures import dice_val_VOI

   warp_nn = SpatialTransformer((H, W, D), mode='nearest').cuda()
   model.eval()
   with torch.no_grad():
       scores = []
       for mov, fix, mov_seg, fix_seg in val_loader:
           flow = model((F.avg_pool3d(mov.cuda(), 2), F.avg_pool3d(fix.cuda(), 2)))
           flow = F.interpolate(flow, scale_factor=2, mode='trilinear', align_corners=False) * 2
           scores.append(dice_val_VOI(warp_nn(mov_seg.cuda(), flow).long(), fix_seg.cuda().long(),
                                      num_clus=n_labels).item())
   torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
              f'experiments/dsc{np.mean(scores):.4f}.pth.tar')

5. Variations
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - To add ...
     - Change
   * - Diffeomorphism
     - ``SVF=True`` on the model (:doc:`transformations`).
   * - Spatially varying regularization
     - an SPR model, ``LocalGrad3d`` and ``logBeta`` in the loss (:doc:`regularization`).
   * - Label supervision
     - a Dice term on warped one-hot labels (:doc:`losses_metrics`).
   * - Multi-modal data
     - ``MIND_loss``, mutual information or correlation ratio instead of NCC.
   * - Warm start
     - load pretrained weights with ``MIR.utils.load_partial_weights`` (:doc:`pretrained_weights`).
   * - Per-case refinement
     - instance optimization: fine-tune a copy of the network for a few dozen steps on one pair at test time.
   * - Mixed precision
     - wrap the forward pass and loss in ``MIR.utils.precision.autocast("fp16")`` and step through
       ``grad_scaler("fp16")``; losses stay in float32 (:doc:`precision_multigpu`).
   * - Several GPUs
     - ``MIR.utils.distributed.init_distributed`` / ``wrap_model`` with ``torchrun``, a ``distributed_sampler``, and
       ``state_dict_for_saving`` for checkpoints (:doc:`precision_multigpu`).

.. tip::

   ``MIR.utils.Logger`` mirrors ``print`` to a log file and ``AverageMeter`` tracks running means, as in
   the tutorial scripts.
