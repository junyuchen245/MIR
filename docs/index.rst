:sd_hide_title:

MIR
===

.. raw:: html

   <div class="mir-hero">
     <span class="mir-eyebrow">PyTorch toolkit for medical image registration</span>
     <h1 class="mir-title">Deformable registration, <span class="mir-grad">from research to results</span></h1>
     <p class="mir-lead">
       MIR brings state-of-the-art registration networks, classical baselines, training losses,
       evaluation metrics and reproducible benchmark pipelines together behind one simple interface:
       a <code>(moving, fixed)</code> pair in, a dense displacement field out.
     </p>
     <div class="mir-actions">
       <a class="mir-btn mir-btn-primary" href="getting_started/installation.html">Get started</a>
       <a class="mir-btn mir-btn-ghost" href="models/index.html">Browse models</a>
       <a class="mir-btn mir-btn-ghost" href="https://github.com/junyuchen245/MIR">GitHub</a>
     </div>
   </div>

.. container:: mir-figure

   .. image:: _static/hero_registration.png
      :alt: A brain MRI pair registered with TransMorph-TVF: moving, fixed, warped moving, deformed grid and log Jacobian determinant

.. container:: mir-caption

   A LUMIR brain MRI pair registered with TransMorph-TVF: moving, fixed and warped images, the deformation
   and its log Jacobian determinant (folding in 0.003% of voxels).

Why MIR
-------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`plug;1.2em` One interface
      :link: user_guide/conventions
      :link-type: doc

      Every model takes ``(moving, fixed)`` and returns a voxel displacement that one
      ``SpatialTransformer`` applies to images and label maps. Swapping architectures is a one-line change.

   .. grid-item-card:: :octicon:`stack;1.2em` A broad model zoo
      :link: models/index
      :link-type: doc

      TransMorph, VFA, SITReg, VoxelMorph, LessNet, Fourier-Net, hypernetwork variants, and the
      classical ConvexAdam, deedsBCV and FireANTs, all callable the same way.

   .. grid-item-card:: :octicon:`git-merge;1.2em` Diffeomorphic by design
      :link: user_guide/transformations
      :link-type: doc

      Choose the transformation model: displacement, stationary or time-varying velocity fields,
      symmetric inverse-consistent mappings, or LDDMM geodesic shooting.

   .. grid-item-card:: :octicon:`meter;1.2em` Spatially varying regularization
      :link: user_guide/regularization
      :link-type: doc

      Learn where the deformation should be smooth and where it may be sharp, with Beta priors,
      hypernetwork-conditioned strength, or weighted metrics.

   .. grid-item-card:: :octicon:`tools;1.2em` Training toolbox
      :link: user_guide/losses_metrics
      :link-type: doc

      NCC, SSIM, MIND, mutual information, correlation ratio and Dice losses; diffusion, local and
      GradICON regularizers; Jacobian-based folding metrics; contrast augmentation.

   .. grid-item-card:: :octicon:`verified;1.2em` Pretrained and benchmarked
      :link: user_guide/pretrained_weights
      :link-type: doc

      Weights for brain MRI (LUMIR, IXI) and PET/CT (autoPET), and end-to-end tutorials that reproduce
      Learn2Reg and IXI benchmarks.

A first look
------------

.. tab-set::

   .. tab-item:: Register with a pretrained model

      .. code-block:: python

         import torch, nibabel as nib
         from MIR import ModelWeights
         from MIR.models import VFA, SpatialTransformer
         import MIR.models.configs_VFA as configs_VFA

         load = lambda p: torch.from_numpy(nib.load(p).get_fdata()[None, None]).float().cuda()
         mov, fix = load('subject.nii.gz'), load('template.nii.gz')

         config = configs_VFA.get_VFA_default_config()
         config.img_size = fix.shape[2:]
         model = VFA(config, device='cuda:0').cuda().eval()
         key = ModelWeights['VFA-LUMIR24-MonoModal']['wts_key']
         model.load_state_dict(torch.load('VFA_LUMIR24.pth')[key])

         with torch.no_grad():
             flow = model((mov, fix))          # [1, 3, D, H, W], in voxels
         warped = SpatialTransformer(fix.shape[2:]).cuda()(mov, flow)

   .. tab-item:: Train a model

      .. code-block:: python

         from MIR.models import TransMorph, SpatialTransformer
         from MIR.image_similarity import NCC_vxm
         from MIR.deformation_regularizer import Grad3d
         import MIR.models.configs_TransMorph as configs_TransMorph

         config = configs_TransMorph.get_3DTransMorph3Lvl_config()
         config.img_size = (160, 192, 224)
         model = TransMorph(config, SVF=True).cuda()   # diffeomorphic (SVF)
         warp = SpatialTransformer(config.img_size).cuda()
         similarity, smoothness = NCC_vxm(), Grad3d(penalty='l2')
         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

         for mov, fix in loader:                       # unsupervised: no labels needed
             mov, fix = mov.cuda(), fix.cuda()
             flow = model((mov, fix))
             loss = similarity(warp(mov, flow), fix) + 1.0 * smoothness(flow, fix)
             optimizer.zero_grad(); loss.backward(); optimizer.step()

   .. tab-item:: Evaluate a registration

      .. code-block:: python

         from MIR.models import SpatialTransformer
         from MIR.accuracy_measures import dice_val_VOI, calc_jac_dets, get_identity_grid, calc_measurements

         warp_labels = SpatialTransformer(fix.shape[2:], mode='nearest').cuda()
         dice = dice_val_VOI(warp_labels(mov_seg, flow).long(), fix_seg.long(), num_clus=n_labels)

         disp = flow[0].cpu().numpy()                  # folding metrics
         jac = calc_jac_dets(disp + get_identity_grid(disp))
         n_folded, _, folded_volume, _ = calc_measurements(jac, mask)

Explore the documentation
-------------------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.2em` Getting started
      :link: getting_started/installation
      :link-type: doc

      Install MIR, run a pretrained model, benchmark on IXI, build a template.

   .. grid-item-card:: :octicon:`book;1.2em` User guide
      :link: user_guide/index
      :link-type: doc

      Conventions, transformation models, regularization, losses and metrics, training recipes.

   .. grid-item-card:: :octicon:`package;1.2em` Models
      :link: models/index
      :link-type: doc

      Every model family with variants, usage, pretrained weights and references.

   .. grid-item-card:: :octicon:`beaker;1.2em` Tutorials
      :link: tutorials
      :link-type: doc

      Runnable training and evaluation pipelines for LUMIR, IXI, autoPET and atlas building.

   .. grid-item-card:: :octicon:`code;1.2em` API reference
      :link: api
      :link-type: doc

      Auto-generated reference of every module, class and function.

   .. grid-item-card:: :octicon:`mortar-board;1.2em` Publications
      :link: about/publications
      :link-type: doc

      The papers implemented in MIR and how to cite them.

Citing MIR
----------

If MIR is useful in your work, please cite the paper of the model you used (listed on each model page
and under :doc:`about/publications`) and link to the
`repository <https://github.com/junyuchen245/MIR>`__.

.. toctree::
   :caption: Getting started
   :hidden:

   getting_started/installation
   getting_started/quickstart
   getting_started/brain_registration
   getting_started/ixi_benchmarking
   getting_started/brain_template

.. toctree::
   :caption: User guide
   :hidden:

   Overview <user_guide/index>
   user_guide/conventions
   user_guide/transformations
   user_guide/regularization
   user_guide/losses_metrics
   user_guide/training
   user_guide/pretrained_weights

.. toctree::
   :caption: Models
   :hidden:

   Overview <models/index>
   models/transmorph
   models/vfa
   models/sitreg
   models/voxelmorph
   models/lessnet
   models/fouriernet
   models/hypertransmorph
   models/hypermorph
   models/lddmm
   models/convexadam
   models/deedsbcv
   models/multimorph
   models/backbones_and_heads

.. toctree::
   :caption: Tutorials and reference
   :hidden:

   tutorials
   api

.. toctree::
   :caption: About
   :hidden:

   about/publications
   about/license
