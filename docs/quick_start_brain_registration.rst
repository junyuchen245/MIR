Brain registration (affine + deformable)
========================================

Use the brain registration tutorial notebook in
`tutorials/brain_registration/deformable_registration.ipynb <https://github.com/junyuchen245/MIR/blob/main/tutorials/brain_registration/deformable_registration.ipynb>`_
to run an affine pre-alignment followed by deformable registration with VFA, TransMorphTVF, or ConvexAdam-MIND.

.. code-block:: bash

    cd tutorials/brain_registration

.. code-block:: python

   from MIR.models import SpatialTransformer, VFA, AffineReg3D, TransMorphTVF, convex_adam_MIND
   import nibabel as nib
   import torch
   from MIR import ModelWeights
   import MIR.models.configs_VFA as CONFIGS_VFA
   import MIR.models.configs_TransMorph as configs_TransMorph
   import MIR.models.convexAdam.configs_ConvexAdam_MIND as CONFIGS_CVXAdam
   import torch.nn.functional as F

   # load images
   img_nib = nib.load('sub-01_T1w.nii.gz')
   template_nib = nib.load('LUMIR_template.nii.gz')

   # affine registration
   img_torch = torch.from_numpy(img_nib.get_fdata()[None, None, ...]).float().cuda(0)
   template_torch = torch.from_numpy(template_nib.get_fdata()[None, None, ...]).float().cuda(0)
   spatial_trans = SpatialTransformer(size=template_torch.shape[2:], mode='bilinear').cuda(0)
   affine_model = AffineReg3D(vol_shape=template_torch.shape[2:], dof="affine").cuda(0)
   output = affine_model.optimize(img_torch, template_torch, steps_per_scale=(50, 50), verbose=True)
   affine_flow = output['flow']
   deformed = output['warped']

   # deformable registration (VFA)
   config = CONFIGS_VFA.get_VFA_default_config()
   config.img_size = template_torch.shape[2:]
   VFA_model = VFA(config, device='cuda:0').cuda()
   weights = torch.load('pretrained_wts/VFA_LUMIR24.pth')[ModelWeights['VFA-LUMIR24-MonoModal']['wts_key']]
   VFA_model.load_state_dict(weights)

   with torch.no_grad():
       deformable_flow = VFA_model((deformed, template_torch))
       flow = deformable_flow + spatial_trans(affine_flow, deformable_flow)
       final_output = spatial_trans(img_torch, flow)

   # deformable registration (TransMorphTVF)
   tm_config = configs_TransMorph.get_3DTransMorph3Lvl_config()
   tm_config.img_size = tuple(s // 2 for s in template_torch.shape[2:])
   tm_config.window_size = tuple(s // 64 for s in template_torch.shape[2:])
   tm_config.out_chan = 3
   TM_model = TransMorphTVF(tm_config, time_steps=7).cuda(0)
   tm_weights = torch.load('pretrained_wts/TransMorphTVF_LUMIR24.pth.tar')[ModelWeights['TransMorphTVF-LUMIR24-MonoModal']['wts_key']]
   TM_model.load_state_dict(tm_weights)

   with torch.no_grad():
       mov_small = F.avg_pool3d(deformed, 2)
       fix_small = F.avg_pool3d(template_torch, 2)
       tm_flow = TM_model((mov_small, fix_small))
       tm_flow = F.interpolate(tm_flow, size=template_torch.shape[2:], mode='trilinear', align_corners=True)
       tm_flow = tm_flow + spatial_trans(affine_flow, tm_flow)
       tm_output = spatial_trans(img_torch, tm_flow)

   # deformable registration (ConvexAdam-MIND)
   cvx_config = CONFIGS_CVXAdam.get_ConvexAdam_MIND_brain_default_config()
   cvx_flow = convex_adam_MIND(deformed, template_torch, cvx_config)
   cvx_flow = cvx_flow + spatial_trans(affine_flow, cvx_flow)
   cvx_output = spatial_trans(img_torch, cvx_flow)
