Registration in the browser
===========================

.. raw:: html

   <p><a class="mir-btn mir-btn-primary" href="https://junyuchen.me/playground/">Open the live playground</a></p>

MIR models are plain PyTorch modules, so they export to ONNX and run anywhere ONNX Runtime does, including a
web page. The `registration playground <https://junyuchen.me/playground/>`__ on Junyu Chen's site does exactly
that: a hypernetwork-conditioned VFA model with **spatially varying regularization** (SPR), the method
introduced in `Chen et al., Medical Image Analysis 2025 <https://www.sciencedirect.com/science/article/pii/S1361841525004335>`__
and implemented in MIR, registers brain MRI slices in the visitor's browser. Two sliders re-condition the
network on the fly (prior strength :math:`\kappa` and regularization weight :math:`w`), and the page shows the
warped image, the deformed grid, the learned per-pixel regularization map :math:`\lambda(x)`, the Jacobian
determinant, Dice and folding, all computed locally in about 300 ms per registration.

The complete recipe is the tutorial
`tutorials/web_playground_HyperVFASPR_2D <https://github.com/junyuchen245/MIR/tree/main/tutorials/web_playground_HyperVFASPR_2D>`__.
This page summarizes the parts that generalize to any MIR model.

What runs in the page
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Choice
   * - Models
     - ``HyperVFASPR`` (two hyperparameters, returns ``(flow, lambda)``) and ``HyperVFA`` (one hyperparameter,
       spatially invariant baseline), both 2D, ``start_channels=8``, ``max_channels=32``,
       ``nb_hyp_units=16``, 3.8 M parameters each.
   * - Data
     - 192×160 axial slices of the preprocessed IXI volumes (CC BY-SA 3.0); eight test subjects in the page.
   * - Loss
     - ``mse(warp(moving, flow), fixed) + w · LocalGrad2d('l2')(flow, lambda) + logBeta()(lambda, 1 + κ·w)``
   * - Runtime
     - `ONNX Runtime Web <https://onnxruntime.ai/docs/tutorials/web/>`__ (WebAssembly), models exported at
       opset 16 with float16 initializers (7.7 MB each).

The per-pixel optimum of the regularizer is :math:`\lambda^*(x) = \kappa / |\nabla u(x)|^2` (clipped to 1):
where the deformation must be sharp, :math:`\lambda` drops and the penalty relaxes; where tissue moves coherently
it stays near 1. Parameterizing the Beta prior as :math:`\alpha = 1 + \kappa w` makes the map independent of
the overall regularization weight, which is what makes the two sliders behave intuitively.

Training in 2D: what is different
---------------------------------

Everything in MIR that the tutorial uses is dimension-agnostic (``VFA``, ``HyperVFA``, ``HyperVFASPR``,
``SpatialTransformer``, ``LocalGrad2d`` / ``LocalGrad3d``, ``logBeta``), but two things that work out of the
box in 3D need attention in 2D:

- **Similarity.** The local NCC (``NCC_vxm``) with a 9-pixel window is degenerate on intensity-normalized,
  nearly uniform 2D slices: the fixed numerical epsilon in the denominator dominates the 81-pixel variance sums
  (in 3D the window has 729 voxels). Use MSE, or a much larger window, in 2D.
- **Hypernetwork initialization.** With a narrow embedding (``nb_hyp_units=16`` instead of the default 96) the
  default initialization leaves the hyperparameter MLP nearly dead, and a plain Kaiming initialization makes the
  generated convolution kernels about five times too large, which saturates VFA's attention softmax. The
  tutorial's ``init_hypernetwork`` uses Kaiming weights with small positive biases for the MLP, then rescales
  each kernel generator so that the generated kernels have He-normal scale at the reference code 0.5.

``config.initialize`` (VFA's initial displacement scale) of 1.0 instead of 0.1 also speeds up 2D training
considerably.

Exporting a model
-----------------

.. code-block:: python

   import torch, onnx
   from MIR.models import HyperVFASPR
   import MIR.models.configs_VFA as configs_VFA

   cfg = configs_VFA.get_VFA_default_config()
   cfg.img_size, cfg.start_channels, cfg.matching_channels = (192, 160), 8, 8
   cfg.max_channels, cfg.nb_hyp_units, cfg.initialize = 32, 16, 1.0
   model = HyperVFASPR(cfg, device='cpu').eval()
   model.load_state_dict(torch.load('last.pth', map_location='cpu')['state_dict'])

   class Wrap(torch.nn.Module):               # ONNX wants positional tensor inputs
       def __init__(self, m): super().__init__(); self.m = m
       def forward(self, mov, fix, h): return self.m((mov, fix), h)

   mov = fix = torch.zeros(1, 1, 192, 160); h = torch.tensor([[0.5, 0.5]])
   torch.onnx.export(Wrap(model), (mov, fix, h), 'hypervfa_spr_2d.onnx',
                     input_names=['moving', 'fixed', 'hyper'], output_names=['flow', 'weights'],
                     opset_version=16, do_constant_folding=True)
   onnx.checker.check_model('hypervfa_spr_2d.onnx')

Notes:

- Opset 16 or newer is required for ``grid_sample``. The VFA decoders switch to an ONNX-friendly patch
  extraction automatically during export, and use non-negative transpose axes.
- Store large initializers as float16 with a ``Cast`` back to float32 (see ``fp16_weights`` in the tutorial)
  to halve the download without changing the computation. Dynamic int8 quantization quantizes the
  kernel-generating linear layers of the hypernetwork and visibly degrades the deformation.
- Always compare the ONNX Runtime output with PyTorch on a real pair before publishing.

Running it in JavaScript
------------------------

.. code-block:: javascript

   const session = await ort.InferenceSession.create('models/hypervfa_spr_2d_fp16w.onnx',
                                                     { executionProviders: ['wasm'] });
   const feeds = {
     moving: new ort.Tensor('float32', movingPixels, [1, 1, 192, 160]),   // intensities in [0, 1]
     fixed:  new ort.Tensor('float32', fixedPixels,  [1, 1, 192, 160]),
     hyper:  new ort.Tensor('float32', new Float32Array([kappaCode, wCode]), [1, 2]),
   };
   const { flow, weights } = await session.run(feeds);
   // flow.data: Float32Array [2, H, W], channel 0 along rows, channel 1 along columns, in pixels,
   // with the MIR convention warped(y, x) = moving(y + flow0, x + flow1).

Warping, label warping, Dice, the Jacobian determinant and the deformed-grid overlay are a few dozen lines of
JavaScript (``web/playground.js`` in the tutorial). Nothing leaves the visitor's device.

Reference
---------

- Chen, Wei, Liu, Bian, He, Carass, Bai, Du. Unsupervised learning of spatially varying regularization for
  diffeomorphic image registration. *Medical Image Analysis*, 2025.
  `Paper <https://www.sciencedirect.com/science/article/pii/S1361841525004335>`__ |
  `Code <https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg>`__
- Hoopes, Hoffmann, Fischl, Guttag, Dalca. HyperMorph: amortized hyperparameter learning for image registration.
  *IPMI*, 2021 (hypernetwork conditioning).
- Liu, Chen, Zuo, Carass, Prince. Vector field attention for deformable image registration. *JMI*, 2024.
