# Registration in the browser: HyperVFA-SPR in 2D with ONNX Runtime Web

Live demo: **https://junyuchen.me/playground/**

This tutorial trains two small 2D registration networks with MIR and runs them *in the browser*:

- `HyperVFASPR`: VFA with **spatially varying regularization** (SPR), the method introduced in
  *Unsupervised learning of spatially varying regularization for diffeomorphic image registration*
  (Chen et al., Medical Image Analysis, 2025), conditioned on two hyperparameters by a hypernetwork;
- `HyperVFA`: the same backbone with one global (spatially invariant) regularization weight, as a baseline.

Because both models are hypernetwork-conditioned, one network covers the whole hyperparameter range: moving a
slider in the page is a single forward pass (about 300 ms in WebAssembly on one CPU thread). The page shows the
warped image with a deformation grid, checkerboard / difference / overlay views, the learned regularization map
λ(x), the Jacobian determinant, Dice over the slice labels, and the folding percentage.

Everything below runs from inside this folder.

## 1. Data: 2D axial slices from IXI

The preprocessed IXI volumes (160×192×224, with label maps) are the ones released with TransMorph; see the
[TransMorph IXI page](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md).
IXI is distributed under CC BY-SA 3.0.

```bash
python extract_slices.py --ixi_dir /path/to/IXI --out data
```

This writes `data/{train,val,test}.npz` with five axial slices per subject (192×160 after cropping), their label
maps, and a preview image.

## 2. Train

```bash
python train_web_hypervfa.py --model spr   --iters 30000 --out experiments/spr    # spatially varying
python train_web_hypervfa.py --model plain --iters 30000 --out experiments/plain  # spatially invariant
```

Each run takes about an hour on one GPU. Validation prints Dice, folding and the mean λ at the corners of the
hyperparameter space every 1000 iterations and keeps `best.pth` / `last.pth`.

Hyperparameter codes in `[0, 1]` map to loss weights (log-spaced) and are stored in every checkpoint as
`hyper_map`, which the web page reads:

| code | meaning | range |
|---|---|---|
| `w` | weight of the λ-weighted L2 gradient penalty (`LocalGrad2d`) | 0.0005 – 0.2 |
| `κ` (SPR only) | Beta-prior strength relative to `w`: `α = 1 + κ·w` in `logBeta` | 0.03 – 3 |

With this parameterization the per-pixel optimum of the regularizer is `λ* = κ / |∇u|²` (clipped to 1), so κ
directly sets how much local deformation energy it takes before λ drops below 1.

Training recipe and the reasons behind it (all documented in the script):

- **Similarity: MSE.** On these intensity-normalized, nearly uniform 2D slices the local NCC with a 9-pixel window
  is degenerate (its numerical epsilon dominates the 81-pixel variance sums) and even a 15-pixel window trains
  poorly; MSE trains reliably. In 3D with 9³ windows this is not an issue.
- **Hypernetwork initialization** (`init_hypernetwork`): with a narrow embedding (16 units) the default init
  gives a nearly dead embedding, and a plain Kaiming init produces generated conv kernels about five times too
  large, which saturates VFA's attention. The script uses Kaiming + small positive biases for the MLP, then
  rescales every kernel generator to He-normal scale at the reference code 0.5.
- `config.initialize = 1.0` (VFA's initial displacement scale) instead of 0.1 for much faster 2D training.
- `config.nb_hyp_units = 16` keeps the model at 3.8 M parameters (the hypernetwork width multiplies the
  parameter count; the 3D default is 96).

## 3. Export to ONNX

```bash
python export_web_models.py --ckpt experiments/spr/last.pth   --model spr   --out web/models
python export_web_models.py --ckpt experiments/plain/last.pth --model plain --out web/models
```

The exporter uses opset 16 (needed for `grid_sample`), checks numerical parity with PyTorch and with
`onnxruntime`, and writes both an fp32 file and a `*_fp16w.onnx` file whose large initializers are stored in
float16 and cast back to float32 inside the graph (half the download, no measurable accuracy change). Dynamic
int8 quantization is *not* suitable here: it quantizes the kernel-generating linear layers and visibly degrades
the deformation. A `*.json` file with the model's `hyper_map` and validation numbers is written next to each
model.

MIR's VFA decoders contain an ONNX-friendly path for the 3×3 patch extraction (`_candidates_by_slicing`,
enabled automatically during `torch.onnx.export`), and all transposes use non-negative axes, so no graph
surgery is needed.

## 4. Demo images and the page

```bash
python make_demo_assets.py --data_dir data --out web/samples
cd web && python3 -m http.server 8000      # then open http://localhost:8000/
```

`web/index.html` + `web/playground.js` + `web/playground.css` are a standalone copy of the live page. By default
`index.html` points `data-assets` at `https://junyuchen.me/assets/playground` so it works without training
anything; change it to `.` to use your own `web/samples` and `web/models`. The JavaScript does the warping
(bilinear), label warping (nearest), Dice, Jacobian determinants and the deformed-grid overlay itself; the
network is run with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WebAssembly, loaded from
jsdelivr).

Minimal inference in JavaScript:

```js
const session = await ort.InferenceSession.create('models/hypervfa_spr_2d_fp16w.onnx',
                                                  { executionProviders: ['wasm'] });
const feeds = {
  moving: new ort.Tensor('float32', movingPixels, [1, 1, 192, 160]),   // values in [0, 1]
  fixed:  new ort.Tensor('float32', fixedPixels,  [1, 1, 192, 160]),
  hyper:  new ort.Tensor('float32', new Float32Array([kappaCode, wCode]), [1, 2]),
};
const { flow, weights } = await session.run(feeds);  // flow: [1,2,H,W] voxel displacements (row, col); weights: λ(x)
```

## Files

| File | Purpose |
|---|---|
| `extract_slices.py` | IXI volumes → `data/*.npz` (2D slices + labels) |
| `train_web_hypervfa.py` | training (`--model spr|plain`), hyperparameter mapping, validation |
| `export_web_models.py` | ONNX export, fp16 weights, parity check, metadata JSON |
| `make_demo_assets.py` | PNG slices + label maps + manifest for the page |
| `web/` | standalone page (`index.html`, `playground.js`, `playground.css`) |

Documentation: https://junyuchen.me/MIR/user_guide/browser_deployment.html
