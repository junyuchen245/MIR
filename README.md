# MIR: Medical Image Registration toolkit

<a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square"></a>
<a href="https://github.com/junyuchen245/MIR/actions/workflows/tests.yml"><img src="https://github.com/junyuchen245/MIR/actions/workflows/tests.yml/badge.svg" alt="Run Tests"></a>
<a href="https://github.com/junyuchen245/MIR/actions/workflows/docs.yml"><img src="https://github.com/junyuchen245/MIR/actions/workflows/docs.yml/badge.svg" alt="Build and Deploy Docs"></a>

**MIR** is a PyTorch research toolkit for deformable medical image registration. It brings
state-of-the-art registration networks (TransMorph, VFA, SITReg, VoxelMorph, LessNet, Fourier-Net and
their variants), classical baselines, training losses, evaluation metrics and reproducible benchmark
pipelines together behind one interface: a `(moving, fixed)` pair in, a dense displacement field out.

**Documentation:** https://junyuchen.me/MIR

## Highlights

- **One convention for every model.** A registration network takes a `(moving, fixed)` pair of
  `[B, 1, D, H, W]` tensors and returns a displacement field in voxels that `SpatialTransformer` applies
  to any image or label map. Swapping architectures is a one-line change.
- **Diffeomorphic by design.** Choose the transformation model: displacement, stationary or time-varying
  velocity fields, symmetric inverse-consistent mappings, or LDDMM geodesic shooting.
- **Spatially varying regularization (SPR)**: learn where the deformation should be smooth and where it
  may be sharp, with a per-voxel regularization map and a Beta prior.
- **Hypernetwork conditioning** (HyperMorph, HyperTransMorph, HyperVFA) to cover a range of
  regularization strengths with one model.
- **Mixed precision and multi-GPU.** Train and run any model in float16 or bfloat16 and on several GPUs with
  DistributedDataParallel or DataParallel; losses and warping stay in float32 for accuracy
  ([docs](https://junyuchen.me/MIR/user_guide/precision_multigpu.html), [example](tutorials/multi_gpu_mixed_precision)).
- **Runs in the browser.** Models export to ONNX; the
  [registration playground](https://junyuchen.me/playground/) runs a 2D HyperVFA-SPR with ONNX Runtime Web
  ([tutorial](tutorials/web_playground_HyperVFASPR_2D), [docs](https://junyuchen.me/MIR/user_guide/browser_deployment.html)).
- **Classical and third-party baselines** behind the same interface: ConvexAdam, deedsBCV, FireANTs,
  SITReg, MultiMorph, intensity normalization.
- **Training and evaluation batteries**: NCC / SSIM / MIND / mutual information / correlation ratio /
  Dice losses, diffusion and local regularizers, Beta priors, GradICON, Jacobian-based folding metrics,
  augmentation, and a registry of pretrained weights.

## Installation

Install a CUDA-enabled PyTorch build first, then MIR in editable mode:

```bash
git clone https://github.com/junyuchen245/MIR.git
cd MIR
python3.8 -m pip install -U pip
python3.8 -m pip install -e .
```

Python 3.8+ is required. deedsBCV is Linux-only; everything else is platform independent.
See the [installation guide](https://junyuchen.me/MIR/getting_started/installation.html) for optional
dependencies and the [pretrained weights](https://junyuchen.me/MIR/user_guide/pretrained_weights.html).

## Sixty-second example

```python
import torch, nibabel as nib
from MIR import ModelWeights
from MIR.models import VFA, SpatialTransformer
import MIR.models.configs_VFA as configs_VFA

mov = torch.from_numpy(nib.load('subject.nii.gz').get_fdata()[None, None]).float().cuda()
fix = torch.from_numpy(nib.load('template.nii.gz').get_fdata()[None, None]).float().cuda()

config = configs_VFA.get_VFA_default_config()
config.img_size = fix.shape[2:]
model = VFA(config, device='cuda:0').cuda().eval()
model.load_state_dict(torch.load('VFA_LUMIR24.pth')[ModelWeights['VFA-LUMIR24-MonoModal']['wts_key']])

with torch.no_grad():
    flow = model((mov, fix))                       # [1, 3, D, H, W] displacement in voxels
warped = SpatialTransformer(fix.shape[2:]).cuda()(mov, flow)
```

## Models

| Family | Summary | Docs |
|---|---|---|
| TransMorph, TransMorph-TVF, TransMorph-TVF-SPR, TransMorph-DCA | Swin-Transformer registration networks | [TransMorph](https://junyuchen.me/MIR/models/transmorph.html) |
| VFA, VFA-SPR, TransVFA, HyperVFA | Vector field attention | [VFA](https://junyuchen.me/MIR/models/vfa.html) |
| HyperTransMorph, HyperMorph | Hypernetwork-conditioned regularization | [HyperTransMorph](https://junyuchen.me/MIR/models/hypertransmorph.html) |
| VoxelMorph, LessNet, Fourier-Net(+) | Convolutional baselines (with SVF and SPR variants) | [VoxelMorph](https://junyuchen.me/MIR/models/voxelmorph.html), [LessNet](https://junyuchen.me/MIR/models/lessnet.html), [Fourier-Net](https://junyuchen.me/MIR/models/fouriernet.html) |
| SITReg | Symmetric, inverse-consistent, topology-preserving registration | [SITReg](https://junyuchen.me/MIR/models/sitreg.html) |
| ConvexAdam, FireANTs, deedsBCV | Optimization-based and classical baselines | [ConvexAdam](https://junyuchen.me/MIR/models/convexadam.html), [deedsBCV](https://junyuchen.me/MIR/models/deedsbcv.html) |
| TransMorph-LDDMM, LDDMMWrapper | Time-varying velocity fields (geodesic shooting) for any backbone | [LDDMM](https://junyuchen.me/MIR/models/lddmm.html) |
| MultiMorph, TemplateCreation | Atlas construction | [Atlas construction](https://junyuchen.me/MIR/models/multimorph.html) |

Conventions, transformation models, regularization, losses, metrics and a complete training recipe are
explained in the [user guide](https://junyuchen.me/MIR/user_guide/index.html).

## Tutorials

Runnable training, inference and evaluation scripts live under [`tutorials/`](tutorials/), one folder per
dataset and model (see the [tutorial catalogue](https://junyuchen.me/MIR/tutorials.html) and
[`tutorials/README.md`](tutorials/README.md)). Good starting points:

- [`tutorials/brain_registration`](tutorials/brain_registration): affine + deformable brain registration with pretrained weights.
- [`tutorials/IXI_benchmarking`](tutorials/IXI_benchmarking): benchmark seven models on IXI from a notebook.
- [`tutorials/lumir25_TransMorph`](tutorials/lumir25_TransMorph): a complete training template (Learn2Reg LUMIR25).
- [`tutorials/build_brain_template`](tutorials/build_brain_template): population template construction.

## Included projects

1. **Pretraining Deformable Image Registration Networks with Random Images**
   *Chen, Junyu, et al.* ***MIDL Short Papers***, *2025.*
   [[Paper](https://openreview.net/forum?id=NJANlZzxfi#discussion)] | [[Repo](https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs)]

2. **Correlation Ratio for Unsupervised Learning of Multi-modal Deformable Registration**
   *Chen, Xiaojian, et al.* ***SPIE Medical Imaging: Image Processing***, *2025.*
   [[Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13406/1340632/Correlation-ratio-for-unsupervised-learning-of-multi-modal-deformable-registration/10.1117/12.3047699.full)] | [[Repo](https://github.com/junyuchen245/Correlation_Ratio)]

3. **Unsupervised Learning of Spatially Varying Regularization for Diffeomorphic Image Registration**
   *Chen, Junyu, et al.* ***Medical Image Analysis***, *2025.*
   [[Paper](https://arxiv.org/abs/2412.17982)] | [[Repo](https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg)]

4. **Vector Field Attention for Deformable Image Registration**
   *Liu, Yihao, et al.* ***Journal of Medical Imaging***, *2024.*
   [[Paper](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-11/issue-6/064001/Vector-field-attention-for-deformable-image-registration/10.1117/1.JMI.11.6.064001.full)] | [[Repo](https://github.com/yihao6/vfa/)]

5. **Unsupervised Learning of Multi-modal Affine Registration for PET/CT**
   *Chen, Junyu, et al.* ***IEEE NSS/MIC/RTSD***, *2024.*
   [[Paper](https://arxiv.org/pdf/2409.13863)] | [[Repo](https://github.com/junyuchen245/Correlation_Ratio)]

6. **On Finite Difference Jacobian Computation in Deformable Image Registration**
   *Liu, Yihao, et al.* ***International Journal of Computer Vision***, *2024.*
   [[Paper](https://link.springer.com/article/10.1007/s11263-024-02047-1)] | [[Repo](https://github.com/yihao6/digital_diffeomorphism)]

7. **Deformable Cross-Attention Transformer for Medical Image Registration**
   *Chen, Junyu, et al.* ***MLMI Workshop***, *MICCAI 2023.*
   [[Paper](https://arxiv.org/abs/2303.06179)] | [[Repo](https://github.com/junyuchen245/TransMorph_DCA)]

8. **Unsupervised Learning of Diffeomorphic Image Registration via TransMorph**
   *Chen, Junyu, et al.* ***WBIR***, *2022.*
   [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-11203-4_11)] | [[Repo](https://github.com/junyuchen245/TransMorph_TVF)]

9. **TransMorph: Transformer for Unsupervised Medical Image Registration**
   *Chen, Junyu, et al.* ***Medical Image Analysis***, *2022.*
   [[Paper](https://arxiv.org/abs/2111.10480)] | [[Repo](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)]

## External packages wrapped

1. **Intensity Normalization Toolkit** -- *Reinhold, Jacob C., et al.* ***SPIE MI***, *2019.*
   [[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/)] | [[Repo](https://github.com/jcreinhold/intensity-normalization)]
2. **ConvexAdam: Dual-Optimization-Based 3D Registration** -- *Siebert, Hanna, et al.* ***IEEE TMI***, *2024.*
   [[Paper](https://ieeexplore.ieee.org/abstract/document/10681158)] | [[Repo](https://github.com/multimodallearning/convexAdam)]
3. **MultiMorph: On-demand Atlas Construction** -- *Abulnaga, S. Mazdak, et al.* ***CVPR***, *2025.*
   [[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Abulnaga_MultiMorph_On-demand_Atlas_Construction_CVPR_2025_paper.html)] | [[Repo](https://github.com/mabulnaga/multimorph)]
4. **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration** -- *Honkamaa, Joel, and Pekka Marttinen.* ***MELBA***, *2024.*
   [[Paper](https://www.melba-journal.org/papers/2024:026.html)] | [[Repo](https://github.com/honkamj/SITReg)]
5. **FireANTs** -- GPU greedy and SyN registration. [[Repo](https://github.com/rohitrango/fireants)]
6. **deedsBCV** -- [[Repo](https://github.com/mattiaspaul/deedsBCV)] (Linux only)
7. **LessNet** and **Fourier-Net / Fourier-Net+** -- *Jia, Xi, et al.* [[LessNet](https://github.com/xi-jia/LessNet)] | [[Fourier-Net](https://github.com/xi-jia/Fourier-Net)]

## Contributing

Issues and pull requests are welcome. See the
[contributing guide](https://junyuchen.me/MIR/development/contributing.html) for the development setup,
code style and the compatibility checks every change must pass, and the
[changelog](https://junyuchen.me/MIR/about/changelog.html) for recent changes.

## Citing

Please cite the paper of the model you use (listed above and on each model page) and link to this
repository.

## License

Apache 2.0. Vendored third-party code (for example `src/MIR/models/SITReg`) keeps its original license,
included alongside the code where the upstream project provides one.

### <a href="https://junyuchen245.github.io">About the author</a>
