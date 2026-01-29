<a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square"></a>
<img src="https://github.com/junyuchen245/MIR/actions/workflows/tests.yml/badge.svg" alt="Run Tests">
<img src="https://github.com/junyuchen245/MIR/actions/workflows/docs.yml/badge.svg" alt="Build and Deploy Docs">

## Overview

This repository hosts implementations of several projects I’ve contributed to, along with helpful preprocessing and post-processing utilities for medical image analysis. It also provides wrappers for selected external packages to enable seamless integration into **TransMorph-like registration workflows**.

Auto-generated documentation is available at: [https://junyuchen245.github.io/MIR](https://junyuchen245.github.io/MIR)

## Installation

**Prerequisite:** Install a CUDA-enabled PyTorch build before installing MIR.

### Editable install (recommended)

```
git clone https://github.com/junyuchen245/MIR.git
cd MIR
python3.8 -m pip install -U pip
python3.8 -m pip install -e .
```

### PyTorch CUDA build (required)

Install a CUDA build of PyTorch following the official instructions, then install MIR:

```
python3.8 -m pip install -e .
```

## Quick start: IXI benchmarking

You can quickly benchmark several models on the IXI dataset using the scripts in
[tutorials/IXI_benchmarking](tutorials/IXI_benchmarking). See the README there for
dataset download and run commands.

## 📄 Included Projects

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

---

## 🔌 External Wrappers Included

1. **Intensity Normalization Toolkit**  
   *Reinhold, Jacob C., et al. ***SPIE MI***, 2019.*  
   [[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/)] | [[Repo](https://github.com/jcreinhold/intensity-normalization)]

2. **ConvexAdam: Dual-Optimization-Based 3D Registration**  
   *Siebert, Hanna, et al. ***IEEE TMI***, 2024.*  
   [[Paper](https://ieeexplore.ieee.org/abstract/document/10681158)] | [[Repo](https://github.com/multimodallearning/convexAdam)]

3. **MultiMorph: On-demand Atlas Construction**  
   *Abulnaga, S. Mazdak, et al. ***CVPR***, 2025.*  
   [[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Abulnaga_MultiMorph_On-demand_Atlas_Construction_CVPR_2025_paper.html)] | [[Repo](https://github.com/mabulnaga/multimorph)]

4. **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration.**\
   *Honkamaa, Joel, and Pekka Marttinen. ***MELBA***, 2024.*\
   [[Paper](https://www.melba-journal.org/papers/2024:026.html)] | [[Repo](https://github.com/honkamj/SITReg)]
---


### <a href="https://junyuchen245.github.io"> About Me</a>