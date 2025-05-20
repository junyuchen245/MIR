# Pretraining Deformable Image Registration Networks with Random Images

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

keywords: Image Registration, Self-supervised Learning, Pretraining

This is a **PyTorch** implementation of my paper:

[Chen, Junyu, et al. "Pretraining Deformable Image Registration Networks with Random Images." Medical Imaging with Deep Learning. 2025.](https://openreview.net/forum?id=NJANlZzxfi)

The core idea of this work is to leverage randomly generated images to initialize (or pretrain) the encoder of an image registration network. To achieve this, we designed temporary lightweight decoders that are attached to the encoder of the registration DNN, and pretrained the resulting network using a standard image registration loss function on the task of aligning pairs of random images.\
This approach is implemented in the [MIR package](https://github.com/junyuchen245/MIR). The source code for generating random images can be found [here](https://github.com/junyuchen245/MIR/tree/main/MIR/random_image_generation), and the lightweight decoder is implemented [here](https://github.com/junyuchen245/MIR/blob/main/MIR/models/Selfsupervised_Learning_Heads.py). The repository also includes training and inference scripts to reproduce the results reported in the paper.

This README mirrors the original repository for this work -> [GitHub Repo](https://github.com/junyuchen245/Pretraining_Image_Registration_DNNs)