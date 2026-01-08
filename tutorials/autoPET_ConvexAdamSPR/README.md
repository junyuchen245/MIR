We show here that the idea of spatially varying regularization that was presented in our paper "[**Unsupervised Learning of Spatially Varying Regularization for Diffeomorphic Image Registration**](https://arxiv.org/abs/2412.17982)" can also be extended to instance-optimization.

The code is in `MIR/src/MIR/models/convexAdam/convex_adam_MIND_SPR.py`.

Conventional MIND-based ConvexAdam: `Best is trial 196 with value: 0.6402014079687036`\
MIND-based ConvexAdam with SPR: `Best is trial 91 with value: 0.6413333494330519`
