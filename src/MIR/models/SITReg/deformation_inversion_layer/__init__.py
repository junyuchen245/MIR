"""Deformation inversion layer is a neural network layer for inverting deformation fields

The package has been developed as part of SITReg, a deep learning intra-modality
image registration arhitecture fulfilling strict symmetry properties.

Deformation inversion layer is published under MIT licence.

If you use deformation inversion layer, or other parts of the repository, please cite:

- **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration using deformation inversion layers**  
[Joel Honkamaa](https://github.com/honkamj "Joel Honkamaa"), Pekka Marttinen  
Under review ([eprint arXiv:2303.10211](https://arxiv.org/abs/2303.10211 "eprint arXiv:2303.10211"))
"""

from .fixed_point_invert_deformation import fixed_point_invert_deformation, DeformationInversionArguments

__all__ = [
    "fixed_point_invert_deformation",
    "DeformationInversionArguments",
]