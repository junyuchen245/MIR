"""TransMorph-LDDMM: TransMorph with a time-varying velocity field (LDDMM) parameterization.

Instead of a stationary velocity field (SVF) integrated by scaling-and-squaring, the
transformation is the end point of a flow

    d phi_t / dt = v_t o phi_t,   phi_0 = id,   t in [0, 1]

with v_t living in a reproducing-kernel Hilbert space V defined by a smoothing kernel K
(the inverse of a differential operator L).  Two parameterizations are offered:

* ``param='shooting'`` (geodesic shooting / EPDiff).  The network predicts the initial
  momentum m_0 at t=0.  The momentum is transported by the flow using the conservation law
      m_t = |D psi_t^{-1}| (D psi_t^{-1})^T  m_0 o psi_t^{-1},
  the velocity is recovered through v_t = K m_t and the map is updated by composition.
  The regularizer is the kinetic energy ||v_0||_V^2 = <m_0, K m_0>, which is conserved
  along the geodesic.

* ``param='tvf'`` (relaxation, Beg et al. 2005).  The network predicts a field p_t at every
  time step (conditioned on the currently warped moving image); the velocity is
  v_t = K p_t, and the regularizer is the path energy  sum_t dt <p_t, K p_t>.

The generic kernel/integrator live in ``MIR.models.lddmm``; ``LDDMMWrapper`` there applies the
same idea to any other backbone (VFA, VoxelMorph, SITReg, ...).  Both variants are end-to-end differentiable, so TransMorph and the LDDMM flow are
trained jointly rather than used as a two-stage "predict then refine" pipeline.

Conventions follow the rest of MIR: a displacement field ``u`` warps the moving image as
``mov(x + u(x))`` (``SpatialTransformer``), and ``psi = id + u`` corresponds to the LDDMM
flow phi_1.  The inverse map ``id + w`` is integrated alongside and returned as well.

References:
    - Beg, Miller, Trouve, Younes. Computing large deformation metric mappings via geodesic
      flows of diffeomorphisms. IJCV 2005.
    - Vialard, Risser, Rueckert, Cotter. Diffeomorphic 3D image registration via geodesic
      shooting using an efficient adjoint calculation. IJCV 2012.
    - Zhang & Fletcher. Fast diffeomorphic image registration via Fourier-approximated Lie
      algebras (FLASH). IJCV 2019.
    - Yang, Kwitt, Styner, Niethammer. Quicksilver: Fast predictive image registration.
      NeuroImage 2017.
    - Chen et al. TransMorph: Transformer for unsupervised medical image registration.
      MedIA 2022.

Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University

"""

import torch
import torch.nn as nn
import MIR.models.Swin_Transformer as swin
import MIR.models.registration_utils as utils
from MIR.models.TransMorph import Conv3dReLU, DecoderBlock, RegistrationHead


# Kernel, integrator and geometry helpers live in MIR.models.lddmm (re-exported here).
from MIR.models.lddmm import (  # noqa: F401 -- several of these are re-exported from this module
    FourierKernel,
    SpatiallyVaryingKernel,
    LDDMMIntegrator,
    jacobian_fd,
    det3x3,
    load_backbone_weights,
)
from MIR import _amp


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TransMorphLDDMM(nn.Module):
    """TransMorph with an LDDMM (time-varying velocity field) transformation model.

    Args:
        config: TransMorph configuration (see ``configs_TransMorph``); ``config.img_size`` is
            the size of the network input.
        param: ``'shooting'`` (predict initial momentum, geodesic shooting) or ``'tvf'``
            (predict a velocity field at every time step, relaxation).
        time_steps: Number of integration steps of the flow.
        kernel, alpha, gamma, power, sigma, kernel_weights: See :class:`FourierKernel`.
            Kernel parameters are expressed in network-input voxels. Ignored when ``sigmas`` is given.
        sigmas: Gaussian standard deviations of a spatially varying kernel (region-specific LDDMM,
            the LDDMM counterpart of TransMorph-SPR). A small head on the decoder features predicts
            per-voxel kernel weights; the output dict then also contains ``kernel_weights`` and the
            ``omt`` penalty to add to the loss. ``None`` = stationary kernel.
        weight_smooth_sigma, omt_power: See :class:`SpatiallyVaryingKernel`.
        spr: ``None``, ``'kernel'`` or ``'energy'``: add the TransMorph-SPR head (``sigma_0..2`` convs
            + sigmoid, one weight map ``lam`` in (0, 1]) on the decoder features.  ``'kernel'`` maps
            ``lam`` to the per-voxel kernel scale of the spatially varying kernel (``sigmas`` required,
            1 = smoothest); ``'energy'`` weights the kinetic-energy integrand by ``lam`` with a
            stationary kernel.  The output dict then contains ``spr_weight`` for the Beta prior.
        int_scale: Integrate the flow on a grid ``int_scale`` times finer than the network
            input (``'shooting'`` only). The returned fields are at that resolution.
        swin_type: Only ``'swin'`` is supported.

    Forward inputs:
        inputs: Tuple ``(mov, fix)`` of tensors [B, 1, D, H, W].
        return_dict: If True, return a dict with ``flow``, ``inv_flow``, ``energy``,
            ``energies`` and ``momentum`` (shooting) / ``pre_velocities`` (tvf), plus
            ``kernel_weights`` and ``omt`` with a spatially varying kernel.

    Forward outputs:
        Dense displacement field [B, 3, D*int_scale, H*int_scale, W*int_scale] (or a dict).

    """

    def __init__(
        self,
        config,
        param='shooting',
        time_steps=7,
        kernel='helmholtz',
        alpha=2.0,
        gamma=1.0,
        power=2,
        sigma=2.0,
        kernel_weights=None,
        sigmas=None,
        weight_smooth_sigma=1.0,
        omt_power=2.0,
        spr=None,
        int_scale=1,
        swin_type='swin',
    ):
        super().__init__()
        if param not in ('shooting', 'tvf'):
            raise ValueError(f"param must be 'shooting' or 'tvf', got {param}")
        if swin_type != 'swin':
            raise ValueError('TransMorphLDDMM currently supports swin_type="swin" only')
        if param == 'tvf' and int_scale != 1:
            raise ValueError("int_scale > 1 is only supported with param='shooting'")
        self.param = param
        self.time_steps = int(time_steps)
        self.int_scale = int(int_scale)
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        self.img_size = tuple(config.img_size)
        embed_dim = config.embed_dim
        self.transformer = swin.SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
        )
        self.up0 = DecoderBlock(
            embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if self.if_transskip else 0, use_batchnorm=False
        )
        self.up1 = DecoderBlock(
            embed_dim * 2, embed_dim, skip_channels=embed_dim if self.if_transskip else 0, use_batchnorm=False
        )
        self.up2 = DecoderBlock(
            embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if self.if_transskip else 0, use_batchnorm=False
        )
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        if self.param == 'shooting':
            self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
            self.up3 = DecoderBlock(
                embed_dim // 2,
                config.reg_head_chan,
                skip_channels=config.reg_head_chan if self.if_convskip else 0,
                use_batchnorm=False,
            )
            self.reg_head = RegistrationHead(in_channels=config.reg_head_chan, out_channels=3, kernel_size=3)
        else:
            self.cs = nn.ModuleList()
            self.up3s = nn.ModuleList()
            self.reg_heads = nn.ModuleList()
            for _ in range(self.time_steps):
                self.cs.append(Conv3dReLU(2, embed_dim // 2, 3, 1))
                self.up3s.append(
                    DecoderBlock(
                        embed_dim // 2,
                        config.reg_head_chan,
                        skip_channels=embed_dim // 2 if self.if_convskip else 0,
                        use_batchnorm=False,
                    )
                )
                self.reg_heads.append(RegistrationHead(in_channels=config.reg_head_chan, out_channels=3, kernel_size=3))
            self.spatial_trans = utils.SpatialTransformer(self.img_size)

        # Kernel parameters are given in network-input voxels; rescale them to the integration grid.
        s = self.int_scale
        if spr not in (None, 'kernel', 'energy'):
            raise ValueError("spr must be None, 'kernel' or 'energy'")
        if spr == 'kernel' and sigmas is None:
            raise ValueError("spr='kernel' needs the sigmas of a spatially varying kernel")
        self.spr = spr
        if spr is not None:  # TransMorph-SPR head
            self.sigma_0 = Conv3dReLU(config.reg_head_chan, embed_dim // 2, 3, 1, use_batchnorm=False)
            self.sigma_1 = Conv3dReLU(embed_dim // 2, embed_dim // 2, 3, 1, use_batchnorm=False)
            self.sigma_2 = nn.Conv3d(embed_dim // 2, 1, kernel_size=3, padding=1)
            nn.init.zeros_(self.sigma_2.weight)
            nn.init.constant_(self.sigma_2.bias, 2.0)  # lam = sigmoid(2) = 0.88 at initialization
            self.wts_act = nn.Sigmoid()
            self.eps = 1e-6
        if sigmas is not None and spr == 'kernel':
            self.kernel = SpatiallyVaryingKernel(sigmas, weight_smooth_sigma, omt_power).scaled(s)
            self.wts_head = None
        elif sigmas is not None:
            self.kernel = SpatiallyVaryingKernel(sigmas, weight_smooth_sigma, omt_power).scaled(s)
            self.wts_head = nn.Sequential(
                Conv3dReLU(config.reg_head_chan, config.reg_head_chan, 3, 1, use_batchnorm=False),
                nn.Conv3d(config.reg_head_chan, len(sigmas), 3, padding=1),
            )
            nn.init.zeros_(self.wts_head[-1].weight)
            bias = torch.zeros(len(sigmas))
            bias[-1] = 3.0  # start with ~87% of the weight on the smoothest kernel
            with torch.no_grad():
                self.wts_head[-1].bias.copy_(bias)
        else:
            self.kernel = FourierKernel(
                kernel=kernel, alpha=alpha, gamma=gamma, power=power, sigma=sigma, weights=kernel_weights
            ).scaled(s)
            self.wts_head = None
        self.int_size = tuple(d * s for d in self.img_size)
        self.integrator = LDDMMIntegrator(self.int_size, self.time_steps, self.kernel)

    @property
    def spatially_varying(self) -> bool:
        """Whether ``self.kernel`` is a :class:`SpatiallyVaryingKernel`."""
        return isinstance(self.kernel, SpatiallyVaryingKernel)

    def _spr_weight(self, x):
        """TransMorph-SPR weight map lam in (eps, 1] from decoder features (None without SPR)."""
        if self.spr is None:
            return None
        lam = self.wts_act(self.sigma_2(self.sigma_1(self.sigma_0(x))))
        return torch.clamp(lam, self.eps, 1.0)

    def _weights(self, x, lam=None):
        """Kernel weights w_0 from decoder features / SPR map (None with a stationary kernel)."""
        if not self.spatially_varying:
            return None
        if self.spr == 'kernel':
            return self.kernel.weights_from_spr(lam)
        return self.kernel.weights_from_logits(self.wts_head(x))

    def encode(self, mov, fix):
        """Shared TransMorph encoder/decoder up to the half-resolution feature map."""
        x_cat = torch.cat((mov, fix), dim=1)
        out_feats = self.transformer(x_cat)
        f3 = self.c1(self.avg_pool(x_cat)) if self.if_convskip else None
        if self.if_transskip:
            f1, f2 = out_feats[-2], out_feats[-3]
        else:
            f1 = f2 = None
        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        return x

    def forward(self, inputs, return_dict=False):
        """Predict the momentum (shooting) or per-step pre-velocities (tvf) and integrate the flow.

        Args:
            inputs: Tuple ``(mov, fix)`` of tensors [B, 1, D, H, W] with ``(D, H, W) == config.img_size``.
            return_dict: If True, return the full output dict (see the class docstring) instead of the flow.

        Returns:
            The forward displacement field ``flow`` [B, 3, D*int_scale, H*int_scale, W*int_scale] or, with
            ``return_dict=True``, the integrator output dict extended with ``momentum`` (shooting) or
            ``pre_velocities`` (tvf) and, when applicable, ``kernel_weights``, ``omt`` and ``spr_weight``.
        """
        mov, fix = inputs
        feats = self.encode(mov, fix)
        if self.param == 'shooting':
            f4 = self.c2(torch.cat((mov, fix), dim=1)) if self.if_convskip else None
            x = self.up3(feats, f4)
            m0 = self.reg_head(x)
            lam = self._spr_weight(x)
            w0 = self._weights(x, lam)
            if self.int_scale > 1:
                m0 = (
                    _amp.interpolate(m0, scale_factor=self.int_scale, mode='trilinear', align_corners=False)
                    * self.int_scale
                )
                if w0 is not None:
                    w0 = _amp.interpolate(w0, scale_factor=self.int_scale, mode='trilinear', align_corners=False)
                if lam is not None:
                    lam = _amp.interpolate(lam, scale_factor=self.int_scale, mode='trilinear', align_corners=False)
            out = self.integrator.shoot(m0, weights=w0, energy_weight=lam if self.spr == 'energy' else None)
            out['momentum'] = m0
        else:
            pre_vel = []
            x0 = self.up3s[0](feats, self.cs[0](torch.cat((mov, fix), dim=1)))
            lam = self._spr_weight(x0)
            w0 = self._weights(x0, lam)

            def predict(t, u):
                if t == 0:
                    x_t = x0
                else:
                    def_x = self.spatial_trans(mov, u)
                    x_t = self.up3s[t](feats, self.cs[t](torch.cat((def_x, fix), dim=1)))
                p_t = self.reg_heads[t](x_t)
                pre_vel.append(p_t)
                return p_t

            out = self.integrator.relax(predict, weights=w0, energy_weight=lam if self.spr == 'energy' else None)
            out['pre_velocities'] = pre_vel
        if w0 is not None:
            out['kernel_weights'] = w0
            if self.spr != 'kernel':
                out['omt'] = self.kernel.omt(w0)
        if lam is not None:
            out['spr_weight'] = lam
        if return_dict:
            return out
        return out['flow']
