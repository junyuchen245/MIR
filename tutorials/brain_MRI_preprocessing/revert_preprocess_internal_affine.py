import argparse
import base64
import io
import json
import os

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

from MIR.models import AffineReg3D


def reorient_image_to_match(reference_nii, target_nii):
    reference_ornt = nib.aff2axcodes(reference_nii.affine)
    target_reoriented = nib.as_closest_canonical(target_nii, enforce_diag=False)
    target_ornt = nib.aff2axcodes(target_reoriented.affine)

    if target_ornt != reference_ornt:
        ornt_trans = nib.orientations.ornt_transform(
            nib.io_orientation(target_reoriented.affine),
            nib.io_orientation(reference_nii.affine),
        )
        target_reoriented = target_reoriented.as_reoriented(ornt_trans)
    return target_reoriented


def resampling(img_npy, img_pixdim, tar_pixdim, order, mode="constant"):
    if order == 0:
        img_npy = img_npy.astype(np.uint16)
    img_npy = zoom(
        img_npy,
        (
            img_pixdim[0] / tar_pixdim[0],
            img_pixdim[1] / tar_pixdim[1],
            img_pixdim[2] / tar_pixdim[2],
        ),
        order=order,
        prefilter=False,
        mode=mode,
    )
    return img_npy


def _load_original_from_meta(meta):
    original_shape = meta.get("original_shape")
    original_affine = np.array(meta.get("original_affine"), dtype=np.float64)
    original_pixdim = meta.get("original_pixdim")
    header_b64 = meta.get("original_header")
    if original_shape is None or original_affine is None or original_pixdim is None:
        raise ValueError("--meta must include original_shape, original_affine, original_pixdim")
    original_header = None
    if header_b64 is not None:
        header_bytes = base64.b64decode(header_b64)
        original_header = nib.Nifti1Header.from_fileobj(io.BytesIO(header_bytes))
    original_nib = nib.Nifti1Image(
        np.zeros(original_shape, dtype=np.float32), original_affine, header=original_header
    )
    original_nib.header.set_zooms(original_pixdim)
    return original_nib, original_affine, original_header, original_pixdim


def main():
    parser = argparse.ArgumentParser(
        description="Revert internal affine preprocessing back to original image space (except intensity normalization)."
    )
    parser.add_argument("processed_path", help="Path to processed image in template space (NIfTI)")
    parser.add_argument("--out", dest="output_path", required=True, help="Path to save reverted image (NIfTI)")
    parser.add_argument("--mask", dest="mask_path", default=None, help="Path to mask associated with processed image")
    parser.add_argument("--mask-out", dest="mask_output_path", default=None, help="Path to save reverted mask")
    parser.add_argument("--meta", dest="meta_path", required=True, help="Path to preprocessing metadata JSON")
    parser.add_argument("--interp-order", type=int, default=2, help="Interpolation order for image resampling")
    parser.add_argument(
        "--resample-order", type=int, default=2, help="Interpolation order for spacing resampling"
    )
    parser.add_argument("--org", dest="original_path", default=None, help="Original image path (optional)")
    parser.add_argument("--template", dest="template_path", default=None, help="Template path (optional)")
    parser.add_argument("--device", default="cpu", help="Device to run affine inversion (cpu or cuda:0)")
    args = parser.parse_args()

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if args.template_path is None:
        args.template_path = meta.get("template_path")

    if args.template_path is None:
        raise ValueError("template_path is required unless provided in --meta")

    affine_matrix = meta.get("affine_matrix")
    if affine_matrix is None:
        raise ValueError("affine_matrix not found in meta. Re-run preprocessing to save it.")

    processed_nib = nib.load(args.processed_path)
    template_nib = nib.load(args.template_path) if os.path.exists(args.template_path) else None

    if args.original_path is not None:
        original_nib = nib.load(args.original_path)
        original_header = original_nib.header.copy()
        original_affine = original_nib.affine.copy()
        original_pixdim = original_nib.header.structarr["pixdim"][1:-4]
    else:
        original_nib, original_affine, original_header, original_pixdim = _load_original_from_meta(meta)

    if template_nib is None:
        template_shape = meta.get("template_shape")
        template_affine = np.array(meta.get("template_affine"), dtype=np.float64)
        if template_shape is None or template_affine is None:
            raise ValueError("--meta must include template_shape and template_affine")
        template_nib = nib.Nifti1Image(np.zeros(template_shape, dtype=np.float32), template_affine)

    original_reoriented = reorient_image_to_match(template_nib, original_nib)

    tar_pixdim = [1.0, 1.0, 1.0]
    if meta.get("target_pixdim") is not None:
        tar_pixdim = meta.get("target_pixdim")

    original_reoriented_npy = original_reoriented.get_fdata()
    ref_resampled_npy = resampling(original_reoriented_npy, original_pixdim, tar_pixdim, order=2)

    device = torch.device(args.device)
    reg = AffineReg3D(vol_shape=processed_nib.shape, match_fixed=True).to(device)
    processed_torch = torch.from_numpy(processed_nib.get_fdata()).unsqueeze(0).unsqueeze(0).float()
    processed_torch = processed_torch.to(device)

    affine_tensor = torch.tensor(affine_matrix, dtype=processed_torch.dtype, device=device)
    inv_warped = reg.apply_affine(
        processed_torch,
        affine_tensor,
        invert=True,
        target_shape=ref_resampled_npy.shape,
    )
    inv_npy = inv_warped.squeeze(0).squeeze(0).detach().cpu().numpy()

    reverted_npy = resampling(inv_npy, tar_pixdim, original_pixdim, order=args.resample_order)

    reverted_nib = nib.Nifti1Image(reverted_npy, original_reoriented.affine, header=original_reoriented.header)
    reverted_nib = reorient_image_to_match(original_nib, reverted_nib)

    out_header = original_header if original_header is not None else original_nib.header.copy()
    qform_code = out_header["qform_code"]
    sform_code = out_header["sform_code"]
    if hasattr(qform_code, "__len__"):
        qform_code = int(np.array(qform_code).ravel()[0])
    if hasattr(sform_code, "__len__"):
        sform_code = int(np.array(sform_code).ravel()[0])
    out_header.set_qform(original_affine, code=int(qform_code))
    out_header.set_sform(original_affine, code=int(sform_code))
    out_nib = nib.Nifti1Image(reverted_nib.get_fdata(), original_affine, header=out_header)
    nib.save(out_nib, args.output_path)

    if args.mask_path is not None:
        mask_nib = nib.load(args.mask_path)
        mask_torch = torch.from_numpy(mask_nib.get_fdata()).unsqueeze(0).unsqueeze(0).float().to(device)
        inv_mask = reg.apply_affine(
            mask_torch,
            affine_tensor,
            invert=True,
            target_shape=ref_resampled_npy.shape,
        )
        inv_mask_npy = inv_mask.squeeze(0).squeeze(0).detach().cpu().numpy()
        reverted_mask_npy = resampling(inv_mask_npy, tar_pixdim, original_pixdim, order=0)

        reverted_mask_nib = nib.Nifti1Image(
            reverted_mask_npy, original_reoriented.affine, header=original_reoriented.header
        )
        reverted_mask_nib = reorient_image_to_match(original_nib, reverted_mask_nib)

        mask_out_header = out_header.copy()
        mask_out_header.set_qform(original_affine, code=int(qform_code))
        mask_out_header.set_sform(original_affine, code=int(sform_code))

        mask_output_path = args.mask_output_path
        if mask_output_path is None:
            mask_output_path = args.output_path.replace(".nii.gz", ".mask.nii.gz")
            if mask_output_path == args.output_path:
                mask_output_path = f"{args.output_path}.mask.nii.gz"

        out_mask_nib = nib.Nifti1Image(
            reverted_mask_nib.get_fdata(), original_affine, header=mask_out_header
        )
        nib.save(out_mask_nib, mask_output_path)


if __name__ == "__main__":
    main()
