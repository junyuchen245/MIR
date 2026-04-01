import numpy as np
import nibabel as nib
import os
import glob
import sys, ants
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.measure import label
from skimage.filters import threshold_otsu
from scipy import ndimage
from ants import resample_image_to_target

def crop_or_pad(volume: np.ndarray,
                target_shape,
                pad_value: float = 0) -> np.ndarray:
    """
    Center-crop or center-pad a NumPy array to exactly match target_shape.

    Args:
        volume: Input array of shape (d0, d1, ..., dN).
        target_shape: Desired shape (t0, t1, ..., tN).
        pad_value: Constant value to use for padding.

    Returns:
        A new array of shape target_shape.
    """
    output = volume
    ndim = output.ndim

    if ndim != len(target_shape):
        raise ValueError(f"volume.ndim={ndim} but target_shape has length {len(target_shape)}")

    for axis, tgt in enumerate(target_shape):
        cur = output.shape[axis]

        if cur < tgt:
            # need to pad
            total_pad = tgt - cur
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width = [(0, 0)] * ndim
            pad_width[axis] = (pad_before, pad_after)
            output = np.pad(output,
                            pad_width=pad_width,
                            mode="constant",
                            constant_values=pad_value)

        elif cur > tgt:
            # need to crop
            start = (cur - tgt) // 2
            end = start + tgt
            slicer = [slice(None)] * ndim
            slicer[axis] = slice(start, end)
            output = output[tuple(slicer)]

        # if cur == tgt, nothing to do on this axis

    return output

def zoom_img(img, pixel_dims, order=3):
    img_pixdim, tar_pix = pixel_dims
    ratio = np.array(img_pixdim) / np.array(tar_pix)
    img = zoom(img, ratio, order=order)
    return img

def norm_img(img):
    img[img < -300] = -300
    img[img > 300] = 300
    norm = (img - img.min()) / (img.max() - img.min())
    norm[norm>1] = 1
    norm[norm<0] = 0
    return norm

def crop_img(img, tar_sz):
    size_diff = np.array(img.shape) - np.array(tar_sz)
    img = img[size_diff[0] // 2:size_diff[0] // 2 + tar_sz[0], size_diff[1] // 2:size_diff[1] // 2 + tar_sz[1],
             size_diff[2] // 2:size_diff[2] // 2 + tar_sz[2]]
    return img

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def remove_bed(img):
    img_ = img.copy()
    img_ = (img_-img_.min())/(img_.max()-img_.min())
    threshold_global_otsu = threshold_otsu(img_)
    mask = img >= -800#threshold_global_otsu
    #mask = (mask + seg_simp>0)>0
    mask = ndimage.binary_erosion(mask).astype(mask.dtype)
    mask = ndimage.binary_fill_holes(mask).astype(mask.dtype)
    mask = ndimage.binary_dilation(mask).astype(mask.dtype)
    mask3D = getLargestCC(mask)
    mask3D = ndimage.binary_dilation(mask3D).astype(mask3D.dtype)
    mask3D = ndimage.binary_closing(mask3D).astype(mask3D.dtype)
    img[mask3D == 0] = np.percentile(img, 0.5)
    return img

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

data_dir = '/scratch2/jchen/PSMA_JHU/JHU/'
data_ctseg_dir = '/scratch2/jchen/PSMA_JHU/JHU_CT_seg/'
out_dir = '/scratch2/jchen/PSMA_JHU/Preprocessed/JHU/'
CT_atlas = '/scratch/jchen/python_projects/AutoPET/atlas/ct/TransMorphAtlas_MAE_1_MS_1_diffusion_1/dsc0.5425.nii.gz'
proxy = nib.load(CT_atlas)
CT_atlas = proxy.get_fdata()
CT_atlas[CT_atlas>1] = 1
CT_atlas[CT_atlas<0] = 0
#print(CT_atlas.max(), CT_atlas.min())
y_ct_py = ants.from_numpy(CT_atlas)
target_pixdim = [2.8, 2.8, 3.8]
tar_sz = [192, 192, 256]


for img in glob.glob(data_dir + '*_SUVBW.nii.gz'):
    print(img)
    pat_name = img.split('_SUVBW')[0].split('/')[-1]
    print(pat_name)
    pat_CT = data_dir + pat_name + '_CT.nii.gz'
    pat_CT_seg = data_ctseg_dir + pat_name + '_CT_seg.nii.gz'
    pat_SUV = data_dir + pat_name + '_SUVBW.nii.gz'
    pat_SUV_seg = data_dir + pat_name + '_SUV_Seg.nii.gz'
    proxy = nib.load(pat_SUV)
    img_suv = proxy.get_fdata()
    header_info = proxy.header
    suv_pixdim = header_info['pixdim'][1:4]

    proxy = nib.load(pat_SUV_seg)
    seg_suv = proxy.get_fdata()
    proxy = nib.load(pat_CT)
    img_ct = proxy.get_fdata()
    header_info = proxy.header
    proxy = nib.load(pat_CT_seg)
    seg_ct = proxy.get_fdata()

    ct_pixdim = header_info['pixdim'][1:4]
    img_ct = zoom_img(img_ct, (ct_pixdim, target_pixdim), 3)
    img_suv = zoom_img(img_suv, (suv_pixdim, target_pixdim), 3)
    seg_suv = zoom_img(seg_suv, (suv_pixdim, target_pixdim), 0)
    seg_ct = zoom_img(seg_ct, (ct_pixdim, target_pixdim), 0)
    img_suv = crop_or_pad(img_suv, img_ct.shape, img_suv.min())
    seg_suv = crop_or_pad(seg_suv, img_ct.shape, seg_suv.min())
    #img_ct = crop_img(img_ct, tar_sz)
    #img_suv = crop_img(img_suv, tar_sz)
    #seg_suv = crop_img(seg_suv, tar_sz)
    print(img_suv.shape, img_ct.shape)
    img_ct = remove_bed(img_ct).astype(np.float32)
    img_suv = img_suv.astype(np.float32)

    #plt.figure()
    #plt.subplot(1, 4, 1)
    #plt.imshow(img_suv[:, img_suv.shape[1]//2, :], cmap='gray', vmin=0, vmax=4)
    #plt.imshow(img_ct[:, img_ct.shape[1] // 2, :], cmap='gray', alpha=0.4)
    #plt.subplot(1, 4, 2)
    #plt.imshow(img_ct[:, img_ct.shape[1] // 2, :], cmap='gray')
    #plt.subplot(1, 4, 3)
    #plt.imshow(seg_suv[:, img_ct.shape[1] // 2, :])
    #plt.subplot(1, 4, 4)
    #plt.imshow(seg_ct[:, img_ct.shape[1] // 2, :])
    #plt.show()
    img_ct_norm = norm_img(img_ct.copy())
    x_ct_py_norm = ants.from_numpy(img_ct_norm)
    x_ct_py = ants.from_numpy(img_ct.copy())
    x_ct_seg_py = ants.from_numpy(seg_ct.copy())
    x_suv_seg_py = ants.from_numpy(seg_suv.copy())
    x_suv_py = ants.from_numpy(img_suv.copy())

    reg12 = ants.registration(y_ct_py, x_ct_py_norm, 'Affine', reg_iterations=(500, 500, 500),
                              syn_metric='meansquares')

    x_ct_seg_def = ants.apply_transforms(fixed=y_ct_py,
                                         moving=x_ct_seg_py,
                                         transformlist=reg12['fwdtransforms'],
                                         interpolator='nearestNeighbor')

    x_suv_seg_def = ants.apply_transforms(fixed=y_ct_py,
                                             moving=x_suv_seg_py,
                                             transformlist=reg12['fwdtransforms'],
                                             interpolator='nearestNeighbor')

    x_suv_def = ants.apply_transforms(fixed=x_suv_seg_py,
                                      moving=x_suv_py,
                                      transformlist=reg12['fwdtransforms'],
                                      interpolator='bSpline', defaultvalue=np.percentile(img_suv, 0.5), verbose=True)

    x_ct_def = ants.apply_transforms(fixed=y_ct_py,
                                     moving=x_ct_py,
                                     transformlist=reg12['fwdtransforms'],
                                     interpolator='bSpline', defaultvalue=np.percentile(img_ct, 0.5))

    img_suv = x_suv_def.numpy()
    img_ct = x_ct_def.numpy()
    seg_ct = x_ct_seg_def.numpy()
    seg_suv = x_suv_seg_def.numpy()
    print(img_suv.shape, img_ct.shape, seg_suv.shape, seg_ct.shape)
    plt.figure(dpi=150)
    plt.subplot(1, 5, 1)
    plt.imshow(img_suv[:, img_suv.shape[1]//2, :], cmap='gray', vmin=0, vmax=4)
    plt.subplot(1, 5, 2)
    plt.imshow(img_ct[:, img_ct.shape[1] // 2, :], cmap='gray')
    plt.subplot(1, 5, 3)
    plt.imshow(seg_suv[:, img_ct.shape[1] // 2, :])
    plt.subplot(1, 5, 4)
    plt.imshow(seg_ct[:, img_ct.shape[1] // 2, :])
    plt.subplot(1, 5, 5)
    plt.imshow(CT_atlas[:, img_ct.shape[1] // 2, :], cmap='gray')
    plt.savefig('tmp.png')

    affine = make_affine_from_pixdim(target_pixdim)
    nib.save(nib.Nifti1Image(img_suv, affine), out_dir + pat_name + '_SUV.nii.gz')
    nib.save(nib.Nifti1Image(seg_suv, affine), out_dir + pat_name + '_SUV_seg.nii.gz')
    nib.save(nib.Nifti1Image(img_ct, affine), out_dir + pat_name + '_CT.nii.gz')
    nib.save(nib.Nifti1Image(seg_ct, affine), out_dir + pat_name + '_CT_seg.nii.gz')
    sys.exit(0)
    # sys.exit()

