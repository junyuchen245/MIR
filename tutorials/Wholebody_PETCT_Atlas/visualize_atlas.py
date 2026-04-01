import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.filters import threshold_otsu
from scipy import ndimage

def maximum_intensity_projection(image, axis):
    return np.max(image, axis=axis)

def mean_intensity_projection(image, axis):
    return np.sum(image, axis=axis)

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def remove_bed(img):
    img_ = img.copy()
    img_ = (img_-img_.min())/(img_.max()-img_.min())
    threshold_global_otsu = threshold_otsu(img_)
    mask = img >= 0#threshold_global_otsu
    #mask = (mask + seg_simp>0)>0
    mask = ndimage.binary_erosion(mask).astype(mask.dtype)
    mask = ndimage.binary_erosion(mask).astype(mask.dtype)
    mask = ndimage.binary_fill_holes(mask).astype(mask.dtype)
    mask3D = getLargestCC(mask)
    mask = ndimage.binary_dilation(mask).astype(mask.dtype)
    mask3D = ndimage.binary_dilation(mask3D).astype(mask3D.dtype)
    mask3D = ndimage.binary_closing(mask3D).astype(mask3D.dtype)
    img[mask3D == 0] = np.percentile(img, 0.5)
    return img

def transpose_flip_image(image):
    return np.flip(image.T, axis=0)

sex_flag = 'women'
base_path = "/scratch/jchen/python_projects/custom_packages/MIR/tutorials/Wholebody_PETCT_Atlas/wholebody_petct_sitreg_step1/atlas/"
ct_atlas = nib.load(base_path + "ct/{}/epoch0006_ssim0.9546.nii.gz".format(sex_flag))
fdg_atlas = nib.load(base_path + "pet/fdg_{}/epoch0006_ssim0.9546.nii.gz".format(sex_flag))
if sex_flag == 'men':
    psma_atlas = nib.load(base_path + "pet/psma_{}/epoch0006_ssim0.9546.nii.gz".format(sex_flag))
    psma_atlas = psma_atlas.get_fdata()
    psma_atlas = remove_bed(psma_atlas)
    
ct_atlas = ct_atlas.get_fdata()
fdg_atlas = fdg_atlas.get_fdata()
fdg_atlas = remove_bed(fdg_atlas)

plt.figure(figsize=(12, 6), dpi=150)
plt.subplot(1, 3, 1)
plt.title("CT Atlas - MIP")
plt.imshow(transpose_flip_image(mean_intensity_projection(ct_atlas, axis=1)), cmap='gray')
plt.subplot(1, 3, 2)
plt.title("FDG PET Atlas - MIP")
plt.imshow(transpose_flip_image(maximum_intensity_projection(fdg_atlas, axis=1)), cmap='gray_r', vmax=np.max(fdg_atlas)*0.25)
if sex_flag == 'men':
    plt.subplot(1, 3, 3)
    plt.title("PSMA PET Atlas - MIP")
    plt.imshow(transpose_flip_image(maximum_intensity_projection(psma_atlas, axis=1)), cmap='gray_r', vmax=np.max(psma_atlas)*0.7)
plt.savefig("wholebody_petct_atlas_mip0.png")
plt.close()

plt.figure(figsize=(12, 6), dpi=150)
plt.subplot(1, 3, 1)
plt.title("CT Atlas - MIP")
plt.imshow(transpose_flip_image(ct_atlas[:, ct_atlas.shape[1]//2+10, :]), cmap='gray')
plt.subplot(1, 3, 2)
plt.title("FDG PET Atlas - MIP")
plt.imshow(transpose_flip_image(fdg_atlas[:, fdg_atlas.shape[1]//2+10, :]), cmap='gray_r', vmax=np.max(fdg_atlas)*0.25)
if sex_flag == 'men':
    plt.subplot(1, 3, 3)
    plt.title("PSMA PET Atlas - MIP")
    plt.imshow(transpose_flip_image(psma_atlas[:, psma_atlas.shape[1]//2+10, :]), cmap='gray_r', vmax=np.max(psma_atlas)*0.6)
plt.savefig("wholebody_petct_atlas_slice.png")
plt.close()


plt.figure(figsize=(12, 6), dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(transpose_flip_image(ct_atlas[:, ct_atlas.shape[1]//2+10, :]), cmap='gray')
plt.imshow(transpose_flip_image(fdg_atlas[:, fdg_atlas.shape[1]//2+10, :]), cmap='hot', vmax=np.max(fdg_atlas)*0.25, alpha=0.3)
if sex_flag == 'men':
    plt.subplot(1, 2, 2)
    plt.imshow(transpose_flip_image(ct_atlas[:, ct_atlas.shape[1]//2+10, :]), cmap='gray')
    plt.imshow(transpose_flip_image(psma_atlas[:, psma_atlas.shape[1]//2+10, :]), cmap='hot', vmax=np.max(psma_atlas)*0.7, alpha=0.3)
plt.savefig("wholebody_petct_atlas_overlay.png")
plt.close()
