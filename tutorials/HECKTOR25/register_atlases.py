import numpy as np
import nibabel as nib
import ants
import matplotlib.pyplot as plt

hecktor_atlas = nib.load("/scratch/jchen/python_projects/custom_packages/MIR/tutorials/HECKTOR25/atlas/ct/SITRegAtlas_JHU_SSIM_1_MS_1_diffusion_1/ssim0.9744.nii.gz")
hecktor_seg = nib.load("/scratch/jchen/python_projects/custom_packages/MIR/tutorials/HECKTOR25/atlas/seg/SITRegAtlas_JHU_SSIM_1_MS_1_diffusion_1/ctseg_atlas_118lbls.nii.gz")

wholebody_atlas = nib.load("/scratch/jchen/python_projects/custom_packages/MIR/tutorials/Wholebody_PETCT_Atlas/wholebody_petct_sitreg_step1/atlas/ct/men/epoch0006_ssim0.9546.nii.gz")
wholebody_seg = nib.load('/scratch/jchen/python_projects/custom_packages/MIR/tutorials/Wholebody_PETCT_Atlas/wholebody_petct_sitreg_step1/atlas/seg/ct/men/seg_atlas_132lbls.nii.gz')

hecktor_atlas_np = hecktor_atlas.get_fdata()
hecktor_seg_np = hecktor_seg.get_fdata()
wholebody_atlas_np = wholebody_atlas.get_fdata()
wholebody_atlas_np[..., 0:145] = 0
wholebody_seg_np = wholebody_seg.get_fdata()

fix_img = ants.from_numpy(hecktor_atlas_np.astype(np.float32))
fix_seg = ants.from_numpy(hecktor_seg_np.astype(np.float32))
mov_img = ants.from_numpy(wholebody_atlas_np.astype(np.float32))
mov_seg = ants.from_numpy(wholebody_seg_np.astype(np.float32))

reg = ants.registration(
    fixed=fix_img,
    moving=mov_img,
    type_of_transform='TRSAA',
)
warped_mov = ants.apply_transforms(
    fixed=fix_img,
    moving=mov_img,
    transformlist=reg['fwdtransforms'],
)
warped_mov_seg = ants.apply_transforms(
    fixed=fix_seg,
    moving=mov_seg,
    transformlist=reg['fwdtransforms'],
    interpolator='nearestNeighbor',
)

#reg = ants.registration(
#    fixed=fix_img,
#    moving=warped_mov,
#    type_of_transform='SyN',
#)
#warped_mov = ants.apply_transforms(
#    fixed=fix_img,
#    moving=warped_mov,
#    transformlist=reg['fwdtransforms'],
#)

warped_mov_np = warped_mov.numpy()
warped_mov_seg_np = warped_mov_seg.numpy().astype(np.int32)

label_set = [124, 126, 127, 128, 129, 130, 131, 132]
for lbl in label_set:
    hecktor_seg_np[warped_mov_seg_np == lbl] = lbl

print(np.unique(hecktor_seg_np))

plt.figure(figsize=(12, 6))
plt.subplot(3, 3, 1)
plt.imshow(hecktor_atlas_np[hecktor_atlas_np.shape[0] // 2, ...], cmap='gray')
plt.subplot(3, 3, 2)
plt.imshow(warped_mov_np[warped_mov_np.shape[0] // 2, ...], cmap='gray')
plt.subplot(3, 3, 3)
plt.imshow(hecktor_atlas_np[hecktor_atlas_np.shape[0] // 2, ...], cmap='gray')
plt.imshow(warped_mov_np[warped_mov_np.shape[0] // 2, ...], cmap='hot', alpha=0.5)
plt.subplot(3, 3, 4)
plt.imshow(hecktor_atlas_np[:, hecktor_atlas_np.shape[1] // 2, :], cmap='gray')
plt.subplot(3, 3, 5)
plt.imshow(warped_mov_np[:, warped_mov_np.shape[1] // 2, :], cmap='gray')
plt.subplot(3, 3, 6)
plt.imshow(hecktor_atlas_np[:, hecktor_atlas_np.shape[1] // 2, :], cmap='gray')
plt.imshow(warped_mov_np[:, warped_mov_np.shape[1] // 2, :], cmap='hot', alpha=0.5)
plt.subplot(3, 3, 7)
plt.imshow(hecktor_atlas_np[:, :, hecktor_atlas_np.shape[2] // 2], cmap='gray')
plt.subplot(3, 3, 8)
plt.imshow(warped_mov_np[:, :, warped_mov_np.shape[2] // 2], cmap='gray')
plt.subplot(3, 3, 9)
plt.imshow(hecktor_atlas_np[:, :, hecktor_atlas_np.shape[2] // 2], cmap='gray')
plt.imshow(warped_mov_np[:, :, warped_mov_np.shape[2] // 2], cmap='hot', alpha=0.5)
plt.savefig("temp_registration.png")

merged_header = hecktor_seg.header.copy()
merged_header.set_data_dtype(np.uint16)
merged_header.set_slope_inter(1, 0)
new_hecktor_seg_nib = nib.Nifti1Image(hecktor_seg_np.astype(np.int16), hecktor_seg.affine, merged_header)
nib.save(new_hecktor_seg_nib, "/scratch/jchen/python_projects/custom_packages/MIR/tutorials/HECKTOR25/atlas/seg/SITRegAtlas_JHU_SSIM_1_MS_1_diffusion_1/ctseg_atlas_132lbls.nii.gz")