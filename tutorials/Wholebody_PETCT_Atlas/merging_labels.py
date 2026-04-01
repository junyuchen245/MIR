import numpy as np
import nibabel as nib

# new label starts from 124
sex = ['men', 'women']
for s in sex:
    mov_atlas = nib.load(f"/scratch/jchen/python_projects/custom_packages/MIR/tutorials/Wholebody_PETCT_Atlas/wholebody_petct_sitreg_step1/atlas/seg/ct/{s}/seg_atlas_125lbls.nii.gz")
    new_labels_fdg = nib.load(f"/scratch/jchen/python_projects/custom_packages/MIR/tutorials/Wholebody_PETCT_Atlas/wholebody_petct_sitreg_step1/atlas/seg/pet/fdg_{s}/Segmentation-label.nii.gz")
    new_labels_ct = nib.load(f"/scratch/jchen/python_projects/custom_packages/MIR/tutorials/Wholebody_PETCT_Atlas/wholebody_petct_sitreg_step1/atlas/seg/ct/{s}/Segmentation-Thymus-label.nii.gz")

    mov_atlas_data = np.asanyarray(mov_atlas.dataobj).copy().astype(np.uint16)
    new_labels_fdg_data = np.asanyarray(new_labels_fdg.dataobj).astype(np.int16)
    new_labels_ct_data = np.asanyarray(new_labels_ct.dataobj).astype(np.int16)

    print("Unique labels in original atlas:", np.unique(mov_atlas_data))
    print("Unique labels in new FDG segmentation:", np.unique(new_labels_fdg_data))
    print("Unique labels in new CT segmentation:", np.unique(new_labels_ct_data))

    mov_atlas_data[new_labels_ct_data == 1] = 124

    for lbl in np.unique(new_labels_fdg_data):
        if lbl == 0:
            continue
        new_label = int(lbl) + 124
        print(new_label)
        mov_atlas_data[new_labels_fdg_data == lbl] = new_label

    print("Unique labels in merged atlas:", np.unique(mov_atlas_data))

    # Save the merged atlas
    merged_header = mov_atlas.header.copy()
    merged_header.set_data_dtype(np.uint16)
    merged_header.set_slope_inter(1, 0)
    merged_atlas_nib = nib.Nifti1Image(mov_atlas_data, mov_atlas.affine, merged_header)
    nib.save(merged_atlas_nib, f"/scratch/jchen/python_projects/custom_packages/MIR/tutorials/Wholebody_PETCT_Atlas/wholebody_petct_sitreg_step1/atlas/seg/ct/{s}/seg_atlas_132lbls.nii.gz")
