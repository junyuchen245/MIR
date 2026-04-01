import glob
import subprocess
import os
import nibabel as nib
import numpy as np
img_dir = '/scratch2/jchen/DATA/data_dsa_xa_ct_new/'
output_dir = img_dir + 'stripped/'
os.makedirs(output_dir, exist_ok=True)

for img_i in glob.glob(img_dir+"*0216*.nii"):
    img_name = img_i.split('/')[-1].split('.ni')[0]
    print(img_name)
    img_nib = nib.load(img_i).get_fdata()
    #img_nib = np.transpose(img_nib, (2,1,0))
    img_nib = np.flip(img_nib, axis=1)
    nib_img = nib.Nifti1Image(img_nib, np.eye(4))
    nib.save(nib_img, output_dir+img_name+'.nii.gz')
    subprocess.call('python3.8 ./synthstrip-docker -i {} -o {} -m {}'.format(output_dir+img_name+'.nii.gz', output_dir+img_name+'_stripped.nii.gz', output_dir+img_name+'_mask.nii.gz'), shell=True)