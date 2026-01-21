import glob
import subprocess, os, shutil

'''
data_dir = '/scratch2/jchen/PSMA_JHU/JHU/'
out_dir = '/scratch2/jchen/PSMA_JHU/JHU_CT_seg/'
begin = False
idx = 0

for img in glob.glob(data_dir + '*_SUVBW.nii.gz'):
    pat_name = img.split('/')[-1].split('_SUVBW')[0]
    print(pat_name)
    pat_CT = data_dir + pat_name + '_CT.nii.gz'
    idx += 1
    subprocess.call(["TotalSegmentator", "--ml", "-i", pat_CT, "-o", out_dir+pat_name+'_CT_seg.nii.gz'])
'''

data_dir = '/scratch2/jchen/DATA/PSMA_autoPET/CT/'
out_dir = '/scratch2/jchen/DATA/PSMA_autoPET/CT_seg/'
begin = False
idx = 0
name = 'psma_d4b471bab61342ff_2020-09-28'

for img in glob.glob(data_dir + '*_0000.nii.gz'):
    pat_name = img.split('/')[-1].split('_0000')[0]
    print(pat_name)
    if not begin:
        if pat_name != name:
            continue
        else:
            begin = True
    pat_CT = data_dir + pat_name + '_0000.nii.gz'
    idx += 1
    
    subprocess.call(["TotalSegmentator", "--ml", "-i", pat_CT, "-o", out_dir+pat_name+'_CT_seg.nii.gz'])