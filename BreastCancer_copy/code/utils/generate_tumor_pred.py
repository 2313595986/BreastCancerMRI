import nibabel as nib
import os
import numpy as np


def crop_xyz_axis(volume_np):
    x = 0
    while (volume_np[x, :, :].max() < 1).all():
        x += 1
    start_x = x - 10
    if start_x < 0:
        start_x = 0

    x = volume_np.shape[0] - 1
    while (volume_np[x, :, :].max() < 1).all():
        x -= 1
    end_x = x + 10 + 1
    if end_x > volume_np.shape[0]:
        end_x = volume_np.shape[0]

    y = 0
    while (volume_np[:, y, :].max() < 1).all():
        y += 1
    start_y = y - 10
    if start_y < 0:
        start_y = 0

    y = volume_np.shape[1] - 1
    while (volume_np[:, y, :].max() < 1).all():
        y -= 1
    end_y = y + 10 + 1
    if end_y > volume_np.shape[1]:
        end_y = volume_np.shape[1]

    z = 0
    while (volume_np[:, :, z].max() < 1).all():
        z += 1
    start_z = z - 10
    if start_z < 0:
        start_z = 0

    z = volume_np.shape[2] - 1
    while (volume_np[:, :, z].max() < 1).all():
        z -= 1
    end_z = z + 10 + 1
    if end_z > volume_np.shape[2]:
        end_z = volume_np.shape[2]

    return [start_x, end_x, start_y, end_y, start_z, end_z]



root_path = r'/home/wh/datasets/BreastCancerMRI_185/CropZero_down0.5XY/Norm'
pred_root_path = r'/home/wh/datasets/BreastCancerMRI_185/BreastCancer_copy/data/inference'

exp_name = 'ResUnetMini_BreastROIdown0.5Z_1&sub_fold0_resunet'
image_index = '1670980'
fold_index = 'fold_0'

label_path = os.path.join(root_path, 'BreastMask', 'BreastMask_'+image_index+'.nii.gz')
pred_tumor_path = os.path.join(pred_root_path, exp_name, fold_index, 'best', image_index+'pred.nii.gz')
C1_image_path = os.path.join(root_path, 'DCE-C1', 'DCE-C1_'+image_index+'.nii.gz')
Tumor_mask_path = os.path.join(root_path, 'TumorMask', 'TumorMask_'+image_index+'.nii.gz')
save_path_1 = os.path.join(root_path, 'TumorPred_'+image_index+'.nii.gz')
save_path_2 = os.path.join(root_path, 'TumorMask_'+image_index+'.nii.gz')


pred_image = nib.load(pred_tumor_path)
label = nib.load(label_path)
C1_image = nib.load(C1_image_path)
Tumor_mask = nib.load(Tumor_mask_path)

# affine
C1_image_affine = C1_image.affine


# 数据
pred_image_np = pred_image.get_fdata()
label_np = label.get_fdata()
mask_np = Tumor_mask.get_fdata()

# box
breast_xyz_list = crop_xyz_axis(label_np)

orignal_size = label_np.shape
orignal_tumor = np.zeros(orignal_size)

for x_index, y in enumerate(pred_image_np):
    for y_index, z in enumerate(y):
        for z_index, pixel in enumerate(z):
            if pixel == 1:
                orignal_tumor[breast_xyz_list[0]+x_index, breast_xyz_list[2]+y_index, breast_xyz_list[4]+z_index] = 1

nib.save(nib.Nifti1Image(orignal_tumor, C1_image_affine), save_path_1)
nib.save(nib.Nifti1Image(mask_np, C1_image_affine), save_path_2)