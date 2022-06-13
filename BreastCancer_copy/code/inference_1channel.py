"""
inference for unlabeled data
"""
import os
import torch
import math
import time
import codecs
import datetime
import codecs
from glob import glob
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from skimage.transform import resize
from scipy.ndimage import label
from scipy.ndimage import interpolation

from utils.util import get_test_list
from networks.nnResUnet_se import nnResUNet as BreastSegModel
from networks.nnResUnet_se import nnResUNet as TumorSegModel


def norm(array):
    mean = array.mean()
    std = array.std()
    return (array-mean)/std


def crop_y_axis_for_step1(volume_np):
    y = 0
    while (volume_np[:, y, :] < 50).all():
        y += 1
    start = y - 40
    if start < 0:
        start = 0
    crop_volume1_np = volume_np[:, start:, :]
    return crop_volume1_np, start


def crop_y_axis_for_step2(volume_np):
    y = 0
    while (volume_np[:, y, :] < 50).all():
        y += 1
    start = y - 10
    if start < 0:
        start = 0

    y = volume_np.shape[1] - 1
    while (volume_np[:, y, :] < 50).all():
        y -= 1
    end = y + 10 + 1
    if end > volume_np.shape[1]:
        end = volume_np.shape[1]

    crop_volume_np = volume_np[:, start:end, :]
    return crop_volume_np, start, end


def inference_window_slice_1channel(net, image, stride_x, stride_y, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_x*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_y*y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)[0]
                    # ensemble
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_root_path = '/raid/hra/dataset/BreastCancerMRI1000/YN_16/RawData'
    test_list = get_test_list('../data/test_16.txt')

    model1_path = '../model/end2end_1channel/BreastSegModel.pth'
    model2_path = '../model/end2end_1channel/TumorSegModel.pth'

    output_root_dir = '../data/YN20/'
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    # load model
    print('loading model...')
    start_time1 = time.time()
    model1 = BreastSegModel(1, 2).cuda()
    model1.load_state_dict(torch.load(model1_path))

    model2 = TumorSegModel(1, 2).cuda()
    model2.load_state_dict(torch.load(model2_path))

    # time
    end_time1 = time.time()
    used_time_loading_model = datetime.timedelta(seconds=end_time1-start_time1).seconds
    print('loading model done, used time=%ds\n' % used_time_loading_model)

    for test_id in test_list:
        print('-----------------------------------------')
        print('data index is %s' % os.path.basename(test_id).split('.')[0])
        output_path = os.path.join(output_root_dir, os.path.basename(test_id).split('.')[0])
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        c1_path = os.path.join(test_root_path, 'DCE-C1', 'DCE-C1_%s' % test_id)

        # load volume
        print('loading data...')
        start_time2 = time.time()
        c1_volume = nib.load(c1_path)
        c1_volume_np = c1_volume.get_fdata()

        affine = c1_volume.affine
        spacing = np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])
        volume_per_voxel = float(np.prod(spacing, dtype=np.float64))
        # time
        end_time2 = time.time()
        used_time_loading_data = datetime.timedelta(seconds=end_time2 - start_time2).seconds
        print('loading data done, used time=%ds\n' % used_time_loading_data)

        # pre-process
        # crop
        print('pre-processing data...')
        start_time3 = time.time()
        c1_volume_crop, crop_start_index_step1 = crop_y_axis_for_step1(c1_volume_np)

        # downsample (*0.36 for xy, 1 for z)
        origin_shape = c1_volume_crop.shape
        # new_shape = np.array(origin_shape)//2
        c1_volume_crop_down = interpolation.zoom(c1_volume_crop, (0.36, 0.36, 1), order=3)
        # norm
        c1_volume_crop_down_norm = norm(c1_volume_crop_down)
        # time
        end_time3 = time.time()
        used_time_preprocess_data = datetime.timedelta(seconds=end_time3 - start_time3).seconds
        print('pre-processing data done, used time=%ds\n' % used_time_preprocess_data)

        # run
        # generate breast mask
        print('generating breast mask...')
        start_time4 = time.time()
        breast_mask, _ = inference_window_slice_1channel(model1, c1_volume_crop_down_norm, 100, 100, 30, (256, 256, 64),
                                                         num_classes=2)

        # breast_mask upsample (/0.36 for xy)
        breast_mask = resize(breast_mask.astype(np.float32), origin_shape, order=0).astype(np.uint8)
        breast_mask_recover = np.zeros_like(c1_volume_np)
        breast_mask_recover[:, crop_start_index_step1:, :] = breast_mask
        nib.save(nib.Nifti1Image(breast_mask_recover.astype(np.uint8), affine),
                 os.path.join(output_path, 'BreastMask_%s' % test_id))

        # time
        end_time4 = time.time()
        used_time_breast_mask = datetime.timedelta(seconds=end_time4 - start_time4).seconds
        print('generating breast mask done, used time=%ds\n' % used_time_breast_mask)

        print('generating tumor mask...')
        start_time5 = time.time()
        # mask * img
        c1_breast_roi = breast_mask * c1_volume_crop
        # crop zero
        c1_breast_roi_crop, crop_start_index_step2, crop_end_index_step2 = crop_y_axis_for_step2(c1_breast_roi)
        # downsample c1 c0 volume (*0.5 for xy)
        c1_breast_roi_crop_down = interpolation.zoom(c1_breast_roi_crop, (0.5, 0.5, 1), order=3)
        # norm
        c1_breast_roi_crop_down_norm = norm(c1_breast_roi_crop_down)
        # generate tumor label
        tumor_mask, _ = inference_window_slice_1channel(model2, c1_breast_roi_crop_down_norm,
                                                        120, 40, 30, (288, 96, 80), num_classes=2)
        # recover original shape
        tumor_mask = resize(tumor_mask.astype(np.float32), c1_breast_roi_crop.shape, order=0).astype(np.uint8)
        tumor_reshape1 = np.zeros_like(c1_breast_roi)
        tumor_reshape1[:, crop_start_index_step2:crop_end_index_step2, :] = tumor_mask
        tumor_reshape2 = np.zeros_like(c1_volume_np)
        tumor_reshape2[:, crop_start_index_step1:, :] = tumor_reshape1
        tumor_final = tumor_reshape2
        nib.save(nib.Nifti1Image(tumor_final.astype(np.uint8), affine),
                 os.path.join(output_path, 'TumorMask_%s' % test_id))

        # time
        end_time5 = time.time()
        used_time_tumor_mask = datetime.timedelta(seconds=end_time5 - start_time5).seconds
        print('generating tumor mask done, used time=%ds\n' % used_time_tumor_mask)

        # time
        print('total inference used time=%ds' % datetime.timedelta(seconds=end_time5 - start_time2).seconds)

        # txt
        with codecs.open(os.path.join(output_path, test_id.split('.')[0] + '.txt'),
                         mode='w', encoding='utf-8') as f:
            f.write(test_id.split('.')[0] + '\n')
            f.write('load data used time=%ds\n' % used_time_loading_data)
            f.write('pre-preprocess data used time=%ds\n' % used_time_preprocess_data)
            f.write('generate breast mask used time=%ds\n' % used_time_breast_mask)
            f.write('generate tumor mask used time=%ds\n' % used_time_tumor_mask)
            f.write('total used time=%ds\n' % datetime.timedelta(seconds=end_time5 - start_time2).seconds)

