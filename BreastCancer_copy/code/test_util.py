import os
import math
import xlwt
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from medpy import metric
import torch.nn.functional as F


def test_all_case_1channel(net, image_list, data_root_path, img_prefix, label_prefix,
                           exp_name, num_classes, patch_size, stride_x, stride_y, stride_z,
                           save_result, test_save_path, use_mirror_ensemble):
    all_case_dice = []
    all_case_jc = []
    all_case_asd = []
    all_case_95hd = []
    for index, image_id in enumerate(image_list):
        image_path = os.path.join(data_root_path, img_prefix, img_prefix + '_' + image_id)
        affine = nib.load(image_path).affine
        image = nib.load(image_path).get_fdata()
        label_path = os.path.join(data_root_path, label_prefix, label_prefix + '_' + image_id)
        label = nib.load(label_path).get_fdata()
        name = image_id.split('.')[0]

        prediction, score_map = test_single_case_1channel(net, image, stride_x, stride_y, stride_z, patch_size, num_classes)

        if use_mirror_ensemble:
            prediction_flip_reverse_list = []
            for flip_axis in range(3):
                image_flip = np.flip(image, axis=flip_axis)
                prediction_flip, _ = test_single_case_1channel(net, image_flip, stride_x, stride_y, stride_z, patch_size, num_classes)
                prediction_flip_reverse = np.flip(prediction_flip, axis=flip_axis)
                prediction_flip_reverse_list.append(prediction_flip_reverse)
            # ensemble
            prediction_flip_reverse_sum = 0
            for i in range(3):
                prediction_flip_reverse_sum += prediction_flip_reverse_list[i]
            prediction += prediction_flip_reverse_sum
            prediction = (prediction >= 2).astype(np.uint8)

        if prediction.max() == 1:
            single_case_dc = metric.binary.dc(prediction, label)
            single_case_jc = metric.binary.jc(prediction, label)
            single_case_asd = metric.binary.asd(prediction, label)
            single_case_hd95 = metric.binary.hd95(prediction, label)
        else:
            single_case_dc = 0
            single_case_jc = 0
            single_case_asd = np.nan
            single_case_hd95 = np.nan
        all_case_dice.append(single_case_dc)
        all_case_jc.append(single_case_jc)
        all_case_asd.append(single_case_asd)
        all_case_95hd.append(single_case_hd95)


        print('%2d/%d %s: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (index + 1, len(image_list), name,
                                                                        single_case_dc, single_case_jc,
                                                                        single_case_asd, single_case_hd95))

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), test_save_path + "/%spred.nii.gz" % name)
            # nib.save(nib.Nifti1Image(image.astype(np.float32), affine), test_save_path + "/%simg.nii.gz" % name)
            nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), test_save_path + "/%sgt.nii.gz" % name)

    mean_dice = np.nanmean(np.array(all_case_dice))
    mean_jc = np.nanmean(np.array(all_case_jc))
    mean_asd = np.nanmean(np.array(all_case_asd))
    mean_95hd = np.nanmean(np.array(all_case_95hd))

    std_dice = np.nanstd(np.array(all_case_dice))
    std_jc = np.nanstd(np.array(all_case_jc))
    std_asd = np.nanstd(np.array(all_case_asd))
    std_95hd = np.nanstd(np.array(all_case_95hd))

    print('mean: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (mean_dice, mean_jc, mean_asd, mean_95hd))
    print('std:  Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (std_dice, std_jc, std_asd, std_95hd))

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('result', cell_overwrite_ok=True)
    i = 1
    worksheet.write(0, 0, 'name')
    worksheet.write(0, 1, 'Dice')
    worksheet.write(0, 2, 'Jaccard')
    worksheet.write(0, 3, 'ASD')
    worksheet.write(0, 4, '95HD')
    for img_, dice, jc, asd, hd95 in zip(image_list, all_case_dice, all_case_jc, all_case_asd, all_case_95hd):
        img_name = os.path.basename(img_).split('.')[0]
        worksheet.write(i, 0, img_name)
        worksheet.write(i, 1, round(dice, 4))
        worksheet.write(i, 2, round(jc, 4))
        worksheet.write(i, 3, round(asd, 2))
        worksheet.write(i, 4, round(hd95, 2))
        i += 1
    worksheet.write(i, 0, 'mean')
    worksheet.write(i, 1, round(mean_dice, 4))
    worksheet.write(i, 2, round(mean_jc, 4))
    worksheet.write(i, 3, round(mean_asd, 2))
    worksheet.write(i, 4, round(mean_95hd, 2))

    worksheet.write(i + 1, 0, 'std')
    worksheet.write(i + 1, 1, round(std_dice, 4))
    worksheet.write(i + 1, 2, round(std_jc, 4))
    worksheet.write(i + 1, 3, round(std_asd, 2))
    worksheet.write(i + 1, 4, round(std_95hd, 2))
    workbook.save(os.path.join(test_save_path, 'origin_result_%s.xls' % exp_name))

    return mean_dice, mean_jc, mean_asd, mean_95hd


def test_single_case_1channel(net, image, stride_x, stride_y, stride_z, patch_size, num_classes=1):
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

    for x in tqdm(range(0, sx)):
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



def test_all_case_2channel(net, image_list, data_root_path, img1_prefix, img2_prefix, label_prefix,
                           exp_name, num_classes, patch_size, stride_x, stride_y, stride_z,
                           save_result, test_save_path, use_mirror_ensemble):
    all_case_dice = []
    all_case_jc = []
    all_case_asd = []
    all_case_95hd = []
    for index, image_id in enumerate(image_list):
        image1_path = os.path.join(data_root_path, img1_prefix, img1_prefix + '_' + image_id)
        affine = nib.load(image1_path).affine
        image1 = nib.load(image1_path).get_fdata()
        image2_path = os.path.join(data_root_path, img2_prefix, img2_prefix + '_' + image_id)
        image2 = nib.load(image2_path).get_fdata()
        label_path = os.path.join(data_root_path, label_prefix, label_prefix + '_' + image_id)
        label = nib.load(label_path).get_fdata()
        name = image_id.split('.')[0]

        prediction, score_map = test_single_case_2channel(net, image1, image2, stride_x, stride_y,
                                                          stride_z, patch_size, num_classes)
        # prediction, score_map, fixed_map, moving_map = test_single_case_2channel_debug(net, image1, image2, stride_x, stride_y,
        #                                                   stride_z, patch_size, num_classes)

        if use_mirror_ensemble:
            prediction_flip_reverse_list = []
            for flip_axis in range(3):
                image_flip1 = np.flip(image1, axis=flip_axis)
                image_flip2 = np.flip(image2, axis=flip_axis)
                prediction_flip, _ = test_single_case_2channel(net, image_flip1, image_flip2, stride_x, stride_y,
                                                               stride_z, patch_size, num_classes)
                prediction_flip_reverse = np.flip(prediction_flip, axis=flip_axis)
                prediction_flip_reverse_list.append(prediction_flip_reverse)
            # ensemble
            prediction_flip_reverse_sum = 0
            for i in range(3):
                prediction_flip_reverse_sum += prediction_flip_reverse_list[i]
            prediction += prediction_flip_reverse_sum
            prediction = (prediction >= 2).astype(np.uint8)

        if prediction.max() == 1:
            single_case_dc = metric.binary.dc(prediction, label)
            single_case_jc = metric.binary.jc(prediction, label)
            single_case_asd = metric.binary.asd(prediction, label)
            single_case_hd95 = metric.binary.hd95(prediction, label)
        else:
            single_case_dc = 0
            single_case_jc = 0
            single_case_asd = np.nan
            single_case_hd95 = np.nan
        all_case_dice.append(single_case_dc)
        all_case_jc.append(single_case_jc)
        all_case_asd.append(single_case_asd)
        all_case_95hd.append(single_case_hd95)

        print('%2d/%d %s: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (index + 1, len(image_list), name,
                                                                        single_case_dc, single_case_jc,
                                                                        single_case_asd, single_case_hd95))

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), test_save_path + "/%spred.nii.gz" % name)
            # nib.save(nib.Nifti1Image(image.astype(np.float32), affine), test_save_path + "/%simg.nii.gz" % name)
            nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), test_save_path + "/%sgt.nii.gz" % name)
            # nib.save(nib.Nifti1Image(fixed_map.astype(np.float32), affine), test_save_path + "/%s_fixed.nii.gz" % name)
            # nib.save(nib.Nifti1Image(moving_map.astype(np.float32), affine), test_save_path + "/%s_moving.nii.gz" % name)

    mean_dice = np.nanmean(np.array(all_case_dice))
    mean_jc = np.nanmean(np.array(all_case_jc))
    mean_asd = np.nanmean(np.array(all_case_asd))
    mean_95hd = np.nanmean(np.array(all_case_95hd))

    std_dice = np.nanstd(np.array(all_case_dice))
    std_jc = np.nanstd(np.array(all_case_jc))
    std_asd = np.nanstd(np.array(all_case_asd))
    std_95hd = np.nanstd(np.array(all_case_95hd))

    print('mean: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (mean_dice, mean_jc, mean_asd, mean_95hd))
    print('std:  Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (std_dice, std_jc, std_asd, std_95hd))

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('result', cell_overwrite_ok=True)
    i = 1
    worksheet.write(0, 0, 'name')
    worksheet.write(0, 1, 'Dice')
    worksheet.write(0, 2, 'Jaccard')
    worksheet.write(0, 3, 'ASD')
    worksheet.write(0, 4, '95HD')
    for img_, dice, jc, asd, hd95 in zip(image_list, all_case_dice, all_case_jc, all_case_asd, all_case_95hd):
        img_name = os.path.basename(img_).split('.')[0]
        worksheet.write(i, 0, img_name)
        worksheet.write(i, 1, round(dice, 4))
        worksheet.write(i, 2, round(jc, 4))
        worksheet.write(i, 3, round(asd, 2))
        worksheet.write(i, 4, round(hd95, 2))
        i += 1
    worksheet.write(i, 0, 'mean')
    worksheet.write(i, 1, round(mean_dice, 4))
    worksheet.write(i, 2, round(mean_jc, 4))
    worksheet.write(i, 3, round(mean_asd, 2))
    worksheet.write(i, 4, round(mean_95hd, 2))

    worksheet.write(i + 1, 0, 'std')
    worksheet.write(i + 1, 1, round(std_dice, 4))
    worksheet.write(i + 1, 2, round(std_jc, 4))
    worksheet.write(i + 1, 3, round(std_asd, 2))
    worksheet.write(i + 1, 4, round(std_95hd, 2))
    workbook.save(os.path.join(test_save_path, 'origin_result_%s.xls' % exp_name))

    return mean_dice, mean_jc, mean_asd, mean_95hd


def test_single_case_2channel(net, image1, image2, stride_x, stride_y, stride_z, patch_size, num_classes=1):
    w, h, d = image1.shape

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
        image1 = np.pad(image1, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image2 = np.pad(image2, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image1.shape

    sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image1.shape).astype(np.float32)
    cnt = np.zeros(image1.shape).astype(np.float32)

    for x in tqdm(range(0, sx)):
        xs = min(stride_x*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_y*y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch1 = image1[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch2 = image2[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(
                    np.concatenate((np.expand_dims(test_patch1, axis=0), np.expand_dims(test_patch2, axis=0)), axis=0),
                    axis=0).astype(np.float32)
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


def test_single_case_2channel_debug(net, image1, image2, stride_x, stride_y, stride_z, patch_size, num_classes=1):
    w, h, d = image1.shape

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
        image1 = np.pad(image1, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image2 = np.pad(image2, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image1.shape

    sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image1.shape).astype(np.float32)
    fixed_map = np.zeros(image1.shape).astype(np.float32)
    moving_map = np.zeros(image1.shape).astype(np.float32)
    cnt = np.zeros(image1.shape).astype(np.float32)

    for x in tqdm(range(0, sx)):
        xs = min(stride_x*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_y*y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch1 = image1[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch2 = image2[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(
                    np.concatenate((np.expand_dims(test_patch1, axis=0), np.expand_dims(test_patch2, axis=0)), axis=0),
                    axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    seg_output_list, FA_list, flow = net(test_patch)
                    # ensemble
                    y = F.softmax(seg_output_list[0], dim=1)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                fixed = FA_list[0].cpu().data.numpy()
                fixed = fixed[0, :, :, :, :]
                moving = FA_list[1].cpu().data.numpy()
                moving = moving[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                fixed_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = fixed_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + fixed
                moving_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = moving_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + moving
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    fixed_map = fixed_map / cnt
    moving_map = moving_map / cnt
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        fixed_map = fixed_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        moving_map = moving_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map, fixed_map, moving_map


def test_all_case_3channel(net, image_list, data_root_path, img1_prefix, img2_prefix, img3_prefix, label_prefix,
                           exp_name, num_classes, patch_size, stride_x, stride_y, stride_z,
                           save_result, test_save_path, use_mirror_ensemble):
    all_case_dice = []
    all_case_jc = []
    all_case_asd = []
    all_case_95hd = []
    for index, image_id in enumerate(image_list):
        image1_path = os.path.join(data_root_path, img1_prefix, img1_prefix + '_' + image_id)
        affine = nib.load(image1_path).affine
        image1 = nib.load(image1_path).get_fdata()
        image2_path = os.path.join(data_root_path, img2_prefix, img2_prefix + '_' + image_id)
        image2 = nib.load(image2_path).get_fdata()
        image3_path = os.path.join(data_root_path, img3_prefix, img3_prefix + '_' + image_id)
        image3 = nib.load(image3_path).get_fdata()
        label_path = os.path.join(data_root_path, label_prefix, label_prefix + '_' + image_id)
        label = nib.load(label_path).get_fdata()
        name = image_id.split('.')[0]

        prediction, score_map = test_single_case_3channel(net, image1, image2, image3, stride_x, stride_y,
                                                          stride_z, patch_size, num_classes)

        if use_mirror_ensemble:
            prediction_flip_reverse_list = []
            for flip_axis in range(3):
                image_flip1 = np.flip(image1, axis=flip_axis)
                image_flip2 = np.flip(image2, axis=flip_axis)
                image_flip3 = np.flip(image3, axis=flip_axis)
                prediction_flip, _ = test_single_case_3channel(net, image_flip1, image_flip2, image_flip3, stride_x, stride_y,
                                                               stride_z, patch_size, num_classes)
                prediction_flip_reverse = np.flip(prediction_flip, axis=flip_axis)
                prediction_flip_reverse_list.append(prediction_flip_reverse)
            # ensemble
            prediction_flip_reverse_sum = 0
            for i in range(3):
                prediction_flip_reverse_sum += prediction_flip_reverse_list[i]
            prediction += prediction_flip_reverse_sum
            prediction = (prediction >= 2).astype(np.uint8)

        if prediction.max() == 1:
            single_case_dc = metric.binary.dc(prediction, label)
            single_case_jc = metric.binary.jc(prediction, label)
            single_case_asd = metric.binary.asd(prediction, label)
            single_case_hd95 = metric.binary.hd95(prediction, label)
        else:
            single_case_dc = 0
            single_case_jc = 0
            single_case_asd = np.nan
            single_case_hd95 = np.nan
        all_case_dice.append(single_case_dc)
        all_case_jc.append(single_case_jc)
        all_case_asd.append(single_case_asd)
        all_case_95hd.append(single_case_hd95)

        print('%2d/%d %s: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (index + 1, len(image_list), name,
                                                                        single_case_dc, single_case_jc,
                                                                        single_case_asd, single_case_hd95))

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), test_save_path + "/%spred.nii.gz" % name)
            # nib.save(nib.Nifti1Image(image.astype(np.float32), affine), test_save_path + "/%simg.nii.gz" % name)
            nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), test_save_path + "/%sgt.nii.gz" % name)

    mean_dice = np.nanmean(np.array(all_case_dice))
    mean_jc = np.nanmean(np.array(all_case_jc))
    mean_asd = np.nanmean(np.array(all_case_asd))
    mean_95hd = np.nanmean(np.array(all_case_95hd))

    std_dice = np.nanstd(np.array(all_case_dice))
    std_jc = np.nanstd(np.array(all_case_jc))
    std_asd = np.nanstd(np.array(all_case_asd))
    std_95hd = np.nanstd(np.array(all_case_95hd))

    print('mean: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (mean_dice, mean_jc, mean_asd, mean_95hd))
    print('std:  Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (std_dice, std_jc, std_asd, std_95hd))

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('result', cell_overwrite_ok=True)
    i = 1
    worksheet.write(0, 0, 'name')
    worksheet.write(0, 1, 'Dice')
    worksheet.write(0, 2, 'Jaccard')
    worksheet.write(0, 3, 'ASD')
    worksheet.write(0, 4, '95HD')
    for img_, dice, jc, asd, hd95 in zip(image_list, all_case_dice, all_case_jc, all_case_asd, all_case_95hd):
        img_name = os.path.basename(img_).split('.')[0]
        worksheet.write(i, 0, img_name)
        worksheet.write(i, 1, round(dice, 4))
        worksheet.write(i, 2, round(jc, 4))
        worksheet.write(i, 3, round(asd, 2))
        worksheet.write(i, 4, round(hd95, 2))
        i += 1
    worksheet.write(i, 0, 'mean')
    worksheet.write(i, 1, round(mean_dice, 4))
    worksheet.write(i, 2, round(mean_jc, 4))
    worksheet.write(i, 3, round(mean_asd, 2))
    worksheet.write(i, 4, round(mean_95hd, 2))

    worksheet.write(i + 1, 0, 'std')
    worksheet.write(i + 1, 1, round(std_dice, 4))
    worksheet.write(i + 1, 2, round(std_jc, 4))
    worksheet.write(i + 1, 3, round(std_asd, 2))
    worksheet.write(i + 1, 4, round(std_95hd, 2))
    workbook.save(os.path.join(test_save_path, 'origin_result_%s.xls' % exp_name))

    return mean_dice, mean_jc, mean_asd, mean_95hd


def test_single_case_3channel(net, image1, image2, image3, stride_x, stride_y, stride_z, patch_size, num_classes=1):
    w, h, d = image1.shape

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
        image1 = np.pad(image1, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image2 = np.pad(image2, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image3 = np.pad(image3, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image1.shape

    sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image1.shape).astype(np.float32)
    cnt = np.zeros(image1.shape).astype(np.float32)

    for x in tqdm(range(0, sx)):
        xs = min(stride_x*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_y*y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch1 = image1[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch2 = image2[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch3 = image3[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(
                    np.concatenate((np.expand_dims(test_patch1, axis=0),
                                    np.expand_dims(test_patch2, axis=0),
                                    np.expand_dims(test_patch3, axis=0)), axis=0), axis=0).astype(np.float32)
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

