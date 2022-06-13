import os
from glob import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import label
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm
import xlwt


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


def localization_metrics(prediction_path_list):
    # all case result value
    all_cases_result = {}
    all_cases_fp_num = 0
    all_cases_tumors_num = 0
    all_cases_tumors_hit_num = 0
    all_cases_tumors_dice_list = []
    all_cases_tumors_jc_list = []
    all_cases_tumors_asd_list = []
    all_cases_tumors_95hd_list = []

    for prediction_path in tqdm(prediction_path_list):
        # single case result value
        single_result = {}
        FP_num = 0
        hit_num = 0

        # cal volume per voxel
        # print(prediction_path)
        prediction_sitk = sitk.ReadImage(prediction_path)
        volume_per_voxel = float(np.prod(prediction_sitk.GetSpacing(), dtype=np.float64))

        # load target
        target_path = prediction_path.replace('pred', 'gt')
        target = nib.load(target_path).get_fdata()
        # if size < 300, do not count as tumor
        target, _, _ = remove_all_but_the_largest_connected_component(target, [1], volume_per_voxel, {1: 125})
        # cal how many tumors
        # object id range(1,.....)
        # 0 is background
        target_map, target_object_num = label(target.astype(int))
        all_cases_tumors_num += target_object_num
        target_object_size_dict = {}
        # if size < 300, do not count as tumor
        for object_id in range(1, target_object_num+1):
            target_object_size = (target_map == object_id).sum() * volume_per_voxel
            target_object_size_dict[object_id] = target_object_size
        # if len(target_object_size_dict) > 1:
        #     print(os.path.basename(target_path))

        # load prediction
        prediction = nib.load(prediction_path).get_fdata()
        # remove small connected component
        prediction, _, _ = remove_all_but_the_largest_connected_component(prediction, [1], volume_per_voxel, {1: 30})
        # cal how many tumors of prediction
        prediction_map, prediction_object_num = label(prediction.astype(int))

        # {target_object_id: [prediction_object_id, ... ](maybe could have more than one prediction_object id)}
        hit_dict = {}
        # cal hit_num
        # one target object -> all prediction object
        for target_object_id in range(1, target_object_num+1):
            hit_dict[target_object_id] = []
            target_single_object = (target_map == (target_object_id)).astype(np.float64)
            hit_flag = False
            for prediction_object_id in range(1, prediction_object_num+1):
                prediction_single_object = (prediction_map == (prediction_object_id)).astype(np.float64)
                dice = metric.binary.dc(prediction_single_object, (target_single_object*prediction_single_object).astype(np.float64))
                # if dice > threshold, prediction object match to target object
                # and means hit the target object successfully
                if dice > 0:
                    hit_flag = True
                    hit_dict[target_object_id].append(prediction_object_id)
            if hit_flag:
                hit_num += 1
        all_cases_tumors_hit_num += hit_num

        # cal FP_num
        # if prediction_object_id not in hit_dict, means it is a FP
        for prediction_object_id in range(1, prediction_object_num+1):
            is_FP = True
            for hit_list in hit_dict.values():
                if prediction_object_id in hit_list:
                    is_FP = False
                    break
            if is_FP:
                FP_num += 1
                all_cases_fp_num += 1

        # cal Dice
        dice_dict = {}
        jc_dict = {}
        asd_dict = {}
        hd95_dict = {}
        for target_object_id, prediction_object_list in hit_dict.items():
            target_single_object = (target_map == target_object_id).astype(np.float64)
            prediction_match_target_object = np.zeros_like(target_single_object)
            for prediction_object_id in prediction_object_list:
                prediction_single_object = (prediction_map == prediction_object_id).astype(np.float64)
                prediction_match_target_object += prediction_single_object
            if prediction_match_target_object.max() > 1:
                raise ValueError(os.path.basename(prediction_path) + ': prediction_match_target_object.max() > 1')
            # cal metrics
            if prediction_match_target_object.max() == 1:
                dice = metric.binary.dc(prediction_match_target_object, target_single_object)
                dice = round(dice, 4)
                dice_dict[target_object_id] = dice
                all_cases_tumors_dice_list.append(dice)

                jc = metric.binary.jc(prediction_match_target_object, target_single_object)
                jc = round(jc, 4)
                jc_dict[target_object_id] = jc
                all_cases_tumors_jc_list.append(jc)

                asd = metric.binary.asd(prediction_match_target_object, target_single_object)
                asd = round(asd, 4)
                asd_dict[target_object_id] = asd
                all_cases_tumors_asd_list.append(asd)

                hd95 = metric.binary.hd95(prediction_match_target_object, target_single_object)
                hd95 = round(hd95, 4)
                hd95_dict[target_object_id] = hd95
                all_cases_tumors_95hd_list.append(hd95)
            else:
                dice_dict[target_object_id] = 0
                jc_dict[target_object_id] = 0
                asd_dict[target_object_id] = 'NaN'
                hd95_dict[target_object_id] = 'NaN'

        if hit_num != target_object_num:
            print('missed: ', os.path.basename(prediction_path))
            single_result['Hit'] = False
        else:
            single_result['Hit'] = True
        single_result['P_num'] = target_object_num
        single_result['TP_num'] = hit_num
        single_result['FP_num'] = FP_num
        single_result['Dice'] = dice_dict
        single_result['Jaccard'] = jc_dict
        single_result['ASD'] = asd_dict
        single_result['95HD'] = hd95_dict
        all_cases_result[os.path.basename(prediction_path).replace('pred.nii.gz', '')] = single_result

    # mean_dice = round(np.mean(all_cases_tumors_dice_list), 4)
    sensitivity = round(all_cases_tumors_hit_num / all_cases_tumors_num, 4)

    mean = {}
    mean['Dice'] = np.round(np.mean(all_cases_tumors_dice_list), 4)
    mean['Jaccard'] = np.round(np.mean(all_cases_tumors_jc_list), 4)
    mean['ASD'] = np.round(np.mean(all_cases_tumors_asd_list), 4)
    mean['95HD'] = np.round(np.mean(all_cases_tumors_95hd_list), 4)
    std = {}
    std['Dice'] = np.round(np.std(all_cases_tumors_dice_list, ddof=1), 4)
    std['Jaccard'] = np.round(np.std(all_cases_tumors_jc_list, ddof=1), 4)
    std['ASD'] = np.round(np.std(all_cases_tumors_asd_list, ddof=1), 4)
    std['95HD'] = np.round(np.std(all_cases_tumors_95hd_list, ddof=1), 4)


    return all_cases_result, mean, std, sensitivity, all_cases_fp_num


def write_to_xls(cases_result:dict, mean, std, sensitivity, save_path, filename):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('result', cell_overwrite_ok=True)
    style = xlwt.XFStyle()
    al = xlwt.Alignment()
    al.horz = 0x02
    al.vert = 0x01
    style.alignment = al
    worksheet.write(0, 0, 'name', style)
    worksheet.write(0, 1, 'P_num', style)
    worksheet.write(0, 2, 'TP_num', style)
    worksheet.write(0, 3, 'Hit', style)
    worksheet.write(0, 4, 'FP_num', style)
    worksheet.write(0, 5, 'Dice', style)
    worksheet.write(0, 6, 'Jaccard', style)
    worksheet.write(0, 7, 'ASD', style)
    worksheet.write(0, 8, '95HD', style)

    i = 1
    for key, value in cases_result.items():
        worksheet.write(i, 0, key, style)
        worksheet.write(i, 1, value['P_num'], style)
        worksheet.write(i, 2, value['TP_num'], style)
        worksheet.write(i, 3, str(value['Hit']), style)
        worksheet.write(i, 4, value['FP_num'], style)
        worksheet.write(i, 5, str(value['Dice']), style)
        worksheet.write(i, 6, str(value['Jaccard']), style)
        worksheet.write(i, 7, str(value['ASD']), style)
        worksheet.write(i, 8, str(value['95HD']), style)
        i += 1

    worksheet.write(i, 0, 'mean', style)
    worksheet.write(i, 5, mean['Dice'], style)
    worksheet.write(i, 6, mean['Jaccard'], style)
    worksheet.write(i, 7, mean['ASD'], style)
    worksheet.write(i, 8, mean['95HD'], style)
    i += 1

    worksheet.write(i, 0, 'std', style)
    worksheet.write(i, 5, std['Dice'], style)
    worksheet.write(i, 6, std['Jaccard'], style)
    worksheet.write(i, 7, std['ASD'], style)
    worksheet.write(i, 8, std['95HD'], style)

    worksheet.write(i+1, 0, 'sensitivity', style)
    worksheet.write(i+1, 1, sensitivity, style)

    workbook.save(os.path.join(save_path, 'hit_result_%s.xls' % filename))


if __name__ == '__main__':
    exp_name = 'BreastROI_down0.5XY_2channel_sub2_fold0'
    root_path = '../data/inference/' + exp_name + '/fold_0/best'
    prediction_path_list = sorted(glob(root_path + '/*pred*'))
    cases_result, mean, std, sensitivity = localization_metrics(prediction_path_list)
    write_to_xls(cases_result, mean, std, sensitivity, root_path, exp_name + '_target125_predict30')
    print(cases_result)