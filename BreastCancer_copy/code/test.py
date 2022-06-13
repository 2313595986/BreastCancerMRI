import os
import torch
import argparse
import shutil
import sched
import time
from glob import glob
from utils import util
from networks.nnResUnet import nnResUNet as BaselineModel_res
from networks.nnResUnet_se import nnResUNet_SE as BaselineModel_res_se
from networks.UNet3D import UNet3D_DS as BaselineModel_unet
from networks.UNet3D_se import UNet3D_DS_SE as BaselineModel_unet_se
from networks.VNet import VNet_DS as BaselineModel_vnet
from networks.VNet_se import VNet_DS_SE as BaselineModel_vnet_se
from networks.VNet import ResVNet_DS as BaselineModel_resvnet
# from networks.nnResUnet_ShareEncoder_mini_FAlowResBeforeSE_ker3 import nnResUNet as ShareEncoderModel
# from networks.nnResUnet_ShareEncoder_mini_FAlowResBeforeSE_ker3 import nnResUNet as ShareEncoderFAlowModel
# from networks.nnResUnet_ShareEncoder_mini_FAhighResBeforeSE_ker3 import nnResUNet as ShareEncoderFAhighModel
# from networks.nnResUnet_ShareEncoder_mini_FAOriginResBeforeSE_ker3 import nnResUNet as ShareEncoderFAoriginModel
from test_util import test_all_case_1channel, test_all_case_2channel, test_all_case_3channel
from test_localization import localization_metrics, write_to_xls


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--model_type', type=str, default='best', choices=['best', 'iter_num'])
parser.add_argument('--fold_root_path', type=str, default='../data/', help='fold txt path')
parser.add_argument('--data_root_path', type=str, default='/home/wh/datasets/BreastCancerMRI_185/CropZero_down0.5XY/Norm/', help='data path')
parser.add_argument("--img1_prefix", type=str, default='CROPDCE-C0-615', help="img prefix")
parser.add_argument("--img2_prefix", type=str, default='CROPDCE-C1-185', help="img prefix")
parser.add_argument("--img3_prefix", type=str, default='CROPSubtraction-185', help="img prefix")
parser.add_argument("--label_prefix", type=str, default='CROPTumorMask-185', help="label prefix")
parser.add_argument('--model_root_path', type=str, default='../model/', help='model root path')
parser.add_argument('--backbone', type=str, default='unet_1',
                    choices=['resunet_1', 'resunet_2', 'unet_1',
                             'unet_2', 'vnet_1', 'vnet_2', 'unet_2_se', 'vnet_2_se', 'resunet_2_se'])
parser.add_argument('--exp_name', type=str, default='tumor_whole_c1_fold1_unet', help='experiment name')
parser.add_argument('--fold_index', type=int, default=5, help='index of fold')
parser.add_argument('--iter_num', type=int,  default=38000, help='model iteration')
parser.add_argument('--save_result', type=bool, default=True, help='save result?')
parser.add_argument('--use_mirror_ensemble', type=bool, default=False, help='use mirror for ensemble?')
args = parser.parse_args()


def test_calculate_metric(model_type, model_root_path, data_root_path, img1_prefix, img2_prefix, img3_prefix,
                          label_prefix, exp_name, iter_num, backbone, fold_index, test_list, num_classes,
                          patch_size, stride_x, stride_y, stride_z, save_result, use_mirror_ensemble):
    # load model
    if model_type == 'iter_num':
        model_path = os.path.join(model_root_path, exp_name, 'model_%d.pth' % iter_num)
    elif model_type == 'best':
        model_path = os.path.join(model_root_path, exp_name, 'model_%s.pth' % model_type)
    else:
        raise ValueError('model_type')

    if backbone == 'resunet_1':
        net = BaselineModel_res(in_channels=1, out_channels=num_classes).cuda()
    elif backbone == 'resunet_2':
        net = BaselineModel_res(in_channels=2, out_channels=num_classes).cuda()
    elif backbone == 'unet_1':
        net = BaselineModel_unet(1, num_classes).cuda()
    elif backbone == 'unet_2':
        net = BaselineModel_unet(2, num_classes).cuda()
    elif backbone == 'vnet_1':
        net = BaselineModel_vnet(in_channels=1, out_channels=num_classes, normalization='groupnorm').cuda()
    elif backbone == 'vnet_2':
        net = BaselineModel_vnet(in_channels=2, out_channels=num_classes, normalization='groupnorm').cuda()
    elif backbone == 'resvnet':
        net = BaselineModel_vnet(in_channels=1, out_channels=num_classes, normalization='groupnorm').cuda()
    elif backbone == 'vnet_2_se':
        net = BaselineModel_vnet_se(in_channels=2, out_channels=num_classes, normalization='groupnorm').cuda()
    elif backbone == 'unet_2_se':
        net = BaselineModel_unet_se(2, num_classes).cuda()
    elif backbone == 'resunet_3':
        net = BaselineModel_res(in_channels=3, out_channels=num_classes).cuda()
    elif backbone == 'resunet_2_se':
        net = BaselineModel_res_se(in_channels=2, out_channels=num_classes).cuda()
    else:
        raise ValueError('backbone')

    print("init weight from {}".format(model_path))
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # save inference path
    if model_type == 'iter_num':
        test_save_path = os.path.join('../data/inference/', exp_name, 'fold_'+str(fold_index), str(iter_num))
    elif model_type == 'best':
        test_save_path = os.path.join('../data/inference/', exp_name, 'fold_'+str(fold_index), model_type)
    else:
        raise ValueError('test_save_path')
    if use_mirror_ensemble:
        test_save_path += '_mirror_ensemble'
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    print(test_save_path)

    # model
    shutil.copy(model_path, os.path.join(test_save_path, os.path.basename(model_path)))

    test_list = sorted(test_list)

    if backbone in ['resunet_1', 'unet_1', 'vnet_1', 'resvnet', 'resunet_se_1']:
        avg_metric = test_all_case_1channel(net, test_list, data_root_path, img2_prefix, label_prefix,
                                            exp_name, num_classes=num_classes, patch_size=patch_size,
                                            stride_x=stride_x, stride_y=stride_y, stride_z=stride_z,
                                            save_result=save_result,
                                            test_save_path=test_save_path,
                                            use_mirror_ensemble=use_mirror_ensemble)
    elif backbone in ['resunet_2', 'unet_2', 'vnet_2', 'resunet_se_2', 'unet_2_se', 'vnet_2_se', 'resunet_2_se']:
        avg_metric = test_all_case_2channel(net, test_list, data_root_path, img1_prefix, img2_prefix, label_prefix,
                                            exp_name, num_classes=num_classes, patch_size=patch_size,
                                            stride_x=stride_x, stride_y=stride_y, stride_z=stride_z,
                                            save_result=save_result,
                                            test_save_path=test_save_path,
                                            use_mirror_ensemble=use_mirror_ensemble)
    # elif backbone in ['resunet_2', 'unet_2', 'vnet_2']:
    #     avg_metric = test_all_case_2channel(net, test_list, data_root_path, img2_prefix, img3_prefix, label_prefix,
    #                                         exp_name, num_classes=num_classes, patch_size=patch_size,
    #                                         stride_x=stride_x, stride_y=stride_y, stride_z=stride_z,
    #                                         save_result=save_result,
    #                                         test_save_path=test_save_path,
    #                                         use_mirror_ensemble=use_mirror_ensemble)
    elif backbone in ['resunet_3', 'unet_3', 'vnet_3']:
        avg_metric = test_all_case_3channel(net, test_list, data_root_path, img2_prefix, img2_prefix, img3_prefix, label_prefix,
                                                                                    exp_name, num_classes=num_classes, patch_size=patch_size,
                                                                                    stride_x=stride_x, stride_y=stride_y, stride_z=stride_z,
                                                                                    save_result=save_result,
                                                                                    test_save_path=test_save_path,
                                                                                    use_mirror_ensemble=use_mirror_ensemble)

    print('\nCalculating the hit results...\n')
    root_path = test_save_path
    prediction_path_list = sorted(glob(root_path + '/*pred*'))
    cases_result, mean, std, sensitivity, fp_num = localization_metrics(prediction_path_list)
    write_to_xls(cases_result, mean, std, sensitivity, root_path, exp_name)

    print('mean: Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (mean['Dice'], mean['Jaccard'], mean['ASD'], mean['95HD']))
    print('std:  Dice=%.4f Jaccard=%.4f ASD=%.4f 95HD=%.4f' % (std['Dice'], std['Jaccard'], std['ASD'], std['95HD']))
    print('Sensitivity=%.4f, FP num=%d\n' % (sensitivity, fp_num))
    return 1


if __name__ == '__main__':
    # def timedTask():
    #     # 初始化 sched 模块的 scheduler 类
    #     scheduler = sched.scheduler(time.time, time.sleep)
    #     # 增加调度任务
    #     scheduler.enter(25000, 1, task)
    #     # 运行任务
    #     scheduler.run()
    #
    # def task():
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #     num_classes = 2
    #     # patch_size = (288, 96, 80)
    #     # stride_x = 120
    #     # stride_z = 40
    #     # stride_y = 30
    #     patch_size = (256, 64, 48)
    #     stride_x = 120
    #     stride_y = 30
    #     stride_z = 20
    #     _, _, test_list = util.get_train_eval_test_list(args.fold_root_path, args.fold_index)
    #     test_list = test_list
    #     metric = test_calculate_metric(args.model_type, args.model_root_path, args.data_root_path,
    #                                    args.img1_prefix, args.img2_prefix, args.img3_prefix, args.label_prefix,
    #                                    args.exp_name, args.iter_num, args.backbone,
    #                                    args.fold_index, test_list, num_classes,
    #                                    patch_size, stride_x, stride_y, stride_z,
    #                                    args.save_result, args.use_mirror_ensemble)
    #
    # timedTask()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_classes = 2
    # patch_size = (288, 96, 80)
    # stride_x = 120
    # stride_z = 40
    # stride_y = 30
    patch_size = (256, 64, 48)
    stride_x = 120
    stride_y = 30
    stride_z = 20
    _, _, test_list = util.get_train_eval_test_list(args.fold_root_path, args.fold_index)
    test_list = test_list
    metric = test_calculate_metric(args.model_type, args.model_root_path, args.data_root_path,
                                   args.img1_prefix, args.img2_prefix, args.img3_prefix, args.label_prefix,
                                   args.exp_name, args.iter_num, args.backbone,
                                   args.fold_index, test_list, num_classes,
                                   patch_size, stride_x, stride_y, stride_z,
                                   args.save_result, args.use_mirror_ensemble)


