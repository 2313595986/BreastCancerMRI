import os
import torch
import argparse
import shutil
from glob import glob
from utils import util
from networks.nnResUnet import nnResUNet
from networks.UNet3D import UNet3D_DS
from networks.VNet import VNet_DS
from test_util import test_all_case_1channel


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,  default='7', help='GPU to use')
parser.add_argument('--model_type', type=str, default='best', choices=['best', 'iter_num'])
parser.add_argument('--fold_root_path', type=str, default='../data/', help='fold txt path')
parser.add_argument('--data_root_path', type=str, default='/home/wh/datasets/BreastCancerMRI_185/CropZero_down0.5XY/Norm_yuantu/', help='data path')
parser.add_argument("--img1_prefix", type=str, default='DCE-C1-615', help="img prefix")
parser.add_argument("--label_prefix", type=str, default='BreastMask-615', help="label prefix")
parser.add_argument('--model_root_path', type=str, default='../model/', help='model root path')
parser.add_argument('--backbone', type=str, default='resunet')
parser.add_argument('--exp_name', type=str, default='BreastSeg_c1_fold1_resunet', help='experiment name')
parser.add_argument('--fold_index', type=int, default=4, help='index of fold')
parser.add_argument('--iter_num', type=int,  default=22000, help='model iteration')
parser.add_argument('--save_result', type=bool, default=True, help='save result?')
parser.add_argument('--use_mirror_ensemble', type=bool, default=False, help='use mirror for ensemble?')
args = parser.parse_args()


def test_calculate_metric(model_type, model_root_path, data_root_path, img1_prefix,
                          label_prefix, exp_name, iter_num, backbone, fold_index, test_list, num_classes,
                          patch_size, stride_x, stride_z, stride_y, save_result, use_mirror_ensemble):
    # load model
    if model_type == 'iter_num':
        model_path = os.path.join(model_root_path, exp_name, 'model_%d.pth' % iter_num)
    elif model_type == 'best':
        model_path = os.path.join(model_root_path, exp_name, 'model_%s.pth' % model_type)
    else:
        raise ValueError('model_type')

    if backbone == 'resunet':
        net = nnResUNet(in_channels=1, out_channels=2, is_dropput=False).cuda()
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

    avg_metric = test_all_case_1channel(net, test_list, data_root_path, img1_prefix, label_prefix,
                                        exp_name, num_classes=num_classes, patch_size=patch_size,
                                        stride_x=stride_x, stride_y=stride_y, stride_z=stride_z,
                                        save_result=save_result,
                                        test_save_path=test_save_path,
                                        use_mirror_ensemble=use_mirror_ensemble)
    return avg_metric


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_classes = 2
    patch_size = (256, 256, 64)
    stride_x = 120
    stride_y = 120
    stride_z = 30
    _, _, test_list = util.get_train_eval_test_list(args.fold_root_path, args.fold_index)
    test_list = test_list
    metric = test_calculate_metric(args.model_type, args.model_root_path, args.data_root_path,
                                   args.img1_prefix, args.label_prefix,
                                   args.exp_name, args.iter_num, args.backbone,
                                   args.fold_index, test_list, num_classes,
                                   patch_size, stride_x, stride_z, stride_y,
                                   args.save_result, args.use_mirror_ensemble)

