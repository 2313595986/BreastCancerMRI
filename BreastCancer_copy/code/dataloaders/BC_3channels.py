import os
import torch
import random
import nibabel as nib
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Sampler


class BC(Dataset):
    def __init__(self, case_list, data_root_path, img1_prefix='DCE-C1', img2_prefix='DCE-C0', img3_prefix='Subtraction1',
                 label_prefix='TumorMask', transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.img1_prefix = img1_prefix
        self.img2_prefix = img2_prefix
        self.img3_prefix = img3_prefix
        self.label_prefix = label_prefix
        self.transform = transform

    def __getitem__(self, index):
        volume1_path = os.path.join(self.data_root_path, self.img1_prefix,
                                    self.img1_prefix + '_' + self.case_list[index])
        volume2_path = os.path.join(self.data_root_path, self.img2_prefix,
                                    self.img2_prefix + '_' + self.case_list[index])
        volume3_path = os.path.join(self.data_root_path, self.img3_prefix,
                                    self.img3_prefix + '_' + self.case_list[index])
        label_path = os.path.join(self.data_root_path, self.label_prefix,
                                  self.label_prefix + '_' + self.case_list[index])

        volume1 = nib.load(volume1_path).get_fdata()
        volume2 = nib.load(volume2_path).get_fdata()
        volume3 = nib.load(volume3_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        name = self.case_list[index].split('.')[0]
        sample = {'name': name, 'volume1': volume1, 'volume2': volume2, 'volume3': volume3, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.case_list)


class Crop2patchesToTensor(object):
    def __init__(self, output_size, volume_key1='volume1', volume_key2='volume2', volume_key3='volume3', label_key='label'):
        self.output_size = output_size
        self.volume_key1 = volume_key1
        self.volume_key2 = volume_key2
        self.volume_key3 = volume_key3
        self.label_key = label_key

    def __call__(self, sample):
        volume1, volume2, volume3 = sample[self.volume_key1], sample[self.volume_key2], sample[self.volume_key3]
        label = sample[self.label_key]

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            volume1 = np.pad(volume1, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            volume2 = np.pad(volume2, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            volume3 = np.pad(volume3, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        # one patch for random
        (w, h, d) = volume1.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label_random = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume_random1 = volume1[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume_random2 = volume2[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume_random3 = volume3[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        # one patch for foreground
        bbox = self.find_bbox(label)
        if bbox['shape'][0] < self.output_size[0]:
            if max(0, bbox['x2'] - self.output_size[0]) >= min(w - self.output_size[0], bbox['x1']):
                w2 = min(w - self.output_size[0], bbox['x1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                w2 = np.random.randint(max(0, bbox['x2'] - self.output_size[0]), min(w - self.output_size[0], bbox['x1']))
        else:
            w2 = np.random.randint(max(0, bbox['x1'] - self.output_size[0]), min(w - self.output_size[0], bbox['x2']))

        if bbox['shape'][1] < self.output_size[1]:
            if max(0, bbox['y2'] - self.output_size[1]) >= min(h - self.output_size[1], bbox['y1']):
                h2 = min(h - self.output_size[1], bbox['y1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                h2 = np.random.randint(max(0, bbox['y2'] - self.output_size[1]), min(h - self.output_size[1], bbox['y1']))
        else:
            h2 = np.random.randint(max(0, bbox['y1'] - self.output_size[1]), min(h - self.output_size[1], bbox['y2']))

        if bbox['shape'][2] < self.output_size[2]:
            if max(0, bbox['z2'] - self.output_size[2]) >= min(d - self.output_size[2], bbox['z1']):
                d2 = min(d - self.output_size[2], bbox['z1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                d2 = np.random.randint(max(0, bbox['z2'] - self.output_size[2]), min(d - self.output_size[2], bbox['z1']))
        else:
            d2 = np.random.randint(max(0, bbox['z1'] - self.output_size[2]), min(d - self.output_size[2], bbox['z2']))

        label_foreground = label[w2:w2 + self.output_size[0], h2:h2 + self.output_size[1], d2:d2 + self.output_size[2]]
        volume_foreground1 = volume1[w2:w2 + self.output_size[0], h2:h2 + self.output_size[1], d2:d2 + self.output_size[2]]
        volume_foreground2 = volume2[w2:w2 + self.output_size[0], h2:h2 + self.output_size[1], d2:d2 + self.output_size[2]]
        volume_foreground3 = volume3[w2:w2 + self.output_size[0], h2:h2 + self.output_size[1], d2:d2 + self.output_size[2]]

        # concatenate
        volume1 = np.concatenate((np.expand_dims(volume_random1, 0), np.expand_dims(volume_foreground1, 0)))
        volume2 = np.concatenate((np.expand_dims(volume_random2, 0), np.expand_dims(volume_foreground2, 0)))
        volume3 = np.concatenate((np.expand_dims(volume_random3, 0), np.expand_dims(volume_foreground3, 0)))
        label = np.concatenate((np.expand_dims(label_random, 0), np.expand_dims(label_foreground, 0)))

        # To Tensor
        volume1 = torch.Tensor(np.expand_dims(volume1, axis=1).copy())
        volume2 = torch.Tensor(np.expand_dims(volume2, axis=1).copy())
        volume3 = torch.Tensor(np.expand_dims(volume3, axis=1).copy())
        label = torch.Tensor(np.expand_dims(label, axis=1).copy())

        sample[self.volume_key1], sample[self.volume_key2], sample[self.volume_key3] = volume1, volume2, volume3
        sample[self.label_key] = label
        return sample

    def find_bbox(self, array):
        shape = array.shape

        x1 = 0
        x2 = shape[0] - 1
        y1 = 0
        y2 = shape[1] - 1
        z1 = 0
        z2 = shape[2] - 1

        while (array[x1, :, :] == 0).all():
            x1 += 1
        while (array[x2, :, :] == 0).all():
            x2 -= 1

        while (array[:, y1, :] == 0).all():
            y1 += 1
        while (array[:, y2, :] == 0).all():
            y2 -= 1

        while (array[:, :, z1] == 0).all():
            z1 += 1
        while (array[:, :, z2] == 0).all():
            z2 -= 1

        x_len = x2 - x1 + 1
        y_len = y2 - y1 + 1
        z_len = z2 - z1 + 1
        shape = (x_len, y_len, z_len)
        bbox = {'x1': x1,
                'x2': x2,
                'y1': y1,
                'y2': y2,
                'z1': z1,
                'z2': z2,
                'shape': shape}
        return bbox


if __name__ == '__main__':
    from utils import util
    from glob import glob
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from data_augmentation_sub import *
    patch_size = (288, 96, 80)
    fold_root_path = '../../data'
    data_root_path = '/home/hra/dataset/BreastCancerMRI_Reviewed/YN/VOI-present/BreastROI/Norm/'
    sub_root_path = '/home/hra/dataset/BreastCancerMRI_Reviewed/YN/VOI-present/Subtraction/BreastROI/Norm/img'
    save_path = '../../data/demo'
    train_volume_path = sorted(glob(data_root_path + 'img/*'))

    train_dataset = BC(train_volume_path, sub_root_path,
                       transform=transforms.Compose([RandomRotateTransform(angle_range=(-10, 10), p_per_sample=1),
                                                     ScaleTransform(zoom_range=(0.7, 1.4), p_per_sample=1),
                                                     GaussianNoiseTransform(p_per_sample=1),
                                                     GaussianBlurTransform(different_sigma_per_channel=False,
                                                                           blur_sigma=(0.1, 0.3), p_per_sample=1),
                                                     BrightnessMultiplicativeTransform(multiplier_range=(0.70, 1.3),
                                                                                       per_channel=False,
                                                                                       p_per_sample=1),
                                                     ContrastAugmentationTransform(contrast_range=(0.65, 1.5),
                                                                                   per_channel=False,
                                                                                   p_per_sample=1),
                                                     GammaTransform(gamma_range=(0.7, 1.5), retain_stats=True,
                                                                    p_per_sample=1),
                                                     MirrorTransform(axes=(0,1,2)),
                                                     Crop2patchesToTensor(patch_size)]))
    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=False)
    for i_batch, sampled_batch in enumerate(train_dataloader):
        volume_batch = sampled_batch['volume'].to('cuda').float()
        sub_volume_batch = sampled_batch['sub_volume'].to('cuda').float()

        demo = torch.cat((volume_batch, sub_volume_batch), dim=1)
        nib.save(nib.Nifti1Image(demo[0, 0].cpu().numpy(), np.eye(4)), os.path.join(save_path, '0_0.nii.gz'))
        nib.save(nib.Nifti1Image(demo[0, 1].cpu().numpy(), np.eye(4)), os.path.join(save_path, '0_1.nii.gz'))
        nib.save(nib.Nifti1Image(demo[1, 0].cpu().numpy(), np.eye(4)), os.path.join(save_path, '1_0.nii.gz'))
        nib.save(nib.Nifti1Image(demo[1, 1].cpu().numpy(), np.eye(4)), os.path.join(save_path, '1_1.nii.gz'))
        print('save done')
        print("%sms" % sampled_batch['used_time'])


