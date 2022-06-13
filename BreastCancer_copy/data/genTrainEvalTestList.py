import os
from glob import glob
import codecs
import random
if __name__ == '__main__':
    root_path = '/home/wh/datasets/BreastCancerMRI_185/CropZero_down0.5XY/Norm/DCE-C1'
    root_path_2 = '/home/wh/datasets/BreastCancerMRI_185/CropZero_down0.5XY/Norm/DCE-C1_185'
    case_list = sorted(glob(root_path + '/*'))
    case_list_2 = sorted(glob(root_path_2 + '/*'))
    random.shuffle(case_list)
    id_list = [os.path.basename(i).split('_')[1] for i in case_list]
    id_list_2 = [os.path.basename(i).split('_')[1] for i in case_list_2]
    train_list = []
    for i, index in enumerate(id_list):
        if index not in id_list_2:
            train_list.append(index)
    # train_list = id_list[:]
    # eval_list = id_list[74:111]
    # test_list = id_list[111:]
    with codecs.open('fold_3.txt', encoding='utf-8', mode='w') as f:
        f.write('train:\n')
        for name in train_list:
            f.write('%s\n' % name)
        # f.write('eval:\n')
        # for name in eval_list:
        #     f.write('%s\n' % name)
        # f.write('test:\n')
        # for name in test_list:
        #     f.write('%s\n' % name)