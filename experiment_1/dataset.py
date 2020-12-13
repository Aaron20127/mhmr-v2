
import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../../')

import copy
import h5py
import json
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from common.utils import Clock
from config import opt


class DatasetTwoPerson(Dataset):
    def __init__(self, data_dir, split='train', max_data_len=-1):
        self.data_dir = data_dir
        self.split = split
        self.max_data_len=max_data_len

        # load data set
        self._load_data_set()


    def _load_data_set(self):
        clk = Clock()
        print('==> loading DatasetTwoPerson {} data.'.format(self.split))
        self.render_images = []
        self.full_mask_images = []

        anno_file_path = os.path.join(self.data_dir, '{}.h5'.format(self.split))

        with h5py.File(anno_file_path, 'r') as fp:
            self.valid = np.array(fp['valid'])
            self.shape = np.array(fp['shape'])
            self.pose = np.array(fp['pose'])
            self.obj_pose = np.array(fp['obj_pose'])
            self.camera_pose = np.array(fp['camera_pose'])
            self.image_index = np.array(fp['image_index'])

            self.num_camera, self.num_shape, self.num_pose, self.num_person = self.valid.shape

            valid = self.valid[..., 0].reshape(self.num_camera, self.num_shape, self.num_pose)
            self.index = np.argwhere(valid == 1)
            self.data_len = len(self.index)

            image_render_name = np.array(fp['image_render_name'])
            image_full_mask_name = np.array(fp['image_full_mask_name'])
            for i in range(len(image_render_name)):
                self.render_images.append(image_render_name[i].decode())
                self.full_mask_images.append(image_full_mask_name[i].decode())

            if self.data_len > self.max_data_len > 0:
                self.data_len = self.max_data_len

        print('loaded {} samples (t={:.2f}s)'.format(self.data_len, clk.elapsed()))


    def __len__(self):
        return self.data_len


    def _get_image(self, img_id):
        render_img_name = self.render_images[img_id]
        full_mask_img_name = self.full_mask_images[img_id]

        render_img = cv2.imread(os.path.join(self.data_dir, render_img_name))
        full_mask_img = cv2.imread(os.path.join(self.data_dir, full_mask_img_name))

        render_img = render_img.transpose((2, 0, 1)).astype(np.float32)
        full_mask_img.transpose((2, 0, 1)).astype(np.float32)

        return render_img, full_mask_img


    def _get_data(self, index):
        camera_pose = self.camera_pose[index[0]]
        obj_pose = self.obj_pose[index[0], index[1], index[2]]
        camera_pose_inv = np.linalg.inv(camera_pose)

        world_2_camera = np.einsum('nj,bjm-> bnm', camera_pose_inv, obj_pose)
        near_2_far_order = np.argsort(-world_2_camera[:, 2, 3])

        shape = self.shape[index[0], index[1], index[2]]
        pose = self.pose[index[0], index[1], index[2]]

        shape_new = shape[near_2_far_order].astype(np.float32)
        pose_new = pose[near_2_far_order].reshape(-1, 72).astype(np.float32)

        return shape_new, pose_new


    def __getitem__(self, id):
        index = self.index[id]
        img_id = self.image_index[index[0]][index[1]][index[2]]

        ## 1.get img
        render_img, full_mask_img = self._get_image(img_id)

        ## 2.get data
        shape, pose = self._get_data(index)


        ## 3. input
        input = render_img

        return {
            'input': input.astype(np.float32),
            'shape': shape.astype(np.float32),
            'pose': pose.astype(np.float32)
        }


if __name__ == '__main__':
    torch.manual_seed(opt.seed)

    data_dir = os.path.join(abspath, 'data_prepare', 'dataset')

    data = DatasetTwoPerson(
        data_dir = data_dir,
        split = 'train'
    )

    data_loader = DataLoader(data, batch_size=2, shuffle=False)

    from tqdm import trange
    from random import random, randint
    from time import sleep

    t = tqdm(data_loader)
    for i, j in enumerate(t):
        # 描述将显示在左边
        t.set_description('GEN %i' % 1)
        # 后缀将显示在右边，根据参数的数据类型自动格式化
        t.set_postfix(loss=random(), gen=randint(1, 999), str='h',
                      lst=[1, 2])
        # sleep(0.1)


