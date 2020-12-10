
import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../../')

import copy
import h5py
import json
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from common.utils import Clock
from opts import opt


class DatasetTwoPerson(Dataset):
    def __init__(self, data_dir, split='train', max_data_len=-1):
        self.data_dir = data_dir
        self.split = split
        self.max_data_len=-1

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

                if 0 < self.max_data_len <= len(self.render_images):
                    break

        print('loaded {} samples (t={:.2f}s)'.format(self.data_len, clk.elapsed()))


    def __len__(self):
        return self.data_len


    def _get_image(self, img_id):
        render_img_name = self.render_images[img_id]
        full_mask_img_name = self.full_mask_images[img_id]

        render_img = cv2.imread(os.path.join(self.data_dir, render_img_name))
        full_mask_img = cv2.imread(os.path.join(self.data_dir, full_mask_img_name))

        return render_img, full_mask_img


    def _get_data(self, index):
        camera_pose = self.camera_pose[index[0]]
        obj_pose = self.obj_pose[index[0], index[1], index[2]]
        camera_pose_inv = np.linalg.inv(camera_pose)

        world_2_camera = np.einsum('nj,bjm-> bnm', camera_pose_inv, obj_pose)
        near_2_far_order = np.argsort(-world_2_camera[:, 2, 3])

        shape = self.shape[index[0], index[1], index[2]]
        pose = self.pose[index[0], index[1], index[2]]

        shape_new = shape[near_2_far_order]
        pose_new = pose[near_2_far_order]

        return shape_new, pose_new


    def __getitem__(self, id):
        index = self.index[id]
        img_id = self.image_index[index[0]][index[1]][index[2]]

        ## 1.get img
        render_img, full_mask_img = self._get_image(img_id)

        ## 2.get data
        shape_new, pose_new = self._get_data(index)


        return {
            'imagepath': self.img_dir + '/' + imagename,
            'kp2d_input': kp2d_input,
            'kp2d': kp2d,
            'gt': gt
        }


if __name__ == '__main__':
    torch.manual_seed(opt.seed)

    data_dir = os.path.join(abspath, 'data_prepare', 'dataset')

    data = DatasetTwoPerson(
        data_dir = data_dir,
        split = 'train'
    )

    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    for batch in data_loader:
        print(1)

