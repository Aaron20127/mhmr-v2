"""
Example 3. Optimizing textures.
"""
from __future__ import division
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
import cv2

import neural_renderer as nr

abspath = os.path.abspath(os.path.dirname(__file__))


class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # load reference image
        image_ref = torch.from_numpy(imread(filename_ref).astype('float32') / 255.).permute(2,0,1)[None, ::]
        self.register_buffer('image_ref', image_ref)

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer


    def forward(self):
        self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        image, _, _ = self.renderer(self.vertices, self.faces, torch.tanh(self.textures))
        loss = torch.sum((image - self.image_ref) ** 2)
        return loss


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    # parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example3_ref.png'))
    # parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example3_result.gif'))
    # parser.add_argument('-g', '--gpu', type=int, default=0)
    # args = parser.parse_args()

    filename_obj = os.path.join(abspath, 'data/teapot.obj')
    filename_ref = os.path.join(abspath, 'data/example3_ref.png')

    model = Model(filename_obj, filename_ref)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(300))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

    # draw object
    cv2.namedWindow('img', 0)
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))

        cv2.imshow('img', (image*255).astype(np.uint8))
        cv2.waitKey(100)


if __name__ == '__main__':
    main()
