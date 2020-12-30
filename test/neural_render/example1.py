"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import cv2
import os
import argparse

import torch
import numpy as np
import tqdm
import imageio

# https://github.com/daniilidis-group/neural_renderer
# must use pytorch 1.2.0
import neural_renderer as nr

abspath = os.path.abspath(os.path.dirname(__file__))


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    # parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.gif'))
    # parser.add_argument('-g', '--gpu', type=int, default=0)
    # args = parser.parse_args()

    obj_path = os.path.join(abspath, 'data/teapot.obj')


    # other settings
    camera_distance = 2.732
    elevation = 30
    texture_size = 2

    # load .obj
    vertices, faces = nr.load_obj(obj_path)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    # draw object
    cv2.namedWindow('img', 0)

    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]

        cv2.imshow('img', (image*255).astype(np.uint8))
        cv2.waitKey(100)

if __name__ == '__main__':
    main()
