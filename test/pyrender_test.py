
import numpy as np
import os
import sys
import pickle as pkl
import cv2
import pyrender
import trimesh
import h5py
import torch
abspath = os.path.abspath(os.path.dirname(__file__))

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def perspective_render_obj(cam, mesh, width=512,height=512, show_viwer=False):
    scene = pyrender.Scene()

    camera_pose = cam['camera_pose']

    camera=pyrender.camera.IntrinsicsCamera(
            fx=cam['fx'], fy=cam['fy'],
            cx=cam['cx'], cy=cam['cy'])
    scene.add(camera, pose=camera_pose)


    # add spere
    for m in mesh:
        scene.add(m)


    if show_viwer:
        pyrender.Viewer(scene, use_raymond_lighting=True)

    # add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=80)
    scene.add(light, pose=camera_pose)

    # render
    r = pyrender.OffscreenRenderer(viewport_width=width,viewport_height = height,point_size = 1.0)
    color, depth = r.render(scene)

    return color, depth



if __name__ == '__main__':
    """
        pyrender camera axis:
            bottow to top: Y
            front to back: Z
            right hand rule: YxZ -> X
            
        Projective projecton (default image coordinate system is lower left corner) :
            u = (x/z) * f + cx
            v = (y/z) * f + cy
            v' = height - v  # The image coordinate system changes from the lower left corner to the upper left corner
            
    """
    height = 100
    width = 100

    K = np.array([[1000,  0,  width/2],
                  [0,  1000,  height/2],
                  [0,     0,  1]])

    camera_pose = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 10],
                            [0, 0, 0, 1]])

    mesh = []

    ## add obj mesh
    verts = np.array([[0, 0, 0],
                      [0.2, 0, 0],
                      [0.4, 0.4, -3.3333]])

    faces = np.array([[0.,1.,2.]])

    vertex_colors = np.ones([verts.shape[0], 4]) * [0.3, 0.3, 0.3, 1.0]
    tri_mesh = trimesh.Trimesh(verts, faces,
                               vertex_colors=vertex_colors)
    mesh_obj = pyrender.Mesh.from_trimesh(tri_mesh)

    mesh.append(mesh_obj)

    ## add sphere mesh
    # bluce
    sphere_pos = np.array([[0.4, 0, 0]])
    ms = trimesh.creation.uv_sphere(radius=0.2)
    ms.visual.vertex_colors = [1.0, 0.0, 0.0]
    pts = sphere_pos
    tfs = np.tile(np.eye(4), (len(pts), 1, 1))
    tfs[:, :3, 3] = pts
    mesh_sphere = pyrender.Mesh.from_trimesh(ms, poses=tfs)
    # mesh.append(mesh_sphere)

    # green
    sphere_pos = np.array([[0, 0.4, 0]])
    ms = trimesh.creation.uv_sphere(radius=0.2)
    ms.visual.vertex_colors = [0.0, 1.0, 0.0]
    pts = sphere_pos
    tfs = np.tile(np.eye(4), (len(pts), 1, 1))
    tfs[:, :3, 3] = pts
    mesh_sphere = pyrender.Mesh.from_trimesh(ms, poses=tfs)
    # mesh.append(mesh_sphere)

    # red
    sphere_pos = np.array([[0, 0, 0.4]])
    ms = trimesh.creation.uv_sphere(radius=0.2)
    ms.visual.vertex_colors = [0.0, 0.0, 1.0]
    pts = sphere_pos
    tfs = np.tile(np.eye(4), (len(pts), 1, 1))
    tfs[:, :3, 3] = pts
    mesh_sphere = pyrender.Mesh.from_trimesh(ms, poses=tfs)
    # mesh.append(mesh_sphere)

    # add camera
    cam = {
        'fx': K[0, 0],
        'fy': K[1, 1],
        'cx': K[0, 2],
        'cy': K[1, 2],
        'camera_pose': camera_pose
    }

    # render
    color, depth = perspective_render_obj(cam, mesh, width=width, height=height)
    # img = add_blend_smpl(color, img)

    cv2.namedWindow('img',0)
    cv2.imshow('img', color)
    cv2.waitKey(0)
