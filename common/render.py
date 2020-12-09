import cv2
import numpy as np
import trimesh
import pyrender
import torch
import os

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

from .utils import Rx_np

class PerspectiveRender(object):
    def __init__(self, camera_intrinsic, camera_pose, width=256, height=256):
        self.scene = pyrender.Scene()

        # add camera
        camera = pyrender.camera.IntrinsicsCamera(
                fx=camera_intrinsic[0][0], fy=camera_intrinsic[1][1],
                cx=camera_intrinsic[0][2], cy=camera_intrinsic[1][2])

        self.scene.add(camera, pose=camera_pose)

        # add light
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0],
                                    intensity=40 * np.linalg.norm(camera_pose[:3, 3]))

        self.scene.add(light, pose=camera_pose)

        # render
        self.r = pyrender.OffscreenRenderer(viewport_width = width,
                                            viewport_height = height,
                                            point_size = 1.0)


    def run(self, mesh, show_viwer=False):
        # add mesh
        mesh_node = []
        for m in mesh:
            node = self.scene.add(m)
            mesh_node.append(node)

        if show_viwer:
            pyrender.Viewer(self.scene, use_raymond_lighting=True)

        color, depth = self.r.render(self.scene)

        # remove mesh
        for node in mesh_node:
            self.scene.remove_node(node)

        return color, depth



def perspective_render_obj_debug(cam, obj, rotate_x_axis=True, width=512,height=512, show_smpl_joints=False,
                                 show_smpl=True, use_viewer=False, bg_color=[0,0,0,0]):
    scene = pyrender.Scene(bg_color=bg_color)

    # add camera
    camera_pose = np.array([
        [1.0,  0.0,  0.0,   cam['trans_x']],
        [0.0,  1.0,  0.0,   cam['trans_y']],
        [0.0,  0.0,  1.0,   cam['trans_z']],
        [0.0,  0.0,  0.0,   1.0],
    ])
    camera=pyrender.camera.IntrinsicsCamera(
            fx=cam['fx'], fy=cam['fy'],
            cx=cam['cx'], cy=cam['cy'])
    scene.add(camera, pose=camera_pose)

    # add verts and faces
    if rotate_x_axis:
        rot_x = Rx_mat(torch.tensor([np.pi])).numpy()[0]
        obj['verts'] = np.dot(obj['verts'], rot_x.T)
        obj['J'] = np.dot(obj['J'], rot_x.T)

    # add verts and faces
    if show_smpl:
        vertex_colors = np.ones([obj['verts'].shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(obj['verts'], obj['faces'],
                                   vertex_colors=vertex_colors)
        mesh_obj = pyrender.Mesh.from_trimesh(tri_mesh)

        scene.add(mesh_obj)

    # add joints
    if show_smpl_joints:
        ms = trimesh.creation.uv_sphere(radius=0.015)
        ms.visual.vertex_colors = [1.0, 0.0, 0.0]

        pts = obj['J']
        # pts = pts[22,:]

        tfs = np.tile(np.eye(4), (len(pts), 1, 1))
        tfs[:, :3, 3] = pts

        mesh_J = pyrender.Mesh.from_trimesh(ms, poses=tfs)
        scene.add(mesh_J)

    if use_viewer:
        pyrender.Viewer(scene, use_raymond_lighting=True)

    # add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=8*camera_pose[2,3])
    scene.add(light, pose=camera_pose)

    # render
    r = pyrender.OffscreenRenderer(viewport_width=width,viewport_height = height,point_size = 1.0)
    color, depth = r.render(scene)

    return color, depth


def weak_perspective_first_translate(verts, camera):
    '''
    对顶点做弱透视变换，只对x,y操作
    Args:
        verts:
        camera: [s,cx,cy]
    '''
    # camera = camera.view(1, 3)
    v = verts.detach().clone()

    v[..., :2] = v[..., :2] + camera[1:]
    v[..., :2] = v[..., :2] * camera[0]
    return v
