import cv2
import numpy as np
import trimesh
import pyrender
import torch
import os

# os.environ['PYOPENGL_PLATFORM'] = 'egl'


from .utils import Rx_np

# https://github.com/daniilidis-group/neural_renderer
# must use pytorch 1.2.0
class PerspectiveNeuralRender(object):
    def __init__(self, K, R, t, height, width):
        """ Neural renderer.
            K (tensor, cuda, float32, Bx3x3) camera intrinsic
            R (tensor, cuda, float32, Bx3x3) extrinsic
            t (tensor, cuda, float32, Bx1x3) translation
            height (int) output image height
            width (int) output image width
        """
        self.output_size = max(height, width)
        self.height = height
        self.width = width

        # create renderer
        import neural_renderer as nr
        self.renderer = nr.Renderer(image_size=self.output_size,
                                    K=K, R=R, t=t,
                                    orig_size=self.output_size)

    def render_obj(self, vertices, faces, textures=None):
        """ render obj.
            vertices (tensor, float32, BxNx3)
            faces (tensor, int32, BxNx3)
            textures (tensor, float32, BxNxtxtxtx3)
        """
        if textures is None:
            texture_size = 2
            textures = torch.ones(1, faces.shape[1], texture_size,
                                       texture_size, texture_size, 3,
                                       dtype=torch.float32).cuda()

        image_normal, depth, _ = self.renderer(vertices, faces, textures)
        image_normal = image_normal.permute((0, 2, 3, 1))

        image_normal = image_normal[:, :self.height, :self.width, :]
        depth = depth[:, :self.height, :self.width]

        return image_normal, depth

    def render_mask(self, vertices, faces):
        mask = self.renderer(vertices, faces, mode='silhouettes')
        mask = mask[:, :self.height, :self.width]
        return mask


class PerspectivePyrender(object):
    def __init__(self,
                 intrinsic,
                 camera_pose,
                 width,
                 height,
                 light_intensity=4):
        self.scene = pyrender.Scene()

        # add camera
        camera = pyrender.camera.IntrinsicsCamera(
                fx=intrinsic[0][0], fy=intrinsic[1][1],
                cx=intrinsic[0][2], cy=intrinsic[1][2])

        self.camera_pose = camera_pose
        self.camera_scene_node = \
            self.scene.add(camera, pose=camera_pose)

        # add light
        # light = pyrender.PointLight(color=[1.0, 1.0, 1.0],
        #                             intensity=light_intensity)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                          intensity=light_intensity)

        self.scene.add(light, pose=camera_pose)

        # render
        self.r = pyrender.OffscreenRenderer(viewport_width = width,
                                            viewport_height = height,
                                            point_size = 1.0)


    def update_intrinsic(self, intrinsic):
        # add camera
        camera = pyrender.camera.IntrinsicsCamera(
            fx=intrinsic[0][0], fy=intrinsic[1][1],
            cx=intrinsic[0][2], cy=intrinsic[1][2])

        self.scene.remove_node(self.camera_scene_node)
        self.camera_scene_node = \
            self.scene.add(camera, pose=self.camera_pose)


    def render_mesh(self, mesh, show_viewer=False):
        # add mesh
        mesh_node = []
        for m in mesh:
            node = self.scene.add(m)
            mesh_node.append(node)

        if show_viewer:
            pyrender.Viewer(self.scene, use_raymond_lighting=True)

        color, depth = self.r.render(self.scene)

        # remove mesh
        for node in mesh_node:
            self.scene.remove_node(node)

        return color, depth


    def render_obj(self, vertices, faces, show_viewer=False):
        mesh = []
        vertex_colors = np.ones([vertices.shape[0], 4]) * [1.0, 1.0, 1.0, 1.0]
        tri_mesh = trimesh.Trimesh(vertices, faces,
                                   vertex_colors=vertex_colors)
        mesh_obj = pyrender.Mesh.from_trimesh(tri_mesh)
        mesh.append(mesh_obj)

        return self.render_mesh(mesh, show_viewer)



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



