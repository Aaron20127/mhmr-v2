
import numpy as np
import torch
import torch.nn as nn


class CameraPerspective(object):
    def __init__(self, intrinsic, extrinsic=None):
        super(CameraPerspective, self).__init__()
        """
            intrinsic(narray 3x3)
            extrinsic(narray 4x4)
        """
        self.intrinsic = intrinsic

        if extrinsic is not None:
            self.extrinsic = extrinsic
        else:
            self.extrinsic = np.eye(4)


    def world_2_camera(self, pts, extrinsic=None):
        """
            pts(narray Bx3x3)
            extrinsic(narray 4x4)
        """
        homogeneous_vertices = np.concatenate((pts, np.ones((pts.shape[0], pts.shape[1], 1))), axis=2)
        world_2_cam = extrinsic if extrinsic is not None else self.extrinsic
        vertices_camera = np.einsum("ij, bnj->bni", world_2_cam, homogeneous_vertices)[:, :, :3]

        return vertices_camera


    def perspective(self, pts, extrinsic=None):
        """
            pts(narray Bx3x3)
            extrinsic(narray 4x4)
        """
        vertices_camera = self.world_2_camera(pts, extrinsic=extrinsic)
        vertices_normal = vertices_camera / vertices_camera[:, :, 2][:, :, None]
        kp2d_projection = np.einsum("ij, bnj->bni", self.intrinsic, vertices_normal)[:, :, :2]

        return kp2d_projection



class CameraPerspectiveTorch(nn.Module):
    def __init__(self, intrinsic, extrinsic=None, device='cpu', data_type=torch.float32):
        super(CameraPerspectiveTorch, self).__init__()
        """
            intrinsic(narray/tensor 3x3)
            extrinsic(narray/tensor 4x4)
        """
        self.device = device
        self.data_type = data_type
        self.intrinsic = torch.tensor(intrinsic, dtype=data_type).to(device)

        if extrinsic is not None:
            self.extrinsic = torch.tensor(extrinsic, dtype=data_type).to(device)
        else:
            self.extrinsic = torch.eye(4, dtype=data_type).to(device)


    def world_2_camera(self, pts, extrinsic=None):
        """
            pts(narray Bx3x3)
            extrinsic(narray Bx4x4)
        """

        homogeneous_vertices = torch.cat((pts, torch.ones((pts.shape[0], pts.shape[1], 1)).to(pts.device)), axis=2)
        world_2_cam = extrinsic if extrinsic is not None else self.extrinsic
        vertices_camera = torch.einsum("ij, bnj->bni", world_2_cam, homogeneous_vertices)[:, :, :3]

        return vertices_camera


    def perspective(self, pts, intrinsic=None, extrinsic=None):
        """
            pts(tensor BX3x3)
            extrinsic(tensor Bx4x4)
        """
        self.intrinsic = intrinsic if intrinsic is not None else self.intrinsic


        vertices_camera = self.world_2_camera(pts, extrinsic=extrinsic)
        vertices_normal = vertices_camera / vertices_camera[:, :, 2][:, :, None]
        kp2d_projection = torch.einsum("ij, bnj->bni", self.intrinsic, vertices_normal)[:, :, :2]

        return kp2d_projection


class CameraPerspectiveTorchMultiImage(nn.Module):
    def __init__(self, intrinsic, extrinsic=None, device='cpu', data_type=torch.float32):
        super(CameraPerspectiveTorchMultiImage, self).__init__()
        """
            intrinsic(narray/tensor 3x3 or nx3x3)
            extrinsic(narray/tensor nx4x4)
        """
        self.device = device
        self.data_type = data_type
        self.intrinsic = torch.tensor(intrinsic, dtype=data_type).to(device)

        if extrinsic is not None:
            self.extrinsic = torch.tensor(extrinsic, dtype=data_type).to(device).view(-1, 4, 4)
        else:
            self.extrinsic = torch.eye(4, dtype=data_type).to(device).unsqueeze(0)


    def world_2_camera(self, pts, extrinsic=None):
        """
            pts(tensor num_imgxBxnx3))
            extrinsic(tensor num_imgx4x4)
        """
        homogeneous_vertices = torch.cat((pts,
            torch.ones((pts.shape[0], pts.shape[1], pts.shape[2], 1)).to(pts.device)), axis=3)
        world_2_cam = extrinsic if extrinsic is not None else self.extrinsic

        assert world_2_cam.shape[0] == pts.shape[0]

        vertices_camera = torch.einsum("kij, kbnj->kbni", world_2_cam, homogeneous_vertices)[:, :, :, :3]

        return vertices_camera


    def perspective(self, pts, intrinsic=None, extrinsic=None):
        """
            pts(tensor num_imgxBxnx3)
            extrinsic(tensor num_imgx4x4)
        """
        self.intrinsic = intrinsic if intrinsic is not None else self.intrinsic

        vertices_camera = self.world_2_camera(pts, extrinsic=extrinsic)
        vertices_normal = vertices_camera / vertices_camera[:, :, :, 2][:, :, :, None]

        if (len(self.intrinsic.shape) == 2):
            kp2d_projection = torch.einsum("ij, kbnj->kbni", self.intrinsic, vertices_normal)[:, :, :, :2]
        elif (len(self.intrinsic.shape) == 3):
            kp2d_projection = torch.einsum("kij, kbnj->kbni", self.intrinsic, vertices_normal)[:, :, :, :2]

        return kp2d_projection