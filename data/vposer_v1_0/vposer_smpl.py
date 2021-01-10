# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# Vassilis Choutas <https://ps.is.tuebingen.mpg.de/employees/vchoutas> for ContinousRotReprDecoder
#
# 2018.01.02

'''
A human body pose prior built with Auto-Encoding Variational Bayes
'''

__all__ = ['VPoser']

import os, sys, shutil

import torch

from torch import nn
from torch.nn import functional as F

import numpy as np

import torchgeometry as tgm

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class VPoser(nn.Module):
    def __init__(self, num_neurons=512, latentD=32, data_shape=[1, 21, 3], use_cont_repr=True):
        super(VPoser, self).__init__()

        self.latentD = latentD
        self.use_cont_repr = use_cont_repr

        n_features = np.prod(data_shape)
        self.num_joints = data_shape[1]

        self.bodyprior_enc_bn1 = nn.BatchNorm1d(n_features)
        self.bodyprior_enc_fc1 = nn.Linear(n_features, num_neurons)
        self.bodyprior_enc_bn2 = nn.BatchNorm1d(num_neurons)
        self.bodyprior_enc_fc2 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_enc_mu = nn.Linear(num_neurons, latentD)
        self.bodyprior_enc_logvar = nn.Linear(num_neurons, latentD)
        self.dropout = nn.Dropout(p=.1, inplace=False)

        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)

        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()

        self.bodyprior_dec_out = nn.Linear(num_neurons, self.num_joints* 6)

    def encode(self, Pin):
        '''

        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        Xout = Pin.view(Pin.size(0), -1)  # flatten input
        Xout = self.bodyprior_enc_bn1(Xout)

        Xout = F.leaky_relu(self.bodyprior_enc_fc1(Xout), negative_slope=.2)
        Xout = self.bodyprior_enc_bn2(Xout)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_enc_fc2(Xout), negative_slope=.2)
        return torch.distributions.normal.Normal(self.bodyprior_enc_mu(Xout), F.softplus(self.bodyprior_enc_logvar(Xout)))

    def decode(self, Zin, output_type='matrot'):
        assert output_type in ['matrot', 'aa']

        Xout = F.leaky_relu(self.bodyprior_dec_fc1(Zin), negative_slope=.2)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_dec_fc2(Xout), negative_slope=.2)
        Xout = self.bodyprior_dec_out(Xout)
        if self.use_cont_repr:
            Xout = self.rot_decoder(Xout)
        else:
            Xout = torch.tanh(Xout)

        Xout = Xout.view([-1, 1, self.num_joints, 9])
        if output_type == 'aa': return VPoser.matrot2aa(Xout)
        return Xout

    def forward(self, Pin, input_type='matrot', output_type='matrot'):
        '''

        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''
        assert output_type in ['matrot', 'aa']
        # if input_type == 'aa': Pin = VPoser.aa2matrot(Pin)
        q_z = self.encode(Pin)
        q_z_sample = q_z.rsample()
        Prec = self.decode(q_z_sample)
        if output_type == 'aa': Prec = VPoser.matrot2aa(Prec)

        #return Prec, q_z.mean, q_z.sigma
        return {'pose':Prec, 'mean':q_z.mean, 'std':q_z.scale}

    def normal_distribution(self, pose):
        """
        :param pose (tensor, Nx1x21x3)
        :return:
        """
        q_z = self.encode(Pin)
        mean = q_z.mean
        std = q_z.scale

        return mean, std

    def sample_poses(self, num_poses, output_type='aa', seed=None):
        np.random.seed(seed)
        dtype = self.bodyprior_dec_fc1.weight.dtype
        device = self.bodyprior_dec_fc1.weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype).to(device)
        return self.decode(Zgen, output_type=output_type)

    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        batch_size = pose_matrot.size(0)
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
        return pose_body_matrot


if __name__ == '__main__':
    device = 'cpu'

    ## smplx
    abspath = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, abspath + '/../../')

    from common.smpl_x import SMPL_X
    from common.utils import save_obj

    smplx_model_path = os.path.join(abspath, '../')
    smpl = SMPL_X(model_path=smplx_model_path, model_type='smplx', gender='male').to(device)

    betas = torch.randn([1, 10], dtype=torch.float32).to(device)

    save_path = abspath + '/output'
    os.makedirs(save_path, exist_ok=True)

    ##
    para_dict = torch.load(abspath + '/snapshots/TR00_E096.pt', map_location=torch.device('cpu'))
    model = VPoser().to(device)
    model.load_state_dict(para_dict)
    model.eval()

    ##
    body_pose = torch.ones((1, 1, 21, 3), dtype=torch.float32).to(device)
    body_pose.requires_grad=True

    optimizer = torch.optim.Adam([body_pose], lr=30e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=800, threshold=1e-6, verbose=True)

    i = 0
    while(True):
        output = model(body_pose)
        loss = torch.norm(output['mean']).mean() + torch.norm(output['std']-1).mean()

        # print(input.grad)
        optimizer.zero_grad()
        (loss).backward()
        optimizer.step()
        scheduler.step(loss)

        # print(input.grad)
        if i % 1000 == 0:
            print("%d, %f" % (i, loss.item()))

        if i % 10 == 0:
            vertices, kp3d_pre, faces = smpl(body_pose=body_pose.view(1,-1),
                                             betas=betas)
            vertices = vertices.detach().cpu().numpy().squeeze()

            save_obj(save_path + '/%s.obj' % str(i).zfill(8), vertices=vertices, faces=faces)

            # import pyrender
            # import trimesh
            #
            # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
            # tri_mesh = trimesh.Trimesh(vertices, faces,
            #                            vertex_colors=vertex_colors, )
            #
            # mesh = pyrender.Mesh.from_trimesh(tri_mesh, wireframe=True)
            #
            # scene = pyrender.Scene()
            # scene.add(mesh)
            #
            # pyrender.Viewer(scene, use_raymond_lighting=True)

            ## sampl_pose test
            test_sample_pose = True
            if test_sample_pose:
                sampl_pose = model.sample_poses(1)

                vertices, kp3d_pre, faces = smpl(body_pose=sampl_pose.view(1, -1),
                                                 betas=betas)
                vertices = vertices.detach().cpu().numpy().squeeze()

                import pyrender
                import trimesh

                vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
                tri_mesh = trimesh.Trimesh(vertices, faces,
                                           vertex_colors=vertex_colors, )

                mesh = pyrender.Mesh.from_trimesh(tri_mesh, wireframe=True)

                scene = pyrender.Scene()
                scene.add(mesh)

                pyrender.Viewer(scene, use_raymond_lighting=True)

        i+=1

    print('load ok')
