
import os
import sys
import torch
import numpy as np


abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')


############################### loss #######################################
def l1_loss(pred, target):
    loss = torch.abs(pred - target).sum()
    loss = loss / pred.numel()
    return loss


def l2_loss(pred, target):
    loss = torch.sum((pred - target)**2)
    loss = loss / pred.numel()
    return loss


def mask_loss(pred, target):
    out = pred[(((1 - target) + pred) > 1)]
    loss = 0.5 * torch.sum((pred - target)**2) + 1.0 * torch.sum(out**2)
    loss = loss / pred.numel()
    return loss


def part_mask_loss(pred, target):
    out = pred[(((1 - target) + pred) > 1)]
    loss = 0.5 * torch.sum((pred - target)**2) + 1.0 * torch.sum(out**2)
    loss = loss / pred.numel()
    return loss


def _smpl_collision_loss(vertices_a, vertices_center_b, normal_b):
    # min distance index
    va_2_vb_center = vertices_a.unsqueeze(1) - \
                     vertices_center_b.unsqueeze(0). \
                         expand(vertices_a.shape[0], vertices_center_b.shape[0], vertices_center_b.shape[1])
    min_distance_index = torch.argmin(torch.norm(va_2_vb_center, dim=2), dim=1)

    # matching face_center and normal
    vertices_center_b = vertices_center_b[min_distance_index]
    normal_b = normal_b[min_distance_index]

    # points index of a in b
    va_2_vb_center = vertices_a - vertices_center_b
    inner_index = ((va_2_vb_center * normal_b).sum(dim=1) < 0)

    loss = torch.tensor(0.).cuda()
    if inner_index.sum() > 0:
        loss = torch.norm(va_2_vb_center[inner_index], dim=1).mean()

    return loss


def smpl_collision_loss(vertices_batch, faces):

    vertices_center_list = []
    faces_normal_list = []
    # vertices_center_sample_list = []
    # faces_normal_sample_list = []
    for i, vertices in enumerate(vertices_batch):
        # all faces
        faces_vertices = vertices_batch[i][faces.flatten().type(torch.long)].view(-1, 3, 3)
        vertices_center = faces_vertices.mean(dim=1)
        faces_normal = torch.cross(faces_vertices[:, 1, :] - faces_vertices[:, 0, :],
                                   faces_vertices[:, 2, :] - faces_vertices[:, 1, :], dim=1)
        vertices_center_list.append(vertices_center)
        faces_normal_list.append(faces_normal)

    # a in b
    vertices_a = vertices_batch[1]
    vertices_center_b = vertices_center_list[0]
    normal_b = faces_normal_list[0]
    loss = _smpl_collision_loss(vertices_a, vertices_center_b, normal_b)

    # b in a
    vertices_a = vertices_batch[0]
    vertices_center_b = vertices_center_list[1]
    normal_b = faces_normal_list[1]
    loss += _smpl_collision_loss(vertices_a, vertices_center_b, normal_b)

    return loss



def _touch_loss(vertices_a, vertices_b, normal_a, normal_b):
    vector = vertices_a.unsqueeze(1) - \
             vertices_b.unsqueeze(0).\
                expand(vertices_a.shape[0], vertices_b.shape[0], vertices_b.shape[1])
    min_distance_index = torch.argmin(torch.norm(vector, dim=2), dim=1)

    vertices_b = vertices_b[min_distance_index]
    normal_b = normal_b[min_distance_index]

    # loss distance
    loss_distance = torch.norm(vertices_a - vertices_b, dim=1).mean()

    # loss normal
    loss_normal = ((normal_a * normal_b).sum(dim=1) + 1.0).mean()

    loss = (loss_distance + loss_normal) / 2.0

    return loss


def touch_loss(opt, vertices_batch):

    loss = torch.tensor(0.).cuda()
    for touch_part in opt.dataset['touch_pair_list']:
        # faces center and normal
        vertices_center_list = []
        faces_normal_list = []
        vertices_center_sample_list = []
        faces_normal_sample_list = []
        for i, part_faces in touch_part.items():
            # sample faces
            len_part_face = part_faces.shape[0]
            face_index = np.random.randint(len_part_face, size=opt.num_sample_touch_face)
            face_sample = part_faces[face_index]
            vertices_faces = vertices_batch[i][face_sample.flatten()].view(-1, 3, 3)
            vertices_center = vertices_faces.mean(dim=1)
            faces_normal = torch.cross(vertices_faces[:, 1, :] - vertices_faces[:, 0, :],
                                       vertices_faces[:, 2, :] - vertices_faces[:, 1, :], dim=1)

            vertices_center_sample_list.append(vertices_center)
            faces_normal_sample_list.append(faces_normal)

            # all faces
            vertices_faces = vertices_batch[i][part_faces.flatten()].view(-1, 3, 3)
            vertices_center = vertices_faces.mean(dim=1)
            faces_normal = torch.cross(vertices_faces[:, 1, :] - vertices_faces[:, 0, :],
                                       vertices_faces[:, 2, :] - vertices_faces[:, 1, :], dim=1)
            vertices_center_list.append(vertices_center)
            faces_normal_list.append(faces_normal)

        # min distance pairs
        vertices_a = vertices_center_sample_list[0]
        normal_a = faces_normal_sample_list[0]
        vertices_b = vertices_center_list[1]
        normal_b = faces_normal_list[1]
        loss += _touch_loss(vertices_a, vertices_b, normal_a, normal_b)

        vertices_a = vertices_center_sample_list[1]
        normal_a = faces_normal_sample_list[1]
        vertices_b = vertices_center_list[0]
        normal_b = faces_normal_list[0]
        loss += _touch_loss(vertices_a, vertices_b, normal_a, normal_b)

    return loss


def pose_prior_loss(opt, pose):
    pose = pose[:, None, :, :]
    mean, std = opt.pose_prior(pose)

    loss_mean = torch.norm(mean, dim=1).mean()
    loss_std = torch.norm(std - 1., dim=1).mean()

    return loss_mean + loss_std


def coco_l2_loss(pred, target):
    loss_head = torch.sum((pred[:, 5] - target[:, 5])**2) / 5.0
    loss_body = torch.sum((pred[:, 5:] - target[:, 5:])**2)
    loss = (loss_head+loss_body) / pred.numel()
    return loss


def transl_consistency_loss(transl):
    loss = l2_loss(transl[:-1], transl[1:])
    return loss


def pose_consistency_loss(pose):
    loss = l2_loss(pose[:-1], pose[1:])
    return loss


def shape_consistency_loss(shape):
    loss = l2_loss(shape[:-1], shape[1:])
    return loss


def kp3d_consistency_loss(kp3d):
    loss = l2_loss(kp3d[:-1], kp3d[1:])
    return loss