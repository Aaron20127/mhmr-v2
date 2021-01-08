
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


def smpl_collision_loss(vertices_batch, faces):
    face_len = faces.shape[0]
    vertices_len = vertices_batch[0].shape[0]
    vertices_a = vertices_batch[0].cpu()
    vertices_b = vertices_batch[1].cpu()

    faces_b_vertices = vertices_a[faces.flatten().type(torch.long)].view(face_len, 3, 3)

    vector = vertices_b.unsqueeze(1) - \
             torch.mean(faces_b_vertices, dim=1, keepdim=True).permute(1, 0, 2).\
             expand(vertices_len, face_len, 3)
    vector_distance = torch.sqrt(torch.sum(vector**2, dim=2))
    min_vector_distance_index = torch.argmin(vector_distance, dim=1)

    # cuda to cpu
    # faces_b_vertices = faces_b_vertices.cuda()
    # vector = vector.cuda()
    # vector_distance = vector_distance.cuda()
    # min_vector_distance_index = min_vector_distance_index.cuda()

    faces_b_vertices = faces_b_vertices[min_vector_distance_index]
    vector = vector.gather(1, min_vector_distance_index.view(vertices_len,1,1).expand(vertices_len,1,3)).squeeze(1)
    vector_distance = vector_distance.gather(1, min_vector_distance_index.view(vertices_len,1)).squeeze(1)
    # torch.cuda.empty_cache()

    faces_b_normal = torch.cross(faces_b_vertices[:, 1, :] - faces_b_vertices[:, 0, :],
                                 faces_b_vertices[:, 2, :] - faces_b_vertices[:, 1, :], dim=1)
    loss_index = (torch.sum(faces_b_normal * vector, dim=1) < 0)
    loss = vector_distance[loss_index].sum()

    return loss.cuda()


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

    return loss_distance + loss_normal


def touch_loss(opt, vertices_batch, faces):

    loss = torch.tensor([0.0]).to(opt.device)
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
