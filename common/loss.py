
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


def ground_loss(kp3d_pred, ground_normal):
    human_vec = torch.mean(kp3d_pred[:,:,16:18, :], dim=2) -\
                torch.mean(kp3d_pred[:,:,4:6, :], dim=2)
    human_normal = human_vec / torch.norm(human_vec, dim=2, keepdim=True)
    ground_normal = ground_normal[None, ...].expand_as(human_normal)

    loss = torch.sum(torch.sum(human_normal * ground_normal, dim=2) + 1)
    loss = loss / (human_normal.shape[0] * human_normal.shape[1])

    return loss


def kp2d_loss(pred, target, kp2d_weight_list):
    kp2d_weight_list = kp2d_weight_list[None, None, :, None]
    loss = torch.sum((pred - target)**2 * kp2d_weight_list)
    loss = loss / pred.numel()
    return loss


def mask_loss(pred, target, weight):
    out_point = pred[(((1 - target) + pred) > 1)]
    in_point = pred[((target + pred) > 1)]
    loss = weight[0] * torch.sum(in_point**2) + weight[1] * torch.sum(out_point**2)
    loss = loss / pred.numel()
    return loss


def part_mask_loss(pred, target):
    out = pred[(((1 - target) + pred) > 1)]
    loss = 0.5 * torch.sum((pred - target)**2) + 1.0 * torch.sum(out**2)
    loss = loss / pred.numel()
    return loss


def _smpl_collision_loss(vertices_a, vertices_center_b, normal_b):
    # min distance index
    va_2_vb_center = vertices_a.unsqueeze(2) - \
                        vertices_center_b.unsqueeze(1).expand( \
                                            vertices_a.shape[0],
                                            vertices_a.shape[1],
                                            vertices_center_b.shape[1],
                                            vertices_center_b.shape[2])
    min_distance_index = torch.argmin(torch.norm(va_2_vb_center, dim=3), dim=2)

    # matching face_center and normal
    min_distance_index_expand = min_distance_index.unsqueeze(2).expand(
                                            min_distance_index.shape[0],
                                            min_distance_index.shape[1],
                                            vertices_center_b.shape[2])
    vertices_center_b = vertices_center_b.gather(1, min_distance_index_expand)
    normal_b = normal_b.gather(1, min_distance_index_expand)

    # points index of a in b
    va_2_vb_center = vertices_a - vertices_center_b
    inner_index = ((va_2_vb_center * normal_b).sum(dim=2) < 0)

    # distance and number of collision vertex
    loss = torch.tensor(0.).cuda()
    if inner_index.sum() > 0:
        collisoin_distance = torch.norm(va_2_vb_center[inner_index], dim=1)
        loss = collisoin_distance.sum() / inner_index.shape[1]

    return loss


def _smpl_collision_batch_loss(vertices_batch, faces):

    num_img = vertices_batch.shape[0]
    vertices_center_list = []
    faces_normal_list = []
    # vertices_center_sample_list = []
    # faces_normal_sample_list = []

    vertices_batch = vertices_batch.permute(1, 0, 2, 3)

    for i, vertices in enumerate(vertices_batch):
        # all faces
        faces_vertices = vertices_batch[i][:, faces.flatten().type(torch.long)].view(num_img, -1, 3, 3)
        vertices_center = faces_vertices.mean(dim=2)
        faces_normal = torch.cross(faces_vertices[:, :, 1, :] - faces_vertices[:, :, 0, :],
                                   faces_vertices[:, :, 2, :] - faces_vertices[:, :, 1, :], dim=2)
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


def smpl_collision_loss(vertices_batch, faces, batch_size):
    """
    vertices_batch.shape = [5, 2, 10457, 3]
    faces.shape = [20908, 3]
    batch_size = 4
    """
    loss = torch.tensor(0.).cuda()
    batch = vertices_batch.shape[0] // batch_size
    hatch_rest = vertices_batch.shape[0] % batch_size

    for i in range(batch):
        loss += _smpl_collision_batch_loss(
                    vertices_batch[i*batch_size:(i+1)*batch_size], faces)

    if hatch_rest > 0:
        loss += _smpl_collision_batch_loss(
                    vertices_batch[batch*batch_size:], faces)

    return loss


def _smpl_collision_loss_back(vertices_a, vertices_center_b, normal_b):
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


def smpl_collision_loss_back(vertices_batch, faces):

    num_img = vertices_batch.shape[0]
    vertices_center_list = []
    faces_normal_list = []
    # vertices_center_sample_list = []
    # faces_normal_sample_list = []

    vertices_batch = vertices_batch.permute(1, 0, 2, 3)

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


def texture_render_loss(img_pred, img_gt, mask):
    loss = l2_loss(img_pred * mask[..., None], img_gt * mask[..., None])
    return loss


def texture_temporal_consistency_loss(textures):
    loss = l2_loss(textures[:-1], textures[1:])
    return loss


def texture_part_consistency_loss(textures, part_vertices_list):
    num_img = textures.shape[0]
    textures_new = textures.view(num_img, 2, -1, 192)
    loss = 0

    part_list = [part_vertices_list['arm_right'], part_vertices_list['left_hand']]

    for part in part_list:
        vertex_index = part[None, None, :, None].expand(num_img, 2, -1, 192)
        arm_right = textures_new.gather(2, vertex_index)
        texture_index = arm_right.abs().sum(dim=3) > 0

        for i in range(num_img):
            for j in range(2): # todo: male and female
                # if j == 1:
                #     values = arm_right[i, j][texture_index[i, j]]
                #     loss += torch.var(values, dim=0).mean()

                values = arm_right[i, j][texture_index[i, j]]
                loss += torch.var(values, dim=0).mean()


    # vertex_index = part_vertices_list['right_hand'][None, None, :, None].expand(num_img, 2, -1, 192)
    # arm_right = textures_new.gather(2, vertex_index)
    # texture_index = arm_right.abs().sum(dim=3) > 0
    #
    # for i in range(num_img):
    #     for j in range(2): # todo: male and female
    #         # if j == 1:
    #         #     values = arm_right[i, j][texture_index[i, j]]
    #         #     loss += torch.var(values, dim=0).mean()
    #         values = arm_right[i, j][texture_index[i, j]]
    #         loss += torch.var(values, dim=0).mean()

    return loss
