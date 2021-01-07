
import os
import sys
import torch


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

    faces_b_vertices = vertices_batch[1][faces.flatten().type(torch.long)].view(face_len, 3, 3).cpu()

    vector = vertices_batch[0].unsqueeze(1).cpu() - \
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



