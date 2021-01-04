import os
import sys
import pickle as pkl
import cv2
import torch
import pyrender
import trimesh
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片

from smpl_x import SMPL_X
from utils import save_obj


abspath = os.path.abspath(os.path.dirname(__file__))

from vispy import scene, io

def read_obj(dst_path):
    vertices, faces, normals, vt = io.read_mesh(dst_path)

    return {
        'vertices': vertices,
        'faces': faces,
        'normals': normals,
        'vt': vt
    }


def generate_body_part_from_mesh():
    dst_dir = os.path.join(abspath, '..\\data\\uv_map\\smplx')
    smplx_path = os.path.join(dst_dir, "smplx_new_pose_uv.obj")

    part_list = [
        'left_leg',
        'right_leg',
        'left_foot',
        'right_foot',
        'left_arm',
        'right_arm',
        'left_hand',
        'right_hand',
        'head',
        'body',
    ]

    full_label = read_obj(smplx_path)

    smplx_part = {}
    for part_name in part_list:
        part_label = read_obj(os.path.join(dst_dir, 'part_obj', '%s.obj' % part_name))

        # ori_vert_index = []
        smplx_part[part_name] = {
            'faces': np.zeros(part_label['faces'].shape, dtype=np.uint32)
        }

        tqdm_iter = tqdm(part_label['faces'], leave=True)
        tqdm_iter.set_description(part_name)

        for i, face_part in enumerate(tqdm_iter):
            vertex_part = part_label['vertices'][face_part]

            find = False
            for j, face_full in enumerate(full_label['faces']):
                vertex_full = full_label['vertices'][face_full]

                # center of face is equal
                if vertex_part.sum() == vertex_full.sum():
                    smplx_part[part_name]['faces'][i] = face_full
                    find = True
                    break

            if not find:
                print('did not find %s, face %s' % (part_name, face_part))

    ## save
    print('saving...')
    full_label['part_faces'] = smplx_part
    total_faces = 0

    for part_name, face_list in smplx_part.items():
        print(part_name + ' faces: %d' % len(face_list['faces']))
        total_faces += len(face_list['faces'])

    output = open(dst_dir + '\\smplx_part.pkl', 'wb')
    pkl.dump(full_label, output)
    output.close()

    print('done, total %d' % total_faces)


def extend_points(p):
    up = np.array([[1, 0]])
    down = np.array([[-1, 0]])
    lf = np.array([[0, 1]])
    rg = np.array([[0, -1]])

    new_p = np.concatenate((p + up, p+down, p+lf, p+rg), axis=0)
    new_p = np.unique(new_p, axis=0)

    return new_p


def generate_body_part_from_uv():

    dst_dir = os.path.join(abspath, '..\\data\\uv_map\\smplx')
    mask_dir = os.path.join(dst_dir, 'uv_mask')

    part_name_list = [
        'right_leg',
        'left_leg',
        'left_foot',
        'right_foot',
        'left_arm',
        'right_arm',
        'left_hand',
        'right_hand',
        'head',
        'body',
    ]
    part_faces_dict = {}

    # get part mask
    uv_map_mask = mpimg.imread(dst_dir + '\\smplx_uv.png')[:,:,3] > 0

    part_mask_dict = {}
    for part_name in part_name_list:
        part_mask = mpimg.imread(os.path.join(mask_dir, "%s.png" % part_name))[:,:,0] > 0
        part_mask_dict[part_name] = (uv_map_mask & part_mask).astype(np.uint8)

        part_faces_dict[part_name] = []

    # get part faces
    smplx_obj = read_obj(dst_dir + '\\smplx_uv.obj')

    tqdm_iter = tqdm(smplx_obj['faces'], leave=True)
    for i, face_id in enumerate(tqdm_iter):
        vts = smplx_obj['vt'][face_id]
        vts[:, 1] = 1.0 - vts[:, 1] # image coordinate
        vts = vts * 1023
        vts = np.round(vts).astype(np.int32)

        max_part_name = None
        max_in_mask = 0
        extend_total = 2
        for j in range(extend_total):
            for part_name, part_mask in part_mask_dict.items():
                total = 0
                for vt in vts:
                    if part_mask[vt[1], vt[0]] == 1:
                        total += 1

                if total > max_in_mask:
                    max_part_name = part_name
                    max_in_mask = total

            if max_part_name:
                break

            if j != (extend_total-1):
                vts = extend_points(vts)

        if max_part_name is None:
            print('Can not find mask')
            print('face_id\n', face_id)
            print('vts\n', vts)
            sys.exit(1)
        else:
            part_faces_dict[max_part_name].append(face_id)

    # save
    print('saving...')
    smplx_obj['part_faces'] = {}
    total_faces = 0
    for part_name, face_list in part_faces_dict.items():
        faces = np.zeros((len(face_list), 3), dtype=np.int32)
        for i, face in enumerate(face_list):
            faces[i] = face

        smplx_obj['part_faces'][part_name] = faces
        total_faces += len(faces)
        print(part_name + ' faces: %d' % len(faces))

    output = open(dst_dir + '\\smplx_part.pkl', 'wb')
    pkl.dump(smplx_obj, output)
    output.close()

    print('done, total %d' % total_faces)


def generate_new_pose_smplx_uv_obj():
    # get vertices and faces
    device = 'cpu'
    smplx_model_path = os.path.join(abspath, '../data/')
    smpl = SMPL_X(model_path=smplx_model_path, model_type='smplx', gender='male').to(device)

    body_pose = torch.zeros((1, 63), dtype=torch.float32)
    body_pose[0, 2] = np.pi / 4
    body_pose[0, 5] = -np.pi / 4

    vertices, _, faces = smpl(body_pose = body_pose)
    vertices = vertices.detach().cpu().numpy().squeeze()

    # get uv map
    smplx_uv_dir = os.path.join(abspath, '../data/uv_map/smplx')
    smplx_uv_path = os.path.join(smplx_uv_dir, 'smplx_uv.obj')
    full_label = read_obj(smplx_uv_path)

    # save
    save_path = os.path.join(smplx_uv_dir, 'smplx_new_pose_uv.obj')
    save_obj(save_path, vertices=vertices,
                  faces=full_label['faces'],
                  vt=full_label['vt'])

    # io.write_mesh(save_path, vertices=vertices,
    #               faces=full_label['faces'],
    #               normals=full_label['normals'],
    #               texcoords=full_label['vt'],
    #               overwrite=True)


def smplx_part_label():
    dst_dir = os.path.join(abspath, '..\\data\\uv_map\\smplx')

    with open(dst_dir + '\\smplx_part.pkl', 'rb') as f:
        part_label = pkl.load(f, encoding='iso-8859-1')

    return part_label


if __name__ == '__main__':
    # generate_body_part_from_mesh()
    label = smplx_part_label()
