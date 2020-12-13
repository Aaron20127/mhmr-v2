
import os
import pyrender
import numpy as np
import torch
import trimesh
import cv2
import h5py
import sys
import time
import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../../')

from common.utils import Rx_np, Ry_np, Ry_torch
from common.render import PerspectiveRender
from common.smpl import SMPL

def generate_box_vertex_and_face(box):
    A, B, C = box[0]
    D, E, F = box[1]

    vertex = np.array([[A, B, C],
                       [D, B, C],
                       [D, B, F],
                       [A, B, F],
                       [A, E, C],
                       [D, E, C],
                       [D, E, F],
                       [A, E, F]])

    face = np.array([[0, 1, 2],
                     [0, 2, 3],
                     [0, 7, 4],
                     [0, 3, 7],
                     [0, 4, 5],
                     [0, 5, 1],
                     [6, 2, 1],
                     [6, 1, 5],
                     [6, 5, 4],
                     [6, 4, 7],
                     [6, 3, 2],
                     [6, 7, 3]])

    return vertex, face


def save_image(save_dir, image_render_name, image_full_mask_name, render, person_obj,
               moving_radius, show_moving_area=False, show_bbox=False):
    # mesh
    mesh = []

    # pack vertex and face
    vertices = None
    faces = None

    if len(person_obj) == 2:
        vertices = np.concatenate((person_obj[0]['vertices_posed'], person_obj[1]['vertices_posed']), axis=0)
        faces = np.concatenate((person_obj[0]['faces'], person_obj[1]['faces']+6890), axis=0)

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 1.0]

        tri_mesh = trimesh.Trimesh(vertices, faces,
                                   vertex_colors = vertex_colors)
    else:
        assert 'num person != 2'

    # smooth = True: vertex shading
    # smooth = False: face shading
    mesh_obj = pyrender.Mesh.from_trimesh(tri_mesh, wireframe=False, smooth=True)
    mesh.append(mesh_obj)

    if show_moving_area:
        sphere_pos = np.array([[0., 0., 0.]])
        ms = trimesh.creation.uv_sphere(radius=moving_radius)
        ms.visual.vertex_colors = [0.0, 1.0, 0.0, 0.1]
        pts = sphere_pos
        tfs = np.tile(np.eye(4), (len(pts), 1, 1))
        tfs[:, :3, 3] = pts
        mesh_sphere = pyrender.Mesh.from_trimesh(ms, poses=tfs)

        mesh.append(mesh_sphere)

    if show_bbox:
        for obj in person_obj:
            vertices, faces = generate_box_vertex_and_face(obj['bbox'])

            vertex_colors = np.ones([vertices.shape[0], 4]) * [1.0, 0.0, 0.0, 0.5]
            tri_mesh = trimesh.Trimesh(vertices, faces,
                                       vertex_colors=vertex_colors)
            # smooth = True: vertex shading
            # smooth = False: face shading
            mesh_obj = pyrender.Mesh.from_trimesh(tri_mesh, wireframe=True, smooth=False)

            mesh.append(mesh_obj)

    # render
    render_img, depth_img = render.run(mesh, show_viwer=False)

    # save dir
    image_render_dir =  image_render_name.split('/')
    current_save_dir = os.path.join(save_dir, image_render_dir[0], image_render_dir[1], image_render_dir[2])
    os.makedirs(current_save_dir, exist_ok=True)

    image_full_mask_dir =  image_full_mask_name.split('/')
    current_save_dir = os.path.join(save_dir, image_full_mask_dir[0], image_full_mask_dir[1], image_full_mask_dir[2])
    os.makedirs(current_save_dir, exist_ok=True)

    # render_img_gray = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
    full_mask_img = (cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY) < 255) * 255

    cv2.imwrite(os.path.join(save_dir, image_render_name), render_img)
    cv2.imwrite(os.path.join(save_dir, image_full_mask_name), full_mask_img)


def generate_shape_pose_group(pose, shape, number_person=2, num_pose_sample=1):
    # pose_total_com = (len(pose) // number_person) * num_pose_sample

    shape_index_list = []
    pose_index_list = []

    if number_person == 1:
        for i in range(len(pose)):
            pose_index_list.append([i])

        for i in range(len(shape)):
            shape_index_list.append([i])

    elif number_person == 2:
        for i in range(len(shape)):
            for j in range(len(shape)):
                # if i != j:
                shape_index_list.append([i, j])

        for i in range(num_pose_sample):
            pose_index = np.arange(0,len(pose)).tolist()
            pose_com = []
            while len(pose_index) > 0:
                id = np.random.randint(len(pose_index))
                pose_id = pose_index.pop(id)
                pose_com.append(pose_id)

                if len(pose_com) < 2:
                    continue

                pose_index_list.append(pose_com)
                pose_com = []

                if len(pose_index) % 1000 == 0 and len(pose_index) != 0:
                    print("remain pose %d" % (len(pose_index)))

    return pose_index_list, shape_index_list


def person_collision(bbox):
    collision = False

    num_person = len(bbox)
    for i in range(num_person):
        for j in range(i+1, num_person):
            boxA = bbox[i]
            boxB = bbox[j]

            xA = max(boxA[0][0], boxB[0][0])
            yA = max(boxA[0][1], boxB[0][1])
            zA = max(boxA[0][2], boxB[0][2])

            xB = min(boxA[1][0], boxB[1][0])
            yB = min(boxA[1][1], boxB[1][1])
            zB = min(boxA[1][2], boxB[1][2])

            # The method of finding intersection region in space is different from that of image, we don't need add 1.
            interArea = max(0, xB - xA) * max(0, yB - yA) * max(0, zB - zA)

            if interArea > 0:
                collision = True
                break

        if collision == True:
            break

    return collision


def wall_collision(vertices_list, sphere_radius):
    collision = False

    for vertices in vertices_list:
        if np.sum(vertices**2, axis=1).max() >= sphere_radius**2:
            collision = True
            break

    return collision


def generate_person_obj(sphere_radius, poses, shapes, smpl, sample_trans=True, sample_rot=False,
                        max_sample_times=50, device='cpu'):
    person_obj_list = []

    pose = torch.from_numpy(poses.reshape(-1, 24, 3)).to(device).type(torch.float32)
    shape = torch.from_numpy(shapes.reshape(-1, 1, 10)).to(device).type(torch.float32)

    vertices, joints, faces = smpl(shape, pose)

    # vertices = vertices.cpu().numpy()
    # joints = joints.cpu().numpy()

    # pose = pose.cpu().numpy()
    # shape = shape.cpu().numpy()

    # random translation and rotation
    num_person = len(poses)

    sample_times = 0
    find = False
    while sample_times < max_sample_times and not find:
        obj_pose = torch.eye(4).repeat((num_person, 1, 1)).to(device)
        if sample_trans:
            # translation
            random_trans = sphere_radius * (2 * torch.rand(num_person, 2) - 1)

            obj_pose[:, 0, 3] = random_trans[:, 0]
            obj_pose[:, 2, 3] = random_trans[:, 1]

        if sample_rot:
            # rotation
            theta_y =  2 * np.pi * torch.rand(num_person)
            obj_pose[:, :3, :3] = Ry_torch(theta_y)

        # transpose
        homogeneous_vertices = torch.cat((vertices,
                                          torch.ones(vertices.size(0), vertices.size(1), 1).to(device)), 2)
        vertices_posed = torch.einsum("bij, bnj->bni", obj_pose, homogeneous_vertices)[:, :, :3]

        # bbox
        bbox_min = vertices_posed.min(axis=1)[0].cpu().numpy()
        bbox_max = vertices_posed.max(axis=1)[0].cpu().numpy()

        bbox = []
        for i in range(num_person):
            bbox.append([bbox_min[i], bbox_max[i]])

        # calculate person collison
        if person_collision(bbox):
            sample_times += 1
            continue

        # calculate collison between people and wall
        if wall_collision(vertices_posed.cpu().numpy(), sphere_radius):
            sample_times += 1
            continue

        for i in range(num_person):
            new_person_obj = {
                'obj_pose': obj_pose[i].cpu().numpy(),
                'pose': pose[i].cpu().numpy(),
                'shape': shape[i].cpu().numpy(),
                'vertices': vertices[i].cpu().numpy(),
                'vertices_posed': vertices_posed[i].cpu().numpy(),
                # 'joints': joints[i],
                'faces': faces,
                'bbox': bbox[i]
            }

            # add new person
            person_obj_list.append(new_person_obj)

        find = True

    valid = (sample_times < max_sample_times)

    return person_obj_list, valid


def generate_camera(camera_init_pose, camera_intrinsic, img_height, img_width,
                    camera_theta_x_list, camera_theta_y_list):
    camera_pose_list = []

    for theta_x in camera_theta_x_list:
        for theta_y in camera_theta_y_list:
            R = np.eye(4)
            R[:3, :3] = np.dot(Ry_np(theta_y * np.pi / 180.0),
                               Rx_np(theta_x * np.pi / 180.0))

            camera_pose = np.dot(R, camera_init_pose)
            camera_pose_list.append({
                    "camera_pose": camera_pose,
                    "theta_x": theta_x,
                    "theta_y": theta_y,
                    'render': PerspectiveRender(camera_intrinsic, camera_pose,
                                                width=img_width, height=img_height)
                })

    return camera_pose_list


def get_smpl_para_combination(src_path, num_shape=2):
    with h5py.File(src_path, 'r') as fp:
        shape = np.array(fp['shape'])
        pose = np.array(fp['pose'])

    return shape[:num_shape], pose


def generate_dataset():
    ## annotation name, 'train.h5' or 'val.h5'
    annotation_name = 'val.h5'
    total_pose = 10

    ## save dir
    save_dir = os.path.join(abspath, 'dataset')
    os.makedirs(save_dir, exist_ok=True)

    ## smpl para
    smpl_para = os.path.join(abspath, 'smpl_para', 'human36m.h5')

    ## load smpl
    basic_model_netural_path = os.path.join(abspath, '../../', 'data', 'basicModel_netural_lbs_10_207_0_v1.0.0.pkl')
    cocoplus_model_path = os.path.join(abspath, '../../', 'data', 'neutral_smpl_with_cocoplus_reg.pkl')

    device = 'cpu'
    smpl = SMPL(basic_model_path=basic_model_netural_path,
                cocoplus_model_path=cocoplus_model_path,
                joint_type='cocoplus').to(device)


    ## camera intrinsic and pose and half sphere area
    height = width = 256
    cx = cy = height / 2
    focal_coefficient = 2.4
    radius_coefficient = 0.325
    fov = 45.0
    d = 8.0
    camera_theta_x_list = [0]
    camera_theta_y_list = [0, 90, 180, 270]

    tan_half = np.tan(fov / 180.0 * np.pi / 2)
    height_min = d * tan_half
    max_sphere_radius = d * (tan_half / (1.0 - tan_half))
    radius = max_sphere_radius * radius_coefficient

    focal_length = (height / 2.0) / tan_half
    focal_length *= focal_coefficient

    camera_intrinsic = np.eye(3)
    camera_intrinsic[0][0] = focal_length
    camera_intrinsic[1][1] = focal_length
    camera_intrinsic[0][2] = cx
    camera_intrinsic[1][2] = cy

    camera_init_pose = np.eye(4)
    camera_init_pose[2,3] = d + radius
    camera_obj_list = generate_camera(camera_init_pose, camera_intrinsic, height, width,
                                      camera_theta_x_list, camera_theta_y_list)


    ## pose and shape
    num_person = 2

    np.random.seed(9608)
    # shape_arr = (np.random.rand(2,10) - 0.5) * 0.06 # n x 10
    # pose_arr = np.zeros((1, 72)) # n x 72
    shape_arr, pose_arr = get_smpl_para_combination(smpl_para, num_shape=1)
    pose_id_list, shape_id_list = generate_shape_pose_group(pose_arr[:total_pose], shape_arr, number_person=num_person)


    ## generate image and annotations
    num_cam = len(camera_obj_list)
    num_shape_com = len(shape_id_list)
    num_pose_com = len(pose_id_list)

    # _kp2ds = np.zeros((num_cam, num_shape_com, num_pose_com, num_person, 18, 3))
    # _kp3ds = np.zeros((num_cam, num_shape_com, num_pose_com, num_person, 19, 3))
    _valid = np.ones((num_cam, num_shape_com, num_pose_com, num_person), dtype=np.int32)
    _shape = np.zeros((num_cam, num_shape_com, num_pose_com, num_person, 10))
    _pose = np.zeros((num_cam, num_shape_com, num_pose_com, num_person, 24, 3))
    _obj_pose = np.zeros((num_cam, num_shape_com, num_pose_com, num_person, 4, 4))
    _camera_pose = np.zeros((num_cam, 4, 4))
    _camera_intrinsic = camera_intrinsic
    _image_index = np.zeros((num_cam, num_shape_com, num_pose_com), dtype=np.int64)
    _image_render_list = []
    _image_full_mask_list = []

    img_id = 0
    start_time = time.time()

    for pose_id in tqdm.tqdm(range(len(pose_id_list))):
        pose_index = pose_id_list[pose_id]
        for shape_id, shape_index in enumerate(shape_id_list):
            # t1 = time.time()
            person_obj_list, valid = generate_person_obj(radius, pose_arr[pose_index],
                                                         shape_arr[shape_index], smpl,
                                                         max_sample_times=100, device=device)
            # print("t2-t1:", time.time() - t1)

            if not valid:
                _valid[:, shape_id, pose_id, :] = 0
                print('Invalid person obj, shape_id %d, pose_id, %d' % (shape_id, pose_id))
                continue

            for camera_id, camera_obj in enumerate(camera_obj_list):
                _camera_pose[camera_id, :] = camera_obj['camera_pose']

                for obj_id, obj in enumerate(person_obj_list):
                    _shape[camera_id, shape_id, pose_id, obj_id] = obj['shape'].flatten()
                    _pose[camera_id, shape_id, pose_id, obj_id] = obj['pose']
                    _obj_pose[camera_id, shape_id, pose_id, obj_id] = obj['obj_pose']

                _image_index[camera_id, shape_id, pose_id] = len(_image_render_list)

                image_render_name = 'images_%s/camera_rx_%s_ry_%s/render/%s.jpg' % \
                                    (annotation_name[:-3],
                                     camera_obj['theta_x'],
                                     camera_obj['theta_y'],
                                     str(img_id).zfill(8))
                image_full_mask_name = 'images_%s/camera_rx_%s_ry_%s/full_mask/%s.jpg' % \
                                       (annotation_name[:-3],
                                        camera_obj['theta_x'],
                                        camera_obj['theta_y'],
                                        str(img_id).zfill(8))

                _image_render_list.append(image_render_name)
                _image_full_mask_list.append(image_full_mask_name)

                save_image(save_dir, image_render_name, image_full_mask_name, camera_obj['render'],
                           person_obj_list, radius)

            img_id += 1
            # print("%d / %d" % (total_image, img_id))

    print('total time:', time.time() - start_time)

    # save annotations
    dst_file = os.path.join(save_dir, annotation_name)
    dst_fp = h5py.File(dst_file, 'w')

    dst_fp.create_dataset('valid', data=_valid)
    dst_fp.create_dataset('shape', data=_shape)
    dst_fp.create_dataset('pose', data=_pose)
    dst_fp.create_dataset('obj_pose', data=_obj_pose)
    dst_fp.create_dataset('camera_pose', data=_camera_pose)
    dst_fp.create_dataset('camera_intrinsic', data=camera_intrinsic)
    dst_fp.create_dataset('image_index', data=_image_index)
    dst_fp.create_dataset('image_render_name', data=np.array(_image_render_list, dtype='S'))
    dst_fp.create_dataset('image_full_mask_name', data=np.array(_image_full_mask_list, dtype='S'))
    dst_fp.close()


if __name__ == '__main__':
    generate_dataset()