import os
import sys
import pickle as pkl
import cv2
import pyrender
import trimesh
import h5py
import torch
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../")


import pickle
from common.smpl import SMPL
from common.smpl_np import SMPL_np
from common.smpl_x import SMPL_X
from common.render import perspective_render_obj_debug
from common.debug import draw_kp2d
from common.utils import save_obj

import warnings
import numpy as np
warnings.simplefilter("always")

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def perspective_render_obj(camera_pose, K, mesh, width, height, show_viwer=False):

    scene = pyrender.Scene()

    camera=pyrender.camera.IntrinsicsCamera(
            fx=K[0][0], fy=K[1,1],
            cx=K[0][2], cy=K[1][2])
    scene.add(camera, pose=camera_pose)

    # add mesh
    for m in mesh:
        scene.add(m)

    if show_viwer:
        pyrender.Viewer(scene, use_raymond_lighting=True)

    # add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=20)
    scene.add(light, pose=camera_pose)

    # render
    r = pyrender.OffscreenRenderer(viewport_width=width,viewport_height = height,point_size = 1.0)
    color, depth = r.render(scene)

    return color, depth


def add_blend_smpl(render_img, mask, img_raw):
    new_mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    color = render_img * new_mask + img_raw * (1 - new_mask)

    return color.astype(np.uint8)


""" rotation """
def Rx_mat(theta):
    """绕x轴旋转
        batch x theta
    """
    cos = np.cos(theta)
    sin = np.sin(theta)

    M = np.zeros((4, 4))
    M[0, 0]=1
    M[1, 1]=cos
    M[1, 2]=-sin
    M[2, 1]=sin
    M[2, 2]=cos
    M[3, 3]=1

    return M


def Ry_mat(theta):
    """绕y轴旋转
    """
    cos = np.cos(theta)
    sin = np.sin(theta)

    M = np.zeros((4, 4))

    M[1, 1]=1
    M[ 0, 0]=cos
    M[0, 2]=sin
    M[2, 0]=-sin
    M[2, 2]=cos
    M[3, 3] = 1

    return M


def Rz_mat(theta):
    """绕z轴旋转
    """
    cos = np.cos(theta)
    sin = np.sin(theta)

    M = np.zeros((3, 3))

    M[2, 2]=1
    M[0, 0]=cos
    M[0, 1]=-sin
    M[1, 0]=sin
    M[1, 1]=cos

    return M


def PW3D_visualization(save_data=False, view_data=True):
    # 'smpl_x', 'smpl_torch', 'smpl_np'
    smpl_type = 'smpl_x'

    # smpl
    has_cloth = False

    # basic_model_path = 'G:\\paper\\code\\master\\data\\basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    basic_model_m_path = 'G:\\paper\\code\\master\\data\\basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    basic_model_f_path = 'G:\\paper\\code\\master\\data\\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    cocoplus_model_path = 'G:\\paper\\code\\master\\data\\neutral_smpl_with_cocoplus_reg.pkl'
    smplx_model_path = 'G:\\paper\\code\\mhmr-v2\\data'

    if smpl_type == 'smpl_torch':
        device = 'cpu'
        smpl_m = SMPL(basic_model_path=basic_model_m_path,
                   cocoplus_model_path=cocoplus_model_path,
                   joint_type='basic').to(device)

        smpl_f = SMPL(basic_model_path=basic_model_f_path,
                   cocoplus_model_path=cocoplus_model_path,
                   joint_type='basic').to(device)

    elif smpl_type == 'smpl_np':
        smpl_m = SMPL_np(basic_model_m_path, joint_type='smpl')
        smpl_f = SMPL_np(basic_model_f_path, joint_type='smpl')
    elif smpl_type == 'smpl_x':
        device = 'cpu'
        smpl_m = SMPL_X(model_path=smplx_model_path, model_type='smplx', gender='male').to(device)
        smpl_f = SMPL_X(model_path=smplx_model_path, model_type='smplx', gender='female').to(device)

    # file name
    # handle_file = 'courtyard_basketball_00.pkl'
    # data_split = 'train'
    handle_file = 'courtyard_dancing_00.pkl'
    data_split = 'validation'
    data_path = 'F:/paper/dataset/3DPW/sequenceFiles'


    filename = os.path.join(data_path, data_split, handle_file)


    # get data
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        smpl_pose = data['poses']
        smpl_shape = data['betas']
        smpl_shape_clothed = data['betas_clothed']
        smpl_v_template_clothed = data['v_template_clothed']
        poses2d = data['poses2d']
        poses3d = data['jointPositions']
        global_poses = data['cam_poses']
        genders = data['genders']
        valid = np.array(data['campose_valid']).astype(np.bool)
        num_people = len(smpl_pose)
        num_frames = len(smpl_pose[0])
        seq_name = str(data['sequence'])
        img_names = np.array(
            ['F:/paper/dataset/3DPW/imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
        trans_data = data['trans']
        cam_intrinsics = data['cam_intrinsics']


        for img_id in range(0, num_frames):
            pose = []
            shape = []
            shape_clothed = []
            v_template_clothed = []
            kp2d = []
            kp3d = []
            trans = []
            imgname = img_names[img_id]
            world_2_cam = global_poses[img_id]
            gender = []

            for i in range(num_people):
                # if valid[i][img_id] > 0:
                pose.append(smpl_pose[i][img_id])
                shape.append(smpl_shape[i][:10])
                # shape_clothed.append(smpl_shape_clothed[i]) # smpl_v_1.1.0, shape 300
                shape_clothed.append(smpl_shape_clothed[i][:10]) # smpl_v_1.0.0, shape 10
                v_template_clothed.append(smpl_v_template_clothed[i])

                kp2d.append(poses2d[i][img_id].T)
                kp3d.append(poses3d[i][img_id].T)
                trans.append(trans_data[i][img_id])

                gender.append(genders[i])


            if len(pose) == 0:
                continue


            new_para_obj = {}
            for i in range(len(pose)):
            # for i in range(1):
                # np.random.seed(9608)
                # pose = (np.random.rand(24,3) - 0.5) * 0.4
                # shape = (np.random.rand(1,10) - 0.5) * 0.06
                #
                # pose = torch.from_numpy(pose).to(device).type(torch.float32)
                # shape = torch.from_numpy(shape).to(device).type(torch.float32)

                if smpl_type == 'smpl_torch':
                    pose_smpl = torch.tensor(pose[i]).reshape((24,3)).to(device).float()
                elif smpl_type == 'smpl_np':
                    pose_smpl = pose[i].reshape((24,3))
                elif smpl_type == 'smpl_x':
                    pose_smpl = torch.tensor(pose[i]).reshape((24, 3)).to(device).float()

                if has_cloth:
                    if smpl_type == 'smpl_torch':
                        shape_smpl = torch.tensor(shape_clothed[i]).reshape((1, 10)).to(device).float()
                        v_template_smpl = torch.tensor(v_template_clothed[i]).to(device).float()
                    elif smpl_type == 'smpl_np':
                        shape_smpl = shape_clothed[i].reshape((1, 10))
                        v_template_smpl = v_template_clothed[i]
                    elif smpl_type == 'smpl_x':
                        shape_smpl = torch.tensor(shape_clothed[i]).reshape((1, 10)).to(device).float()
                else:
                    if smpl_type == 'smpl_torch':
                        shape_smpl = torch.tensor(shape[i]).reshape((1, 10)).to(device).float()
                    elif smpl_type == 'smpl_np':
                        shape_smpl = shape[i].reshape((1, 10))
                        v_template_smpl = None
                    elif smpl_type == 'smpl_x':
                        shape_smpl = torch.tensor(shape_clothed[i]).reshape((1, 10)).to(device).float()

                smpl_now = smpl_m
                basic_model_path = basic_model_m_path
                if gender[i] == 'f':
                    smpl_now = smpl_f
                    basic_model_path = basic_model_f_path

                if smpl_type == 'smpl_torch':
                    verts, joints, faces = smpl_now(shape_smpl, pose_smpl, v_template_smpl)
                    verts = verts[0]
                    J = joints[0]

                elif smpl_type == 'smpl_np':
                    smpl_now.set_params(beta=shape_smpl.flatten(),
                                        pose=pose_smpl,
                                        v_template=v_template_smpl)
                    obj = smpl_now.get_obj()
                    verts = obj['verts']
                    J = obj['J']
                    faces = obj['faces']

                elif smpl_type == 'smpl_x':
                    global_orient = pose_smpl[0].view(1,-1)
                    body_pose = pose_smpl[1:22].view(1,-1)

                    verts, J, faces = smpl_now(global_orient=global_orient,
                                               body_pose=body_pose,
                                               betas=shape_smpl)
                    verts = verts.detach().cpu().numpy().squeeze()
                    J = J.detach().cpu().numpy().squeeze()

            # if save_obj:
                #     if has_cloth:
                #         # smpl_now.save_obj(verts[0], 'output/%s_clothed_%d_%d.obj' % (basic_model_path.split('\\')[-1],j,i))
                #         smpl_now.save_obj('output/%s_clothed_%d_%d.obj' % (basic_model_path.split('\\')[-1],j,i))
                #     else:
                #         # smpl_now.save_obj(verts[0], 'output/%s_%d_%d.obj' % (basic_model_path.split('\\')[-1],j,i))
                #         smpl_now.save_obj('output/%s_%d_%d.obj' % (basic_model_path.split('\\')[-1],j,i))

                t = trans[i].reshape(3, 1)

                verts_trans = verts + t.reshape(1, 3)

                homogeneous_vertices = np.concatenate((verts_trans, np.ones((verts_trans.shape[0], 1))), axis=1)
                verts_camera = np.einsum("ij, nj->ni", world_2_cam, homogeneous_vertices)[:, :3]

                new_verts = verts_camera


                # add mesh
                if new_para_obj:
                    new_para_obj['faces'] = np.concatenate((new_para_obj['faces'], faces + len(new_para_obj['verts'])), axis=0)
                    new_para_obj['verts'] = np.concatenate((new_para_obj['verts'], new_verts), axis=0)
                else:
                    new_para_obj['faces'] = faces
                    new_para_obj['verts'] = new_verts


                # kp2d_project, kp3d come from smpl joints
                K = cam_intrinsics
                t = trans[i].reshape(3, 1)
                verts_trans = kp3d[i].reshape(24, 3) # no pose trans

                homogeneous_vertices = np.concatenate((verts_trans, np.ones((verts_trans.shape[0], 1))), axis=1)
                verts_camera = np.einsum("ij, nj->ni", world_2_cam, homogeneous_vertices)[:, :3]
                verts_normal = verts_camera / verts_camera[:, 2].reshape(verts_camera.shape[0], 1)

                kp2d_project = np.einsum("ij, nj->ni", K, verts_normal)

                if 'kp2d_project' not in new_para_obj:
                    new_para_obj['kp2d_project'] = kp2d_project.reshape(1, kp2d_project.shape[0], kp2d_project.shape[1])
                else:
                    new_para_obj['kp2d_project'] = np.concatenate((new_para_obj['kp2d_project'],
                                                                   kp2d_project.reshape(1, kp2d_project.shape[0], kp2d_project.shape[1])), axis=0)

            # render
            img = cv2.imread(imgname)
            camera_pose = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])

            # get mesh
            mesh = []
            vertex_colors = np.ones([new_para_obj['verts'].shape[0], 4]) * [1.0, 1.0, 1.0, 1.0]
            tri_mesh = trimesh.Trimesh(new_para_obj['verts'], new_para_obj['faces'],
                                       vertex_colors=vertex_colors)
            mesh_obj = pyrender.Mesh.from_trimesh(tri_mesh)
            mesh.append(mesh_obj)

            # render
            color, depth = perspective_render_obj(camera_pose, cam_intrinsics, mesh, width=img.shape[1],
                                                  height=img.shape[0], show_viwer=False)
            img_add_smpl = add_blend_smpl(color, depth>0, img)


            # draw 2d joint
            img_kp2d = img.copy()
            for p in kp2d:
                img_kp2d = draw_kp2d(img_kp2d, p, draw_conf=True, draw_num=False)


            # draw 2d project joint
            img_kp2d_project = img.copy()
            for p in new_para_obj['kp2d_project']:
                img_kp2d_project = draw_kp2d(img_kp2d_project, p, draw_conf=False, draw_num=True)


            # save data
            if save_data:
                output_dir = os.path.join(abspath, 'output', smpl_type)

                img_kp2d_dir = os.path.join(output_dir, 'kp2d')
                img_kp2d_project_dir = os.path.join(output_dir, 'kp2d_project')
                img_add_smpl_dir = os.path.join(output_dir, 'add_smpl')
                obj_dir = os.path.join(output_dir, 'obj')

                os.makedirs(img_kp2d_dir, exist_ok=True)
                os.makedirs(img_kp2d_project_dir, exist_ok=True)
                os.makedirs(img_add_smpl_dir, exist_ok=True)
                os.makedirs(obj_dir, exist_ok=True)

                cv2.imwrite(os.path.join(img_kp2d_dir, "%s.jpg" % str(img_id).zfill(4)), img_kp2d)
                cv2.imwrite(os.path.join(img_kp2d_project_dir, "%s.jpg" % str(img_id).zfill(4)), img_kp2d_project)
                cv2.imwrite(os.path.join(img_add_smpl_dir, "%s.jpg" % str(img_id).zfill(4)), img_add_smpl)

                save_obj(os.path.join(obj_dir, "%s.obj" % str(img_id).zfill(4)),
                         new_para_obj['verts'], new_para_obj['faces'])

                print("%d / %d" % (img_id, num_frames))

            # show
            if view_data:
                cv2.namedWindow('img_kp2d', 0)
                cv2.imshow('img_kp2d', img_kp2d)

                cv2.namedWindow('img_kp2d_project', 0)
                cv2.imshow('img_kp2d_project', img_kp2d_project)

                cv2.namedWindow('img_add_smpl', 0)
                cv2.imshow('img_add_smpl', img_add_smpl)

                cv2.waitKey(0)



def generate_data():

    # file name
    handle_file = 'courtyard_dancing_00.pkl'
    data_split = 'validation'
    data_path = 'F:/paper/dataset/3DPW/sequenceFiles'

    _kp2d = np.zeros((84000, 2, 18, 3))
    _kp3d = np.zeros((84000, 2, 72)) # kp3d come from smpl joint
    _shape = np.zeros((84000, 2, 10))
    _pose = np.zeros((84000, 2, 72))
    _smpl_trans = np.zeros((84000, 2, 3))
    _camera_pose_valid = np.zeros((84000, 2))

    _camera_intrinsic = np.zeros((3, 3))
    _pose_world_2_camera = np.zeros((84000, 4, 4))

    _pyrender_camera_pose = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])

    _imagename = []


    filename = os.path.join(data_path, data_split, handle_file)
    # get data
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        smpl_pose = data['poses']
        smpl_shape = data['betas']
        smpl_shape_clothed = data['betas_clothed']
        smpl_v_template_clothed = data['v_template_clothed']
        poses2d = data['poses2d']
        poses3d = data['jointPositions']
        global_poses = data['cam_poses']
        genders = data['genders']
        valid = np.array(data['campose_valid']).astype(np.bool)
        num_people = len(smpl_pose)
        num_frame = len(smpl_pose[0])
        seq_name = str(data['sequence'])

        img_names = np.array(
            ['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frame)])
        smpl_trans = data['trans']

        _camera_intrinsic = data['cam_intrinsics']


        for img_id in range(0, num_frame):
            _imagename.append(img_names[img_id])
            _pose_world_2_camera[img_id] = global_poses[img_id]

            for i in range(num_people):
                num_gender = i

                _camera_pose_valid[img_id][num_gender] = valid[num_gender][img_id]
                _pose[img_id][num_gender] = smpl_pose[num_gender][img_id]
                _shape[img_id][num_gender] = smpl_shape[num_gender][:10]
                _kp2d[img_id][num_gender] = poses2d[num_gender][img_id].T
                _kp3d[img_id][num_gender] = poses3d[num_gender][img_id].T
                _smpl_trans[img_id][num_gender] = smpl_trans[num_gender][img_id]

            print('%d / %d' % (img_id, num_frame))

    dst_file = os.path.join(abspath, 'annotation', '3dpw.h5')
    dst_fp = h5py.File(dst_file, 'w')

    dst_fp.create_dataset('gt2d', data=_kp2d[:num_frame])
    dst_fp.create_dataset('gt3d', data=_kp3d[:num_frame])
    dst_fp.create_dataset('shape', data=_shape[:num_frame])
    dst_fp.create_dataset('pose', data=_pose[:num_frame])
    dst_fp.create_dataset('smpl_trans', data=_smpl_trans[:num_frame])
    dst_fp.create_dataset('camera_pose_valid', data=_camera_pose_valid[:num_frame])
    dst_fp.create_dataset('pose_world_2_camera', data=_pose_world_2_camera[:num_frame])
    dst_fp.create_dataset('camera_intrinsic', data=_camera_intrinsic)
    dst_fp.create_dataset('pyrender_camera_pose', data=_pyrender_camera_pose)
    dst_fp.create_dataset('imagename', data=np.array(_imagename[:num_frame], dtype='S'))

    dst_fp.close()

    print('save {}'.format(num_frame))
    print('done, total {}'.format(num_frame))



if __name__ == '__main__':
    # PW3D_visualization(save_data=True, view_data=False)
    generate_data()
