import os
import sys
import pickle as pkl
import cv2
import pyrender
import trimesh
from tqdm import tqdm
import h5py
import torch
import numpy as np
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")

from common.debug import draw_kp2d, draw_mask, add_blend_smpl
from common.smpl_x import SMPL_X
from common.loss import l1_loss, l2_loss
from common.log import Logger
from common.camera import CameraPerspective, CameraPerspectiveTorch
from common.render import PerspectivePyrender

from config import opt


def get_label(img_id=0, gender='male', show_label=False):
    if gender == 'female':
        gender = 0
    elif gender == 'male':
        gender = 1

    label = {}

    # load data from annotation
    ann_file = os.path.join(abspath, '../data_prepare/annotation', '3dpw.h5')
    with h5py.File(ann_file, 'r') as fp:
        gt2d = np.array(fp['gt2d'])
        gt3d = np.array(fp['gt3d'])
        shape = np.array(fp['shape'])
        pose = np.array(fp['pose'])
        trans = np.array(fp['trans'])
        camera_pose_valid = np.array(fp['camera_pose_valid'])
        pose_world_2_camera = np.array(fp['pose_world_2_camera'])
        camera_intrinsic = np.array(fp['camera_intrinsic'])
        pyrender_camera_pose = np.array(fp['pyrender_camera_pose'])
        imagename = np.array(fp['imagename'])

        label["shape"] = shape[img_id][gender].reshape(1, -1)
        label["pose"] = pose[img_id][gender].reshape(-1, 3)[:22]
        label["smpl_kp3d"] = gt3d[img_id][gender].reshape(-1,3)[:22]
        label["trans"] = trans[img_id][gender].reshape(-1,3)
        label["extrinsic"] = pose_world_2_camera[img_id]
        label["intrinsic"] = camera_intrinsic
        label['pyrender_camera_pose'] = pyrender_camera_pose
        label['gender'] = 'female' if gender == 0 else 'male'

    # smpl kp3d projection
    # t = label["trans"].reshape(1, 3) # noly smpl need trans
    kp3d = label["smpl_kp3d"]

    label['kp2d'] = CameraPerspective(intrinsic=label["intrinsic"],
                                      extrinsic=label["extrinsic"]).perspective(kp3d)

    # load img and mask
    img = cv2.imread(os.path.join(abspath, "../data_prepare/3DPW/courtyard_dancing_00",
                                  'image_%s.jpg' % str(img_id).zfill(5)))
    instance = cv2.imread(os.path.join(abspath, "../data_prepare/3DPW/courtyard_dancing_00_mask/instance/",
                                       str(gender),'image_%s.png' % str(img_id).zfill(5)))

    label['img'] = img
    label['instance'] = instance

    # show
    if show_label:
        I = draw_kp2d(img, label['kp2d'])
        I = draw_mask(I, label['instance'], color=(0, 255, 0))

        cv2.namedWindow('img', 0)
        cv2.imshow('img', I)
        cv2.waitKey(0)

    return label


def get_smpl_x(gender='female', device='cpu'):
    smplx_model_path = os.path.join(abspath, '../../../data/')
    return SMPL_X(model_path=smplx_model_path, model_type='smplx', gender=gender).to(device)


def create_log(exp_name):
    log_dir = os.path.join(abspath, 'output', exp_name)
    config_path = os.path.join(abspath, 'config.py')
    return Logger(log_dir, config_path)


def submit(logger, render, id, label, loss_dict, pre_dict):
    logger.update_summary_id(id)

    if id % 9 == 0:
        logger.scalar_summary_dict(loss_dict)

    if id % 499 == 0:
        # kp2d
        img = draw_kp2d(label['img'].copy(),
                        pre_dict['kp2d_body_pre'].detach().cpu().numpy())
        img = draw_kp2d(img.copy(),
                        label['kp2d'], color=(0,255,0))

        logger.add_image('img_kp2d_'+str(id), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # render
        color, depth = render.render_obj(pre_dict['vertices'].detach().cpu().numpy(),
                                         pre_dict['faces'], show_viewer=False)
        img_add_smpl = add_blend_smpl(color, depth > 0, label['img'].copy())
        logger.add_image('img_add_smpl_'+str(id), cv2.cvtColor(img_add_smpl, cv2.COLOR_BGR2RGB))


def optimize(opt):
    ##
    logger = opt.logger
    label = opt.label
    camera = opt.camera
    render = opt.render
    smpl = opt.smpl

    ## gt
    kp2d_gt = torch.tensor(label['kp2d'], dtype=torch.float32).to(opt.device)
    pose_reg = torch.tensor(label['pose'][1:], dtype=torch.float32).to(opt.device)
    shape_reg = torch.tensor(label['shape'], dtype=torch.float32).to(opt.device)

    ## learning parameters
    pose_iter = torch.tensor(label['pose'], dtype=torch.float32).to(opt.device)
    shape_iter = torch.tensor(label['shape'], dtype=torch.float32).to(opt.device)
    transl_iter = torch.tensor(label['trans'], dtype=torch.float32).to(opt.device)
    pose_iter.requires_grad = True
    shape_iter.requires_grad = True
    transl_iter.requires_grad = True
    global_orient = pose_iter[0].view(1, -1)
    body_pose = pose_iter[1:22].view(1, -1)

    optimizer = torch.optim.Adam((pose_iter, shape_iter, transl_iter), lr=opt.lr)

    tqdm_iter = tqdm(range(opt.total_iter), leave=False)
    for it_id in tqdm_iter:
        # forword
        vertices, kp3d_pre, faces = smpl(global_orient=global_orient,
                                         body_pose=body_pose,
                                         betas=shape_iter,
                                         transl=transl_iter)
        vertices = vertices.squeeze(0)
        kp3d_pre = kp3d_pre.squeeze(0)

        # loss
        kp2d_body_pre = camera.perspective(kp3d_pre[:22])

        loss_kp2d = l2_loss(kp2d_body_pre, kp2d_gt)
        loss_pose_reg = l1_loss(pose_iter[1:22], pose_reg)
        loss_shape_reg = l1_loss(shape_iter, shape_reg)
        loss = loss_kp2d * opt.kp2d_weight + loss_pose_reg * opt.pose_weight + loss_shape_reg * opt.shape_weight

        # update grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # submit
        tqdm_iter.set_postfix_str(s='transl_iter=(%.4f,%.4f,%.4f)' % \
                                    (transl_iter[0][0].item(),
                                     transl_iter[0][1].item(),
                                     transl_iter[0][1].item()))

        loss_dict = {
            "loss_kp2d": loss_kp2d,
            "loss_pose_reg": loss_pose_reg,
            "loss_shape_reg": loss_shape_reg,
            "loss": loss
        }
        pre_dict = {
            "kp2d_body_pre": kp2d_body_pre,
            "vertices": camera.world_2_camera(vertices),
            "faces": faces
        }
        submit(logger, render, it_id, label, loss_dict, pre_dict)


def main():
    # submit
    opt.logger = create_log(opt.exp_name)

    # label
    opt.label = get_label(img_id=0, gender=opt.gender, show_label=False)

    # render and camera
    opt.render = PerspectivePyrender(opt.label['intrinsic'], opt.label['pyrender_camera_pose'],
                                     width=opt.label['img'].shape[1], height=opt.label['img'].shape[0])
    opt.camera = CameraPerspectiveTorch(opt.label['intrinsic'], opt.label['extrinsic'], opt.device)

    # smplx
    opt.smpl = get_smpl_x(gender=opt.label['gender'], device=opt.device)

    # optimize
    optimize(opt)


if __name__ == '__main__':
    main()
