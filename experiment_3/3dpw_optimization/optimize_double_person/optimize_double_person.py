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
from common.loss import l1_loss, l2_loss, mask_loss, part_mask_loss, smpl_collision_loss, touch_loss
from common.camera import CameraPerspective, CameraPerspectiveTorch
from common.render import PerspectivePyrender, PerspectiveNeuralRender

from preprocess import get_label, get_smpl_x, create_log
from config import opt




def submit(opt, id, loss_dict, pre_dict):
    logger = opt.logger
    label = opt.label
    render = opt.pyrender

    opt.logger.update_summary_id(id)

    if id % opt.sub_scalar_iter == 0:
        logger.scalar_summary_dict(loss_dict)

    if id % opt.sub_other_iter == 0:
        # kp2d
        img_kp2d = label['img'].copy()
        for i in range(len(label['kp2d'])):
            img_kp2d = draw_kp2d(img_kp2d,
                                 pre_dict['kp2d_body_pre'][i].detach().cpu().numpy(),
                                 radius=np.int(8*opt.image_scale))
            img_kp2d = draw_kp2d(img_kp2d,
                                 label['kp2d'][i], color=(0,255,0),
                                 radius=np.int(8*opt.image_scale))

        logger.add_image('kp2d_'+str(id), cv2.cvtColor(img_kp2d, cv2.COLOR_BGR2RGB))
        logger.save_image('kp2d_%s.png' % str(id).zfill(5), img_kp2d)


        # render
        color, depth = render.render_obj(pre_dict['vertices'].detach().cpu().numpy()[0],
                                         pre_dict['faces'].detach().cpu().numpy()[0],
                                         show_viewer=False)
        img_add_smpl = add_blend_smpl(color, depth > 0, label['img'].copy())
        logger.add_image('img_add_smpl_'+str(id), cv2.cvtColor(img_add_smpl, cv2.COLOR_BGR2RGB))
        logger.save_image('img_add_smpl_%s.png' % str(id).zfill(5), img_add_smpl)

        # add mask
        if 'mask_pre' in pre_dict:
            mask = np.zeros((label['mask'].shape[0], label['mask'].shape[1], 3), dtype=np.uint8)
            mask = draw_mask(mask, label['mask'][:, :, None], color=(0, 255, 0))
            mask = draw_mask(mask, pre_dict['mask_pre'][0][:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
            logger.add_image('mask_'+str(id), cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            logger.save_image('mask_%s.png' % str(id).zfill(5), mask)

        # add part mask
        # part_full_mask = np.zeros((label['mask'].shape[0], label['mask'].shape[1], 3), dtype=np.uint8)
        # for part_name, seg_mask in label['part_segmentation'].items():
        #     part_full_mask = draw_mask(part_full_mask, seg_mask[:, :, None], color=(0, 255, 0))
        # # for part_mask in pre_dict['part_mask_pre']:
        # #     part_full_mask = draw_mask(part_full_mask, part_mask[:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        # part_full_mask = draw_mask(part_full_mask, pre_dict['mask_pre'][:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        # logger.add_image('part_full_mask_'+str(id), cv2.cvtColor(part_full_mask, cv2.COLOR_BGR2RGB))
        # logger.save_image('part_full_mask_%s.png' % str(id).zfill(5), part_full_mask)


        # save obj
        logger.save_obj('%s.obj' % id,
                        pre_dict['vertices'].detach().cpu().numpy()[0],
                        pre_dict['faces'].detach().cpu().numpy()[0])

        # add mesh
        # logger.add_mesh('mesh_'+str(id),
        #                  pre_dict['vertices'].detach().cpu().numpy()[np.newaxis,:],
        #                  pre_dict['faces'][np.newaxis,:])


def dataset(opt):
    label = opt.label

    kp2d_gt = torch.tensor(label['kp2d'], dtype=torch.float32).to(opt.device)
    pose_reg = torch.tensor(label['pose'], dtype=torch.float32).to(opt.device)
    shape_reg = torch.tensor(label['shape'], dtype=torch.float32).to(opt.device)

    # mask
    mask_gt = torch.tensor(label['mask'], dtype=torch.float32).to(opt.device)[None, :, :]
    instance_a_gt = torch.tensor(label['instance_a'], dtype=torch.float32).to(opt.device)[None, :, :]
    instance_b_gt = torch.tensor(label['instance_b'], dtype=torch.float32).to(opt.device)[None, :, :]

    if 'part_segmentation' in label:
        part_mask_gt = np.zeros((len(label['smplx_faces']['part_name_list']),
                                 label['mask'].shape[0],
                                 label['mask'].shape[1]), dtype=np.float32)
        for i, part_name in enumerate(label['smplx_faces']['part_name_list']):
            part_mask_gt[i] = label['part_segmentation'][part_name]

        part_mask_gt = torch.tensor(part_mask_gt, dtype=torch.float32).to(opt.device)
        full_and_part_mask_gt = torch.cat((mask_gt.unsqueeze(0), part_mask_gt), dim=0)

    # part faces
    part_faces_gt = np.zeros((len(label['smplx_faces']['part_name_list']),
                              label['smplx_faces']['faces'].shape[0],
                              label['smplx_faces']['faces'].shape[1]), dtype=np.int32)
    for i, part_name in enumerate(label['smplx_faces']['part_name_list']):
        part_faces = label['smplx_faces']['smplx_part'][part_name]
        part_faces_gt[i][:part_faces.shape[0]] = part_faces
        part_faces_gt[i][part_faces.shape[0]:] = part_faces[0]
    part_faces_gt = torch.tensor(part_faces_gt, dtype=torch.int32).to(opt.device)

    # torch pair
    touch_pair_list = []
    touch_pair_list.append({
        0: torch.tensor(label['smplx_faces']['smplx_part']['body_middle_back_left'].astype(np.int64), dtype=torch.int64),
        1: torch.tensor(label['smplx_faces']['smplx_part']['right_hand'].astype(np.int64), dtype=torch.int64)
    })
    touch_pair_list.append({
        0: torch.tensor(label['smplx_faces']['smplx_part']['right_hand'].astype(np.int64), dtype=torch.int64),
        1: torch.tensor(label['smplx_faces']['smplx_part']['left_hand'].astype(np.int64), dtype=torch.int64)
    })
    touch_pair_list.append({
        0: torch.tensor(label['smplx_faces']['smplx_part']['left_hand'].astype(np.int64), dtype=torch.int64),
        1: torch.tensor(label['smplx_faces']['smplx_part']['arm_right_big'].astype(np.int64), dtype=torch.int64)
    })


    return {
        'kp2d': kp2d_gt,
        'pose_reg': pose_reg,
        'shape_reg': shape_reg,
        'mask': mask_gt,
        'instance_a': instance_a_gt,
        'instance_b': instance_b_gt,
        # 'part_mask': part_mask_gt,
        # 'full_and_part_mask': full_and_part_mask_gt,
        'part_faces': part_faces_gt,
        'touch_pair_list': touch_pair_list
    }



def optimize(opt):
    ##
    logger = opt.logger
    label = opt.label
    camera = opt.camera
    pyrender = opt.pyrender
    neural_render = opt.neural_render
    smpl_male = opt.smpl_male
    smpl_female = opt.smpl_female
    # smpl_neutral = opt.smpl_neutral

    ## gt
    opt.dataset = dataset(opt)

    ## learning parameters
    pose_iter = torch.tensor(label['pose'], dtype=torch.float32).to(opt.device)
    shape_iter = torch.tensor(label['shape'], dtype=torch.float32).to(opt.device)
    transl_iter = torch.tensor(label['trans'], dtype=torch.float32).to(opt.device)

    pose_iter.requires_grad = True
    shape_iter.requires_grad = True
    transl_iter.requires_grad = True

    global_orient = pose_iter[:, 0].view(pose_iter.shape[0], 1, -1)
    body_pose = pose_iter[:, 1:22].view(pose_iter.shape[0], 1, -1)

    optimizer = torch.optim.Adam((pose_iter, shape_iter, transl_iter), lr=opt.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)

    tqdm_iter = tqdm(range(opt.total_iter), leave=True)
    tqdm_iter.set_description(opt.exp_name)
    for it_id in tqdm_iter:
        # forword
        vertices_batch = []
        faces_batch = []
        kp3d_pre_batch = []
        for i in range(len(global_orient)):
            smpl = smpl_female
            if i == 1:
                smpl = smpl_male

            vertices, kp3d_pre, faces = smpl(global_orient=global_orient[i],
                                             body_pose=body_pose[i],
                                             betas=shape_iter[i],
                                             )

            vertices = vertices.squeeze(0) + transl_iter[i]
            kp3d_pre = kp3d_pre.squeeze(0) + transl_iter[i]

            vertices_batch.append(vertices)
            faces_batch.append(torch.tensor(faces.astype(np.int32)).to(opt.device))
            kp3d_pre_batch.append(kp3d_pre)

        vertices_batch = torch.stack(vertices_batch, dim=0)
        faces_batch = torch.stack(faces_batch, dim=0)
        kp3d_pre_batch = torch.stack(kp3d_pre_batch, dim=0)

        # kp2d pred
        kp2d_body_pre = camera.perspective(kp3d_pre_batch)

        # mask and part mask pred
        vertices_batch = camera.world_2_camera(vertices_batch)
        vertices_two_person = torch.cat((vertices_batch[0], vertices_batch[1]), dim=0)[None, ...]
        faces_two_person = torch.cat((faces_batch[0], faces_batch[1] + len(vertices_batch[0])), dim=0)[None, ...]
        # image_normal, depth = neural_render.render_obj(vertices_two_person,
        #                                                faces_two_person, None)
        if opt.mask_weight != 0:
            mask_pre = neural_render.render_mask(vertices_two_person,
                                                 faces_two_person)

        # loss
        if opt.mask_weight != 0:
            loss_mask = mask_loss(mask_pre, opt.dataset['mask'])
        else:
            loss_mask = torch.tensor([0.0]).to(opt.device)

        # loss_part_mask = part_mask_loss(mask_pre[1:], opt.dataset['part_mask'])
        loss_kp2d = l2_loss(kp2d_body_pre[:, :15], opt.dataset['kp2d'][:, :15]) + \
                    l2_loss(kp2d_body_pre[:, 16:22], opt.dataset['kp2d'][:, 16:22])
        loss_pose_reg = l1_loss(pose_iter[:, 1:22], opt.dataset['pose_reg'][:, 1:22])
        loss_shape_reg = l1_loss(shape_iter, opt.dataset['shape_reg'])
        loss_collision = smpl_collision_loss(vertices_batch, faces_batch[0])
        loss_touch = touch_loss(opt, vertices_batch)


        loss_mask = loss_mask * opt.mask_weight
        loss_kp2d = loss_kp2d * opt.kp2d_weight
        loss_pose_reg = loss_pose_reg * opt.pose_weight
        loss_shape_reg = loss_shape_reg * opt.shape_weight
        loss_collision = loss_collision * opt.collision_weight
        loss_touch = loss_touch * opt.touch_weight

        loss =  loss_mask +\
                loss_kp2d + \
                loss_pose_reg + \
                loss_shape_reg + \
                loss_collision + \
                loss_touch

        # update grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        # submit
        tqdm_iter.set_postfix_str(s='transl_iter=(%.4f,%.4f,%.4f, %.4f,%.4f,%.4f)' % \
                                    (transl_iter[0][0][0].item(),
                                     transl_iter[0][0][1].item(),
                                     transl_iter[0][0][1].item(),
                                     transl_iter[1][0][0].item(),
                                     transl_iter[1][0][1].item(),
                                     transl_iter[1][0][1].item()))

        loss_dict = {
            "loss_mask": loss_mask,
            # "loss_part_mask": loss_part_mask,
            "loss_kp2d": loss_kp2d,
            "loss_pose_reg": loss_pose_reg,
            "loss_shape_reg": loss_shape_reg,
            "loss_collision": loss_collision,
            "loss_touch": loss_touch,
            "loss": loss
        }
        pre_dict = {
            # "part_mask_pre": mask_pre[1:],
            # "mask_pre": mask_pre,
            "kp2d_body_pre": kp2d_body_pre[:, :22, :],
            "vertices": vertices_two_person,
            "faces": faces_two_person
        }

        if opt.mask_weight != 0:
            pre_dict["mask_pre"] = mask_pre

        submit(opt, it_id, loss_dict, pre_dict)


def main():
    # submit
    opt.logger = create_log(opt.exp_name)

    # label
    opt.label = get_label(img_id=128, visualize=False)

    # render and camera
    opt.pyrender = PerspectivePyrender(opt.label['intrinsic'], opt.label['pyrender_camera_pose'],
                                       width=opt.label['img'].shape[1], height=opt.label['img'].shape[0])

    K = torch.tensor(opt.label['intrinsic'][None, :, :], dtype=torch.float32).to(opt.device)
    R = torch.tensor(np.eye(3)[None, :, :], dtype=torch.float32).to(opt.device)
    t = torch.tensor(np.zeros((1, 3))[None, :, :], dtype=torch.float32).to(opt.device)
    height, width = opt.label['img'].shape[:2]
    opt.neural_render = PerspectiveNeuralRender(K, R, t, height=height, width=width)

    opt.camera = CameraPerspectiveTorch(opt.label['intrinsic'], opt.label['extrinsic'], opt.device)

    # smplx
    opt.smpl_male = get_smpl_x(gender='male', device=opt.device)
    opt.smpl_female = get_smpl_x(gender='female', device=opt.device)
    # opt.smpl_neutral = get_smpl_x(gender='neutral', device=opt.device)

    # optimize
    optimize(opt)


if __name__ == '__main__':
    main()
