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
from common.loss import coco_l2_loss, l2_loss, mask_loss, part_mask_loss
from common.loss import smpl_collision_loss, touch_loss, pose_prior_loss

from common.camera import CameraPerspective, CameraPerspectiveTorch, CameraPerspectiveTorchMultiImage
from common.render import PerspectivePyrender, PerspectiveNeuralRender
from common.pose_prior import PosePrior

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
        imgs = label['img'].copy()
        for img_id, img in enumerate(imgs):
            kp2d_gt = label['kp2d'][img_id].reshape(-1, 2)
            kp2d = pre_dict['body_kp2d'][img_id].reshape(-1, 2)

            img_kp2d = draw_kp2d(img, kp2d,
                                 radius=np.int(8*opt.image_scale))
            img_kp2d = draw_kp2d(img_kp2d, kp2d_gt, color=(0, 255, 0),
                                 radius=np.int(8*opt.image_scale))

            logger.add_image('img_%s/kp2d_%s' % (str(img_id).zfill(5), str(id)),
                             cv2.cvtColor(img_kp2d, cv2.COLOR_BGR2RGB))
            logger.save_image('kp2d_%s.png' % str(id).zfill(5), img_kp2d, img_id=img_id)


        # render
        imgs = label['img'].copy()
        for img_id, img in enumerate(imgs):
            img_render, img_depth = render.render_obj(pre_dict['vertices'][img_id],
                                                      pre_dict['faces'][img_id],
                                                      show_viewer=False)
            img_add_smpl = add_blend_smpl(img_render, img_depth > 0, img)
            logger.add_image('img_%s/img_add_smpl_%s' % (str(img_id).zfill(5), str(id)),
                             cv2.cvtColor(img_add_smpl, cv2.COLOR_BGR2RGB))
            logger.save_image('img_add_smpl_%s.png' % str(id).zfill(5), img_add_smpl, img_id=img_id)


        # add mask
        if 'mask' in pre_dict:
            for img_id, mask_pre in enumerate(pre_dict['mask']):
                mask_gt = label['mask'][img_id]

                mask = np.zeros((mask_gt.shape[0], mask_gt.shape[1], 3), dtype=np.uint8)
                mask = draw_mask(mask, mask_gt[:, :, None], color=(0, 255, 0))
                mask = draw_mask(mask, mask_pre[:, :, None], color=(0, 0, 255))

                logger.add_image('img_%s/mask_%s' % (str(img_id).zfill(5), str(id)),
                                 cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                logger.save_image('mask_%s.png' % str(id).zfill(5), mask, img_id=img_id)


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
        for img_id in range(opt.num_img):
            logger.save_obj('%s.obj' % img_id,
                            pre_dict['vertices'][img_id],
                            pre_dict['faces'][img_id],
                            img_id=img_id)

        # add mesh
        # logger.add_mesh('mesh_'+str(id),
        #                  pre_dict['vertices'].detach().cpu().numpy()[np.newaxis,:],
        #                  pre_dict['faces'][np.newaxis,:])


def dataset(opt):
    label = opt.label

    kp2d_gt = torch.tensor(label['kp2d'], dtype=torch.float32).to(opt.device)
    # pose_reg = torch.tensor(label['pose'], dtype=torch.float32).to(opt.device)
    # shape_reg = torch.tensor(label['shape'], dtype=torch.float32).to(opt.device)
    shape_reg = torch.tensor(label['mean_shape'][None, None, ...].\
                             repeat(opt.num_img, axis=0).repeat(2, axis=1),
                             dtype=torch.float32).to(opt.device)

    # mask
    mask_gt = torch.tensor(label['mask'], dtype=torch.float32).to(opt.device)
    # instance_a_gt = torch.tensor(label['instance_a'], dtype=torch.float32).to(opt.device)[None, :, :]
    # instance_b_gt = torch.tensor(label['instance_b'], dtype=torch.float32).to(opt.device)[None, :, :]

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
        # 'pose_reg': pose_reg,
        'shape_reg': shape_reg,
        'mask': mask_gt,
        # 'instance_a': instance_a_gt,
        # 'instance_b': instance_b_gt,
        # 'part_mask': part_mask_gt,
        # 'full_and_part_mask': full_and_part_mask_gt,
        'part_faces': part_faces_gt,
        'touch_pair_list': touch_pair_list
    }


def init_para(opt):
    label = opt.label
    faces = label['smplx_faces']['faces'].astype(dtype=np.int32)
    vertices = label['smplx_faces']['vertices'].astype(dtype=np.int32)
    faces_two_person_batch = np.concatenate((faces, faces+len(vertices)), axis=0)[None, ...].\
                                            repeat(opt.num_img, axis=0).astype(dtype=np.int32)

    pose_0_9 = label['mean_pose'][:10][None, None, ...].repeat(opt.num_img, axis=0).\
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_12_21 = label['mean_pose'][12:][None, None, ...].repeat(opt.num_img, axis=0).\
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_10_11 = label['mean_pose'][10:12][None, None, ...].repeat(opt.num_img, axis=0).\
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    transl = label['trans']
    shape = label['mean_shape'][None, None, ...].repeat(opt.num_img, axis=0).\
                                                repeat(2, axis=1).astype(dtype=np.float32)
    left_hand_pose = np.zeros((opt.num_img, 2, 6), dtype=np.float32)
    right_hand_pose = np.zeros((opt.num_img, 2, 6), dtype=np.float32)

    jaw_pose = np.zeros((opt.num_img, 2, 3), dtype=np.float32)
    leye_pose = np.zeros((opt.num_img, 2, 3), dtype=np.float32)
    reye_pose = np.zeros((opt.num_img, 2, 3), dtype=np.float32)
    expression = np.zeros((opt.num_img, 2, 10), dtype=np.float32)

    return {
        'pose_0_9': pose_0_9,
        'pose_12_21': pose_12_21,
        'pose_10_11': pose_10_11,
        'shape': shape,
        'transl': transl,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'expression': expression,
        'faces': faces,
        'faces_two_person_batch': faces_two_person_batch
    }


def loss_f(opt, mask, kp2d_body, body_pose, shape, vertices_batch, faces):
    loss_mask, \
    loss_kp2d, \
    loss_pose_reg, \
    loss_shape_reg,\
    loss_collision, \
    loss_touch, \
    loss_pose_prior = torch.zeros(7).to(opt.device)


    # mask
    if opt.mask_weight > 0:
        loss_mask = mask_loss(mask, opt.dataset['mask'])
    # loss_part_mask = part_mask_loss(mask_pre[1:], opt.dataset['part_mask'])

    # kp2d
    if opt.kp2d_weight > 0:
        kp2d_coco = kp2d_body[:, :, opt.label["kp2d_smplx_2_coco"]]
        # loss_kp2d = coco_l2_loss(coco_pred, opt.dataset['kp2d'])
        loss_kp2d = l2_loss(kp2d_coco, opt.dataset['kp2d'])

    # pose and shape reg
    if opt.pose_reg_weight > 0:
        loss_pose_reg = l2_loss(pose_iter[:, 1:22], opt.dataset['pose_reg'][:, 1:22])
    if opt.shape_reg_weight > 0:
        loss_shape_reg = l2_loss(shape, opt.dataset['shape_reg'])

    # loss_collision
    if opt.collision_weight > 0:
        loss_collision = smpl_collision_loss(vertices_batch, faces)

    # loss_touch
    if opt.touch_weight > 0:
        loss_touch = touch_loss(opt, vertices_batch)

    # loss_pose_prior
    if opt.pose_prior_weight > 0:
        loss_pose_prior = pose_prior_loss(opt, body_pose.view(-1, 21, 3))


    loss_mask = loss_mask * opt.mask_weight
    loss_kp2d = loss_kp2d * opt.kp2d_weight
    loss_pose_reg = loss_pose_reg * opt.pose_reg_weight
    loss_shape_reg = loss_shape_reg * opt.shape_reg_weight
    loss_collision = loss_collision * opt.collision_weight
    loss_touch = loss_touch * opt.touch_weight
    loss_pose_prior = loss_pose_prior * opt.pose_prior_weight

    loss =  loss_mask + \
            loss_kp2d + \
            loss_pose_reg + \
            loss_shape_reg + \
            loss_collision + \
            loss_touch + \
            loss_pose_prior

    loss_dict = {
        "loss_mask": loss_mask.detach().cpu().numpy(),
        # "loss_part_mask": loss_part_mask,
        "loss_kp2d": loss_kp2d.detach().cpu().numpy(),
        "loss_pose_reg": loss_pose_reg.detach().cpu().numpy(),
        "loss_shape_reg": loss_shape_reg.detach().cpu().numpy(),
        "loss_collision": loss_collision.detach().cpu().numpy(),
        "loss_touch": loss_touch.detach().cpu().numpy(),
        "loss_pose_prior": loss_pose_prior.detach().cpu().numpy(),
        "loss": loss.detach().cpu().numpy()
    }

    return loss, loss_dict



def optimize(opt):
    ##
    logger = opt.logger
    label = opt.label
    pyrender = opt.pyrender
    neural_render = opt.neural_render


    ## gt
    opt.dataset = dataset(opt)
    opt.init_para = init_para(opt)

    faces = torch.tensor(opt.init_para['faces'], dtype=torch.int32).to(opt.device)
    faces_two_person_batch =torch.tensor(opt.init_para['faces_two_person_batch'],
                                         dtype=torch.int32).to(opt.device)

    ## learning parameters
    pose_iter_0_9 = torch.tensor(opt.init_para['pose_0_9'], dtype=torch.float32).to(opt.device)
    pose_iter_12_21 = torch.tensor(opt.init_para['pose_12_21'], dtype=torch.float32).to(opt.device)
    pose_10_11 = torch.tensor(opt.init_para['pose_10_11'], dtype=torch.float32).to(opt.device)

    shape_iter = torch.tensor(opt.init_para['shape'], dtype=torch.float32).to(opt.device)
    transl_iter = torch.tensor(opt.init_para['transl'], dtype=torch.float32).to(opt.device)
    left_hand_pose_iter = torch.tensor(opt.init_para['left_hand_pose'], dtype=torch.float32).to(opt.device)
    right_hand_pose_iter = torch.tensor(opt.init_para['right_hand_pose'], dtype=torch.float32).to(opt.device)

    jaw_pose = torch.tensor(opt.init_para['jaw_pose'], dtype=torch.float32).to(opt.device)
    leye_pose = torch.tensor(opt.init_para['leye_pose'], dtype=torch.float32).to(opt.device)
    reye_pose = torch.tensor(opt.init_para['reye_pose'], dtype=torch.float32).to(opt.device)
    expression = torch.tensor(opt.init_para['expression'], dtype=torch.float32).to(opt.device)

    pose_iter_0_9.requires_grad = True
    pose_iter_12_21.requires_grad = True
    shape_iter.requires_grad = True
    transl_iter.requires_grad = True
    left_hand_pose_iter.requires_grad = True
    right_hand_pose_iter.requires_grad = True


    optimizer = torch.optim.Adam([pose_iter_0_9, pose_iter_12_21, shape_iter, transl_iter], lr=opt.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)

    tqdm_iter = tqdm(range(opt.total_iter), leave=True)
    tqdm_iter.set_description(opt.exp_name)
    for it_id in tqdm_iter:
        # pose
        global_orient = pose_iter_0_9[:, :, 0].view(opt.num_img, 2, -1)
        body_pose = torch.cat((pose_iter_0_9[:, :, 1:], pose_10_11, pose_iter_12_21), dim=2)
        body_pose = body_pose.view(opt.num_img, 2, -1)

        # forword
        vertices_batch = []
        kp3d_batch = []

        for i in range(2):
            smpl = opt.smpl_male
            if opt.gender_list[i] == 'female':
                smpl = opt.smpl_female
            elif opt.gender_list[i] == 'neural':
                smpl = opt.smpl_neural

            vertices, kp3d, faces = smpl(global_orient=global_orient[:, i],
                                             body_pose=body_pose[:, i],
                                             betas=shape_iter[:, i].squeeze(1),
                                             left_hand_pose=left_hand_pose_iter[:, i],
                                             right_hand_pose=right_hand_pose_iter[:, i],
                                             jaw_pose=jaw_pose[:, i],
                                             leye_pose=leye_pose[:, i],
                                             reye_pose=reye_pose[:, i],
                                             expression=expression[:, i]
                                             )

            vertices = vertices + transl_iter[:, i]
            kp3d = kp3d + transl_iter[:, i]

            vertices_batch.append(vertices)
            kp3d_batch.append(kp3d)

        vertices_batch = torch.stack(vertices_batch, dim=1)
        kp3d_batch = torch.stack(kp3d_batch, dim=1)

        # kp2d pred
        body_kp2d = opt.camera_sequence.perspective(kp3d_batch)

        # mask and part mask pred
        vertices_batch = opt.camera_sequence.world_2_camera(vertices_batch)
        vertices_two_person_batch = torch.cat((vertices_batch[:, 0],
                                               vertices_batch[:, 1]), dim=1)

        mask = None
        if opt.mask_weight != 0:
            mask = neural_render.render_mask(vertices_two_person_batch,
                                             faces_two_person_batch)


        ## loss
        loss, loss_dict = loss_f(opt, mask, body_kp2d, body_pose,
                                 shape_iter, vertices_batch, faces)

        # update grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)


        ## submit
        pre_dict = {
            # "part_mask_pre": mask_pre[1:],
            "body_kp2d": body_kp2d[:, :, label["kp2d_smplx_2_coco"]].detach().cpu().numpy(),
            "vertices": vertices_two_person_batch.detach().cpu().numpy(),
            "faces": faces_two_person_batch.detach().cpu().numpy()
        }

        if opt.mask_weight != 0:
            pre_dict["mask"] = mask.detach().cpu().numpy()

        submit(opt, it_id, loss_dict, pre_dict)


def main():
    # submit
    opt.logger = create_log(opt.exp_name, opt.image_id_range)

    # label
    opt.label = get_label(img_id_range=opt.image_id_range, visualize=False)

    # render and camera
    height, width = opt.label['img'].shape[1:3]
    opt.pyrender = PerspectivePyrender(opt.label['intrinsic'],
                                       opt.label['pyrender_camera_pose'],
                                       width=width, height=height)

    K = torch.tensor(opt.label['intrinsic'][None, :, :], dtype=torch.float32).to(opt.device)
    R = torch.tensor(np.eye(3)[None, :, :], dtype=torch.float32).to(opt.device)
    t = torch.tensor(np.zeros((1, 3))[None, :, :], dtype=torch.float32).to(opt.device)

    opt.neural_render = PerspectiveNeuralRender(K, R, t, height=height, width=width)

    opt.camera = CameraPerspectiveTorch(opt.label['intrinsic'], opt.label['extrinsic'], opt.device)
    opt.camera_sequence = CameraPerspectiveTorchMultiImage(opt.label['intrinsic'], opt.label['extrinsic'], opt.device)

    # smplx
    opt.smpl_male = get_smpl_x(gender='male', device=opt.device)
    opt.smpl_female = get_smpl_x(gender='female', device=opt.device)
    opt.smpl_neutral = get_smpl_x(gender='neutral', device=opt.device)

    # pose prior
    opt.pose_prior = PosePrior().to(opt.device)

    # optimize
    optimize(opt)


if __name__ == '__main__':
    main()
