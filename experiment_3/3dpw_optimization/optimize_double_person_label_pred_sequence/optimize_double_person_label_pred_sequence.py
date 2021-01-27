import os
import sys
from tqdm import tqdm
import torch
import numpy as np
import signal

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")

from common.loss import coco_l2_loss, l2_loss, mask_loss, part_mask_loss
from common.loss import smpl_collision_loss, touch_loss, pose_prior_loss
from common.loss import pose_consistency_loss, shape_consistency_loss, kp3d_consistency_loss, transl_consistency_loss

from preprocess import init_opt, load_check_point
from post_process import init_submit_thread, post_process, save_data, force_exit_thread


def dataset(opt):
    label = opt.label

    kp2d_gt = torch.tensor(label['kp2d'], dtype=torch.float32).to(opt.device)
    kp2d_mask = torch.tensor(label['kp2d_mask'], dtype=torch.float32).to(opt.device)
    # pose_reg = torch.tensor(label['pose'], dtype=torch.float32).to(opt.device)
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
        'kp2d_mask': kp2d_mask,
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

    pose_0 = label['mean_pose'][0][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                      repeat(2, axis=1).astype(dtype=np.float32)
    pose_1_9 = label['mean_pose'][1:10][None, None, ...].repeat(opt.num_img, axis=0).\
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_12_21 = label['mean_pose'][12:][None, None, ...].repeat(opt.num_img, axis=0).\
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_10_11 = label['mean_pose'][10:12][None, None, ...].repeat(opt.num_img, axis=0).\
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    transl = label['trans'].astype(dtype=np.float32)
    shape = label['mean_shape'][None, None, ...].repeat(opt.num_img, axis=0).\
                                                repeat(2, axis=1).astype(dtype=np.float32)
    left_hand_pose = np.zeros((opt.num_img, 2, 6), dtype=np.float32)
    right_hand_pose = np.zeros((opt.num_img, 2, 6), dtype=np.float32)

    jaw_pose = np.zeros((opt.num_img, 2, 3), dtype=np.float32)
    leye_pose = np.zeros((opt.num_img, 2, 3), dtype=np.float32)
    reye_pose = np.zeros((opt.num_img, 2, 3), dtype=np.float32)
    expression = np.zeros((opt.num_img, 2, 10), dtype=np.float32)


    update_para = {
        'pose_0': pose_0,
        'pose_1_9': pose_1_9,
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
    }

    ## resume
    if opt.resume:
        if os.path.exists(opt.check_point):
            update_para = load_check_point(opt.check_point, update_para)
            print('resume ok.')
        else:
            print('resume failed, check point file is not exist.')


    ## to device
    for k, v in update_para.items():
        update_para[k] = torch.tensor(v,
                              dtype=torch.float32).to(opt.device)
        if k in opt.requires_grad_para_dict:
            update_para[k].requires_grad = opt.requires_grad_para_dict[k]


    other_para = {
        'faces': torch.tensor(faces).to(opt.device),
        'faces_two_person_batch':
            torch.tensor(faces_two_person_batch).to(opt.device)
    }

    return update_para, other_para


def loss_f(opt, mask, kp2d, kp3d, global_pose, transl, body_pose, shape, vertices_batch, faces):
    loss_mask, \
    loss_kp2d, \
    loss_pose_reg, \
    loss_shape_reg,\
    loss_collision, \
    loss_touch, \
    loss_global_pose_consistency, \
    loss_transl_consistency, \
    loss_body_pose_consistency, \
    loss_shape_consistency, \
    loss_kp3d_consistency, \
    loss_pose_prior = torch.zeros(12).to(opt.device)


    # mask
    if opt.mask_weight > 0:
        loss_mask = mask_loss(mask, opt.dataset['mask'], opt.mask_weight_list)
    # loss_part_mask = part_mask_loss(mask_pre[1:], opt.dataset['part_mask'])

    # kp2d
    if opt.kp2d_weight > 0:
        kp2d_mask = opt.dataset['kp2d_mask']
        kp2d_pre = kp2d_mask * kp2d[:, :, opt.label["joint_smplx_2_coco"]]
        kp2d_gt = kp2d_mask * opt.dataset['kp2d']
        loss_kp2d = l2_loss(kp2d_pre, kp2d_gt)

    # pose and shape reg
    if opt.pose_reg_weight > 0:
        loss_pose_reg = l2_loss(pose[:, 1:22], opt.dataset['pose_reg'][:, 1:22])
    if opt.shape_reg_weight > 0:
        loss_shape_reg = l2_loss(shape, opt.dataset['shape_reg'])


    # pose, shape, transl, kp3d consistency
    if opt.global_pose_consistency_weight > 0 and opt.num_img > 1:
        loss_global_pose_consistency = pose_consistency_loss(global_pose)
    if opt.body_pose_consistency_weight > 0 and opt.num_img > 1:
        loss_body_pose_consistency = pose_consistency_loss(body_pose)
    if opt.transl_consistency_weight > 0 and opt.num_img > 1:
        loss_transl_consistency = transl_consistency_loss(transl)

    if opt.shape_consistency_weight > 0 and opt.num_img > 1:
        loss_shape_consistency = shape_consistency_loss(shape)

    if opt.kp3d_consistency_weight > 0 and opt.num_img > 1:
        kp3d_pre = kp3d[:, :, opt.label["joint_smplx_2_coco"]]
        loss_kp3d_consistency = kp3d_consistency_loss(kp3d_pre)


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
    loss_global_pose_consistency = loss_global_pose_consistency * opt.global_pose_consistency_weight
    loss_body_pose_consistency = loss_body_pose_consistency * opt.body_pose_consistency_weight
    loss_transl_consistency = loss_transl_consistency * opt.transl_consistency_weight
    loss_shape_consistency = loss_shape_consistency * opt.shape_consistency_weight
    loss_kp3d_consistency = loss_kp3d_consistency * opt.kp3d_consistency_weight


    loss =  loss_mask + \
            loss_kp2d + \
            loss_pose_reg + \
            loss_shape_reg + \
            loss_collision + \
            loss_touch + \
            loss_pose_prior + \
            loss_global_pose_consistency + \
            loss_body_pose_consistency + \
            loss_transl_consistency + \
            loss_shape_consistency + \
            loss_kp3d_consistency


    loss_dict = {
        "loss_mask": loss_mask.detach().cpu().numpy(),
        # "loss_part_mask": loss_part_mask,
        "loss_kp2d": loss_kp2d.detach().cpu().numpy(),
        "loss_pose_reg": loss_pose_reg.detach().cpu().numpy(),
        "loss_shape_reg": loss_shape_reg.detach().cpu().numpy(),
        "loss_collision": loss_collision.detach().cpu().numpy(),
        "loss_touch": loss_touch.detach().cpu().numpy(),
        "loss_pose_prior": loss_pose_prior.detach().cpu().numpy(),
        "loss_global_pose_consistency": loss_global_pose_consistency.detach().cpu().numpy(),
        "loss_body_pose_consistency": loss_body_pose_consistency.detach().cpu().numpy(),
        "loss_transl_consistency": loss_transl_consistency.detach().cpu().numpy(),
        "loss_shape_consistency": loss_shape_consistency.detach().cpu().numpy(),
        "loss_kp3d_consistency": loss_kp3d_consistency.detach().cpu().numpy(),
        "loss": loss.detach().cpu().numpy()
    }

    return loss, loss_dict



def optimize(opt):
    ##
    label = opt.label

    ## gt
    opt.dataset = dataset(opt)
    opt.update_para, other_para = init_para(opt)

    faces = other_para['faces']
    faces_two_person_batch = other_para['faces_two_person_batch']

    # learning parameters
    pose_iter_0 = opt.update_para['pose_0']
    pose_iter_1_9 = opt.update_para['pose_1_9']
    pose_iter_12_21 = opt.update_para['pose_12_21']
    pose_10_11 = opt.update_para['pose_10_11']

    shape_iter = opt.update_para['shape']
    transl_iter = opt.update_para['transl']
    left_hand_pose_iter = opt.update_para['left_hand_pose']
    right_hand_pose_iter = opt.update_para['right_hand_pose']

    jaw_pose = opt.update_para['jaw_pose']
    leye_pose = opt.update_para['leye_pose']
    reye_pose = opt.update_para['reye_pose']
    expression = opt.update_para['expression']

    optimizer = torch.optim.Adam([pose_iter_0,
                                  pose_iter_1_9,
                                  pose_iter_12_21,
                                  shape_iter,
                                  transl_iter], lr=opt.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)

    ## iter
    tqdm_iter = tqdm(range(opt.total_iter), leave=True)
    tqdm_iter.set_description(opt.exp_name)
    for it_id in tqdm_iter:

        # pose
        global_orient = pose_iter_0.view(opt.num_img, 2, -1)
        body_pose = torch.cat((pose_iter_1_9, pose_10_11, pose_iter_12_21), dim=2)
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
        kp2d_batch = opt.camera_sequence.perspective(kp3d_batch)

        # mask and part mask pred
        vertices_batch = opt.camera_sequence.world_2_camera(vertices_batch)
        vertices_two_person_batch = torch.cat((vertices_batch[:, 0],
                                               vertices_batch[:, 1]), dim=1)

        mask = None
        if opt.mask_weight > 0:
            mask = opt.neural_render.render_mask(vertices_two_person_batch,
                                                 faces_two_person_batch)


        ## loss
        loss, loss_dict = loss_f(opt, mask, kp2d_batch, kp3d_batch, global_orient,
                                 transl_iter, body_pose, shape_iter, vertices_batch, faces)




        # update grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)


        # ## submit
        if it_id % opt.submit_other_iter == 0 or \
           it_id % opt.submit_scalar_iter == 0:

            pred_dict = {
                # "part_mask_pre": mask_pre[1:],
                "kp2d": kp2d_batch[:, :, label["joint_smplx_2_coco"]].detach().cpu().numpy(),
                "vertices": vertices_two_person_batch.detach().cpu().numpy(),
                "faces": faces_two_person_batch.detach().cpu().numpy()
            }

            if opt.mask_weight != 0:
                pred_dict["mask"] = mask.detach().cpu().numpy()

            # save parameters
            pred_dict['para'] = {}
            for k, v in opt.update_para.items():
                pred_dict['para'][k] = v.detach().cpu().numpy()

            save_data(opt, it_id, loss_dict, pred_dict)


    ## post process
    post_process(opt)


def main():
    # signal handle
    def handler(sig, argv):
        force_exit_thread()
        sys.exit(0)
        # os.kill(os.getpid(),signal.SIGKILL)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    # signal.signal(signal.SIGQUIT, handler)

    # init opt
    opt = init_opt()

    # save threading
    if opt.use_save_server:
        init_submit_thread(opt)

    # optimize
    optimize(opt)


if __name__ == '__main__':
    main()
