import os
import sys
from tqdm import tqdm
import torch
import numpy as np
import signal
import cv2


abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")
sys.path.append(abspath + "/../")

from common.loss import coco_l2_loss, l2_loss, mask_loss, part_mask_loss, kp2d_loss
from common.loss import smpl_collision_loss, touch_loss, pose_prior_loss
from common.loss import pose_consistency_loss, shape_consistency_loss
from common.loss import kp3d_consistency_loss, transl_consistency_loss
from common.loss import texture_render_loss, texture_temporal_consistency_loss
from common.loss import texture_part_consistency_loss
from common.loss import ground_loss
from common.debug import draw_kp2d, draw_mask, add_blend_smpl


from preprocess import init_opt, load_check_point
from post_process import init_submit_thread, post_process, force_exit_thread

from common.visualization import imshow_cv2_not_uint8

from data_eval.eval_3DPW import eval_mpjpe_3dpw


def calculate_kp3d_origin(opt, kp3d):
    """
    restore kp3d to before image cropping.
    """
    f = opt.init_camera_para['f']
    y1 = opt.crop_range[0]
    x1 = opt.crop_range[2]

    kp3d_origin = kp3d.copy()
    kp3d_origin[:, :, :, 0] = kp3d_origin[:, :, :, 0] - kp3d_origin[:, :, :, 2] * x1 / f
    kp3d_origin[:, :, :, 1] = kp3d_origin[:, :, :, 1] - kp3d_origin[:, :, :, 2] * y1 / f

    return kp3d_origin


def dataset(opt):
    label = opt.label

    kp2d_gt = torch.tensor(label['kp2d'], dtype=torch.float32).to(opt.device)
    kp2d_mask = torch.tensor(label['kp2d_mask'], dtype=torch.float32).to(opt.device)
    # pose_reg = torch.tensor(label['pose'], dtype=torch.float32).to(opt.device)
    shape_reg = torch.tensor(label['mean_shape'][None, None, ...].\
                             repeat(opt.num_img, axis=0).repeat(2, axis=1),
                             dtype=torch.float32).to(opt.device)

    # mask
    # img_gt = torch.tensor(label['img'] / 255.0, dtype=torch.float32).to(opt.device)
    #
    # mask_gt = None
    # if opt.mask_weight > 0:
    #     mask_gt = torch.tensor(label['mask'], dtype=torch.float32).to(opt.device)
    # instance_a_gt = torch.tensor(label['instance_a'], dtype=torch.float32).to(opt.device)[None, :, :]
    # instance_b_gt = torch.tensor(label['instance_b'], dtype=torch.float32).to(opt.device)[None, :, :]

    # if 'part_segmentation' in label:
    #     part_mask_gt = np.zeros((len(label['smplx_faces']['part_name_list']),
    #                              label['mask'].shape[0],
    #                              label['mask'].shape[1]), dtype=np.float32)
    #     for i, part_name in enumerate(label['smplx_faces']['part_name_list']):
    #         part_mask_gt[i] = label['part_segmentation'][part_name]
    #
    #     part_mask_gt = torch.tensor(part_mask_gt, dtype=torch.float32).to(opt.device)
    #     full_and_part_mask_gt = torch.cat((mask_gt.unsqueeze(0), part_mask_gt), dim=0)

    # part faces
    part_faces_gt = np.zeros((len(label['smplx_misc']['part_name_list']),
                              label['smplx_misc']['faces'].shape[0],
                              label['smplx_misc']['faces'].shape[1]), dtype=np.int32)
    for i, part_name in enumerate(label['smplx_misc']['part_name_list']):
        part_faces = label['smplx_misc']['smplx_part'][part_name]
        part_faces_gt[i][:part_faces.shape[0]] = part_faces
        part_faces_gt[i][part_faces.shape[0]:] = part_faces[0]
    part_faces_gt = torch.tensor(part_faces_gt, dtype=torch.int32).to(opt.device)

    # torch pair
    # touch_pair_list = []
    # touch_pair_list.append({
    #     0: torch.tensor(label['smplx_misc']['smplx_part']['body_middle_back_left'].astype(np.int64), dtype=torch.int64),
    #     1: torch.tensor(label['smplx_misc']['smplx_part']['right_hand'].astype(np.int64), dtype=torch.int64)
    # })
    # touch_pair_list.append({
    #     0: torch.tensor(label['smplx_misc']['smplx_part']['right_hand'].astype(np.int64), dtype=torch.int64),
    #     1: torch.tensor(label['smplx_misc']['smplx_part']['left_hand'].astype(np.int64), dtype=torch.int64)
    # })
    # touch_pair_list.append({
    #     0: torch.tensor(label['smplx_misc']['smplx_part']['left_hand'].astype(np.int64), dtype=torch.int64),
    #     1: torch.tensor(label['smplx_misc']['smplx_part']['arm_right_big'].astype(np.int64), dtype=torch.int64)
    # })

    # texture
    part_vertices = {}
    for i, part_name in enumerate(label['smplx_misc']['part_name_list']):
        part_faces = label['smplx_misc']['smplx_part'][part_name]
        part_vertices[part_name] = np.unique(part_faces.flatten()).astype(np.int64)
        part_vertices[part_name] = torch.tensor(part_vertices[part_name],
                                                dtype=torch.int64).to(opt.device)

    part_vertices['arm_right'] = torch.cat((part_vertices["arm_right_big"],
                                            part_vertices["arm_right_little"]), dim=0)


    return {
        'kp2d': kp2d_gt,
        'kp2d_mask': kp2d_mask,
        # 'pose_reg': pose_reg,
        'shape_reg': shape_reg,
        # 'img': img_gt,
        # 'mask': mask_gt,
        # 'instance_a': instance_a_gt,
        # 'instance_b': instance_b_gt,
        # 'part_mask': part_mask_gt,
        # 'full_and_part_mask': full_and_part_mask_gt,
        'part_faces': part_faces_gt,
        'part_vertices': part_vertices,
        # 'touch_pair_list': touch_pair_list
    }


def init_para(opt):
    label = opt.label

    faces = label['smplx_misc']['faces'].astype(dtype=np.int32)
    vertices = label['smplx_misc']['vertices'].astype(dtype=np.int32)
    faces_two_person = np.concatenate((faces, faces+len(vertices)), axis=0)[None, ...]
    # faces_two_person_batch = np.concatenate((faces, faces+len(vertices)), axis=0)[None, ...].\
    #                                         repeat(opt.num_img, axis=0).astype(dtype=np.int32)


    textures = torch.zeros((1, faces.shape[0] * 2, opt.texture_size,
                            opt.texture_size, opt.texture_size, 3),
                            dtype=torch.float32).to(opt.device)


    pose_0 = label['mean_pose'][0][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                       repeat(2, axis=1).astype(dtype=np.float32)
    pose_1 = label['mean_pose'][1][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_2 = label['mean_pose'][2][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_3 = label['mean_pose'][3][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_4 = label['mean_pose'][4][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_5 = label['mean_pose'][5][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_6 = label['mean_pose'][6][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_7 = label['mean_pose'][7][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_8 = label['mean_pose'][8][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_9 = label['mean_pose'][9][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)

    pose_10_11 = label['mean_pose'][10:12][None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_12 = label['mean_pose'][12][None, None, None, ...].repeat(opt.num_img, axis=0).\
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_13 = label['mean_pose'][13][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_14 = label['mean_pose'][14][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_15 = label['mean_pose'][15][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_16 = label['mean_pose'][16][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_17 = label['mean_pose'][17][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_18 = label['mean_pose'][18][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_19 = label['mean_pose'][19][None, None, None, ...].repeat(opt.num_img, axis=0). \
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_20 = label['mean_pose'][20][None, None, None, ...].repeat(opt.num_img, axis=0).\
                                                        repeat(2, axis=1).astype(dtype=np.float32)
    pose_21 = label['mean_pose'][21][None, None, None, ...].repeat(opt.num_img, axis=0). \
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

    # camera
    camera_f = np.array([label['intrinsic'][0, 0].astype(np.float32)])
    camera_cx = np.array([label['intrinsic'][0, 2].astype(np.float32)])
    camera_cy = np.array([label['intrinsic'][1, 2].astype(np.float32)])

    update_para = {
        'camera_f': camera_f,
        'camera_cx': camera_cx,
        'camera_cy': camera_cy,
        'textures': textures,
        'pose_0': pose_0,
        'pose_1': pose_1,
        'pose_2': pose_2,
        'pose_3': pose_3,
        'pose_4': pose_4,
        'pose_5': pose_5,
        'pose_6': pose_6,
        'pose_7': pose_7,
        'pose_8': pose_8,
        'pose_9': pose_9,
        'pose_10_11': pose_10_11,
        'pose_12': pose_12,
        'pose_13': pose_13,
        'pose_14': pose_14,
        'pose_15': pose_15,
        'pose_16': pose_16,
        'pose_17': pose_17,
        'pose_18': pose_18,
        'pose_19': pose_19,
        'pose_20': pose_20,
        'pose_21': pose_21,
        'shape': shape,
        'transl': transl,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'expression': expression,
    }

    opt.init_update_data = update_para.copy()

    ## resume
    if opt.resume:
        if os.path.exists(opt.check_point):
            update_para = load_check_point(opt, update_para)
            print('resume ok.')
        else:
            print('resume failed, check point file is not exist.')


    ## to device
    for k, v in update_para.items():
        update_para[k] = torch.tensor(v,
                              dtype=torch.float32).to(opt.device)
        # if k in opt.requires_grad_para_dict:
        #     update_para[k].requires_grad = opt.requires_grad_para_dict[k]


    ## other para
    faces_two_person = torch.tensor(faces_two_person).to(opt.device)
    other_para = {
        'faces': torch.tensor(faces).to(opt.device),
        'faces_two_person_batch':
            faces_two_person.expand(opt.num_img,
                                    faces_two_person.shape[1],
                                    faces_two_person.shape[2])
        # 'faces_two_person_batch': torch.tensor(faces_two_person_batch).to(opt.device)
    }

    return update_para, other_para


def loss_f(opt, loss_weight, mask, kp2d, kp3d, global_pose,
           transl, body_pose, shape, vertices_batch, faces,
           img, depth, textures):
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
    loss_pose_prior, \
    loss_texture_render, \
    loss_texture_temporal_consistency, \
    loss_texture_part_consistency, \
    loss_ground = torch.zeros(16).to(opt.device)

    # texture
    if loss_weight.texture_render_weight > 0:
        loss_texture_render = \
            texture_render_loss(
                img,
                opt.dataset['img'][opt.render_sequence.get()],
                depth < 100)

    if loss_weight.texture_temporal_consistency_weight > 0:
        loss_texture_temporal_consistency = \
            texture_temporal_consistency_loss(textures)

    if loss_weight.texture_part_consistency_weight > 0:
        loss_texture_part_consistency = \
            texture_part_consistency_loss(textures, opt.dataset['part_vertices'])

    # ground loss
    if loss_weight.ground_weight > 0:
        loss_ground = ground_loss(kp3d, opt.ground_normal)

    # mask
    if loss_weight.mask_weight > 0:
        loss_mask = mask_loss(mask, opt.dataset['mask'], opt.mask_weight_list)
    # loss_part_mask = part_mask_loss(mask_pre[1:], opt.dataset['part_mask'])

    # kp2d
    if loss_weight.kp2d_weight > 0:
        kp2d_mask = opt.dataset['kp2d_mask']
        kp2d_pre = kp2d_mask * kp2d[:, :, opt.label["joint_smplx_2_coco"]]
        kp2d_gt = kp2d_mask * opt.dataset['kp2d']
        loss_kp2d = kp2d_loss(kp2d_pre, kp2d_gt, loss_weight.kp2d_weight_list)

    # pose and shape reg
    if loss_weight.pose_reg_weight > 0:
        loss_pose_reg = l2_loss(pose[:, 1:22], opt.dataset['pose_reg'][:, 1:22])
    if loss_weight.shape_reg_weight > 0:
        loss_shape_reg = l2_loss(shape, opt.dataset['shape_reg'])


    # pose, shape, transl, kp3d consistency
    if loss_weight.global_pose_consistency_weight > 0 and opt.num_img > 1:
        loss_global_pose_consistency = pose_consistency_loss(global_pose)
    if loss_weight.body_pose_consistency_weight > 0 and opt.num_img > 1:
        loss_body_pose_consistency = pose_consistency_loss(body_pose)
    if loss_weight.transl_consistency_weight > 0 and opt.num_img > 1:
        loss_transl_consistency = transl_consistency_loss(transl)

    if loss_weight.shape_consistency_weight > 0 and opt.num_img > 1:
        loss_shape_consistency = shape_consistency_loss(shape)

    if loss_weight.kp3d_consistency_weight > 0 and opt.num_img > 1:
        kp3d_pre = kp3d[:, :, opt.label["joint_smplx_2_coco"]]
        loss_kp3d_consistency = kp3d_consistency_loss(kp3d_pre)


    # loss_collision
    if loss_weight.collision_weight > 0:
        loss_collision = smpl_collision_loss(vertices_batch, faces,
                                             loss_weight.collision_batch_size)

    # loss_touch
    if loss_weight.touch_weight > 0:
        loss_touch = touch_loss(opt, vertices_batch)

    # loss_pose_prior
    if loss_weight.pose_prior_weight > 0:
        loss_pose_prior = pose_prior_loss(opt, body_pose.view(-1, 21, 3))


    loss_mask *= loss_weight.mask_weight
    loss_kp2d *= loss_weight.kp2d_weight
    loss_pose_reg *= loss_weight.pose_reg_weight
    loss_shape_reg *= loss_weight.shape_reg_weight
    loss_collision *= loss_weight.collision_weight
    loss_touch *= loss_weight.touch_weight
    loss_pose_prior *= loss_weight.pose_prior_weight
    loss_global_pose_consistency *= loss_weight.global_pose_consistency_weight
    loss_body_pose_consistency *= loss_weight.body_pose_consistency_weight
    loss_transl_consistency *= loss_weight.transl_consistency_weight
    loss_shape_consistency *= loss_weight.shape_consistency_weight
    loss_kp3d_consistency *= loss_weight.kp3d_consistency_weight
    loss_texture_render *= loss_weight.texture_render_weight
    loss_texture_temporal_consistency *= loss_weight.texture_temporal_consistency_weight
    loss_texture_part_consistency *= loss_weight.texture_part_consistency_weight
    loss_ground *= loss_weight.ground_weight


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
            loss_kp3d_consistency + \
            loss_texture_render + \
            loss_texture_temporal_consistency + \
            loss_texture_part_consistency + \
            loss_ground


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
        "loss_texture_render": loss_texture_render.detach().cpu().numpy(),
        "loss_texture_temporal_consistency": loss_texture_temporal_consistency.detach().cpu().numpy(),
        "loss_texture_part_consistency": loss_texture_part_consistency.detach().cpu().numpy(),
        "loss_gound": loss_ground.detach().cpu().numpy(),
        "loss": loss.detach().cpu().numpy()
    }

    return loss, loss_dict


def eval_mpjpe(loss_dict, kp3d, vertices, image_id_range,
               gt_dir, save_file='check_point.json', save=False):
    data = eval_mpjpe_3dpw(kp3d, vertices, image_id_range, gt_dir,
                           save_file=save_file, save=save)

    loss_dict['mave'] = data['mave']
    loss_dict['mpjpe'] = data['mpjpe']
    loss_dict['mspe'] = data['mspe']

    return loss_dict



def save_data(opt, save_data_dict):

    pred_dict = save_data_dict["pred_dict"]
    iter_cycle = save_data_dict["iter_cycle"]
    stage_name = save_data_dict["stage_name"]
    it_id = save_data_dict["it_id"]

    ## 1. save image
    opt.pyrender.update_intrinsic(pred_dict['intrinsic'])
    for img_id in np.arange(opt.image_id_range[0],
                            opt.image_id_range[1],
                            opt.save_image_interval):

        # img_id = 10
        img = cv2.imread(os.path.join(opt.img_dir, 'image_%s.jpg'
                                      % str(img_id).zfill(5)))

        # resize
        img_resize = cv2.resize(img, opt.img_size, interpolation=cv2.INTER_CUBIC)

        # render
        img_render, img_depth = opt.pyrender.render_obj(
                                pred_dict['vertices'][img_id-opt.image_id_range[0]],
                                pred_dict['faces'][img_id-opt.image_id_range[0]],
                                show_viewer=False)
        # img_render = cv2.resize(img_render, new_size, interpolation=cv2.INTER_CUBIC)
        # img_depth = cv2.resize(img_depth, new_size, interpolation=cv2.INTER_CUBIC)

        img_add_smpl = add_blend_smpl(img_render, img_depth > 0, img_resize)

        save_dir = os.path.join(opt.logger.save_img_dir,
                                '{}_cycle_{}'.format(stage_name, iter_cycle))
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, 'img_id_{}_iter_{}.jpg'.format(img_id, it_id))
        cv2.imwrite(save_path, img_add_smpl)


    ## save check point
    save_file = os.path.join(opt.logger.save_check_point_dir,
                             'checkpoint_{}_{}_{}.json'.\
                             format(iter_cycle, stage_name, it_id))
    loss_dict = eval_mpjpe({},
                           pred_dict['kp3d'],
                           pred_dict['vertices'].reshape(
                               pred_dict['vertices'].shape[0], 2, -1,
                               pred_dict['vertices'].shape[2]),
                           opt.image_id_range,
                           opt.eval_gt_file_dir,
                           save_file=save_file,
                           save=True)


def forward(opt,
            global_orient,
            body_pose,
            shape_iter,
            left_hand_pose_iter,
            right_hand_pose_iter,
            jaw_pose,
            leye_pose,
            reye_pose,
            expression,
            transl_iter,
            intrinsic):

    # forword
    vertices_batch = []
    kp3d_batch = []

    for i in range(2):
        smpl = opt.smpl_male
        if opt.model_gender[i] == 'female':
            smpl = opt.smpl_female
        elif opt.model_gender[i] == 'neural':
            smpl = opt.smpl_neural

        vertices, kp3d, _ = smpl(global_orient=global_orient[:, i],  # 1x3
                                 body_pose=body_pose[:, i],  # 21 x 3
                                 betas=shape_iter[:, i].squeeze(1),
                                 left_hand_pose=left_hand_pose_iter[:, i], # 1x6
                                 right_hand_pose=right_hand_pose_iter[:, i], # 1x6
                                 jaw_pose=jaw_pose[:, i], # 1 x 3
                                 leye_pose=leye_pose[:, i], # 1 x 3
                                 reye_pose=reye_pose[:, i], # 1 x 3
                                 expression=expression[:, i] # 1 x 10
                                 )

        vertices = vertices + transl_iter[:, i]
        kp3d = kp3d + transl_iter[:, i]

        vertices_batch.append(vertices)
        kp3d_batch.append(kp3d)

    vertices_batch = torch.stack(vertices_batch, dim=1)
    kp3d_batch = torch.stack(kp3d_batch, dim=1)

    # kp2d pred
    kp2d_batch = opt.camera_sequence.perspective(kp3d_batch,
                                                 intrinsic=intrinsic)

    # mask and part mask pred
    vertices_batch = opt.camera_sequence.world_2_camera(vertices_batch)
    vertices_two_person_batch = torch.cat((vertices_batch[:, 0],
                                           vertices_batch[:, 1]), dim=1)

    return kp3d_batch, kp2d_batch, \
           vertices_batch, vertices_two_person_batch



def reinit_update_para(conf, opt):
    if conf.init_para_dict['mean_pose']:
        opt.update_para['pose_1'] = \
            torch.tensor(opt.init_update_data['pose_1'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_2'] = \
            torch.tensor(opt.init_update_data['pose_2'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_3'] = \
            torch.tensor(opt.init_update_data['pose_3'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_4'] = \
            torch.tensor(opt.init_update_data['pose_4'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_5'] = \
            torch.tensor(opt.init_update_data['pose_5'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_6'] = \
            torch.tensor(opt.init_update_data['pose_6'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_7'] = \
            torch.tensor(opt.init_update_data['pose_7'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_8'] = \
            torch.tensor(opt.init_update_data['pose_8'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_9'] = \
            torch.tensor(opt.init_update_data['pose_9'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_10_11'] = \
            torch.tensor(opt.init_update_data['pose_10_11'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_12'] = \
            torch.tensor(opt.init_update_data['pose_12'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_13'] = \
            torch.tensor(opt.init_update_data['pose_13'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_14'] = \
            torch.tensor(opt.init_update_data['pose_14'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_15'] = \
            torch.tensor(opt.init_update_data['pose_15'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_16'] = \
            torch.tensor(opt.init_update_data['pose_16'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_17'] = \
            torch.tensor(opt.init_update_data['pose_17'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_18'] = \
            torch.tensor(opt.init_update_data['pose_18'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_19'] = \
            torch.tensor(opt.init_update_data['pose_19'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_20'] = \
            torch.tensor(opt.init_update_data['pose_20'],
                         dtype=torch.float32).to(opt.device)
        opt.update_para['pose_21'] = \
            torch.tensor(opt.init_update_data['pose_21'],
                         dtype=torch.float32).to(opt.device)

    if conf.init_para_dict['mean_shape']:
        opt.update_para['shape'] = \
            torch.tensor(opt.init_update_data['shape'],
                         dtype=torch.float32).to(opt.device)

    for k, v in opt.update_para.items():
        if k in conf.requires_grad_para_dict:
            opt.update_para[k].requires_grad = \
                conf.requires_grad_para_dict[k]



def optimize_stage(opt, stage_name, iter_cycle):

    if stage_name == 'LG':
        step_id_root = iter_cycle * 100000 + 10000
    elif stage_name == 'LP':
        step_id_root = iter_cycle * 100000 + 20000
    elif stage_name == 'LC':
        step_id_root = iter_cycle * 100000 + 30000


    conf = opt.conf[stage_name]
    other_para = opt.other_para

    faces = other_para['faces']
    faces_two_person_batch = other_para['faces_two_person_batch']

    # require grad, init parameters
    reinit_update_para(conf, opt)
    update_para = opt.update_para

    # learning parameters
    camera_f = update_para['camera_f']
    camera_cx = update_para['camera_cx']
    camera_cy = update_para['camera_cy']

    textures = update_para['textures']

    pose_iter_0 = update_para['pose_0']
    pose_iter_1 = update_para['pose_1']
    pose_iter_2 = update_para['pose_2']
    pose_iter_3 = update_para['pose_3']
    pose_iter_4 = update_para['pose_4']
    pose_iter_5 = update_para['pose_5']
    pose_iter_6 = update_para['pose_6']
    pose_iter_7 = update_para['pose_7']
    pose_iter_8 = update_para['pose_8']
    pose_iter_9 = update_para['pose_9']
    pose_10_11 = update_para['pose_10_11']
    pose_iter_12 = update_para['pose_12']
    pose_iter_13 = update_para['pose_13']
    pose_iter_14 = update_para['pose_14']
    pose_iter_15 = update_para['pose_15']
    pose_iter_16 = update_para['pose_16']
    pose_iter_17 = update_para['pose_17']
    pose_iter_18 = update_para['pose_18']
    pose_iter_19 = update_para['pose_19']
    pose_iter_20 = update_para['pose_20']
    pose_iter_21 = update_para['pose_21']

    shape_iter = update_para['shape']
    transl_iter = update_para['transl']
    left_hand_pose_iter = update_para['left_hand_pose']
    right_hand_pose_iter = update_para['right_hand_pose']

    jaw_pose = update_para['jaw_pose']
    leye_pose = update_para['leye_pose']
    reye_pose = update_para['reye_pose']
    expression = update_para['expression']

    # optimizer
    optimizer = torch.optim.Adam([  camera_f,
                                    textures,
                                    pose_iter_0,
                                    pose_iter_1,
                                    pose_iter_2,
                                    pose_iter_3,
                                    pose_iter_4,
                                    pose_iter_5,
                                    pose_iter_6,
                                    pose_iter_7,
                                    pose_iter_8,
                                    pose_iter_9,
                                    pose_iter_12,
                                    pose_iter_13,
                                    pose_iter_14,
                                    pose_iter_15,
                                    pose_iter_16,
                                    pose_iter_17,
                                    pose_iter_18,
                                    pose_iter_19,
                                    pose_iter_20,
                                    pose_iter_21,
                                    shape_iter,
                                    transl_iter,
                                    right_hand_pose_iter,
                                    left_hand_pose_iter], lr=conf.lr)


    ## optimize
    tqdm_iter = tqdm(range(opt.total_iter), leave=True)
    tqdm_iter.set_description('{}_{}_{}'.\
                              format(iter_cycle, conf.stage_name, opt.exp_name))

    for it_id in tqdm_iter:

        global_orient = pose_iter_0.view(opt.num_img, 2, -1)
        body_pose = torch.cat((pose_iter_1,
                               pose_iter_2,
                               pose_iter_3,
                               pose_iter_4,
                               pose_iter_5,
                               pose_iter_6,
                               pose_iter_7,
                               pose_iter_8,
                               pose_iter_9,
                               pose_10_11,
                               pose_iter_12,
                               pose_iter_13,
                               pose_iter_14,
                               pose_iter_15,
                               pose_iter_16,
                               pose_iter_17,
                               pose_iter_18,
                               pose_iter_19,
                               pose_iter_20,
                               pose_iter_21,), dim=2)
        body_pose = body_pose.view(opt.num_img, 2, -1)

        intrinsic = torch.eye(3).to(opt.device)
        intrinsic[0][0] = camera_f[0]
        intrinsic[1][1] = camera_f[0]
        intrinsic[0][2] = camera_cx[0]
        intrinsic[1][2] = camera_cy[0]


        # forword
        kp3d_batch, kp2d_batch, \
        vertices_batch, vertices_two_person_batch = \
            forward(opt, global_orient, body_pose, shape_iter,
                    left_hand_pose_iter, right_hand_pose_iter,
                    jaw_pose, leye_pose, reye_pose,
                    expression, transl_iter, intrinsic)


        ## loss
        loss, loss_dict = loss_f(opt, conf.loss_weight, None, kp2d_batch, kp3d_batch, global_orient,
                                 transl_iter, body_pose, shape_iter, vertices_batch,
                                 faces, None, None, textures)


        ## stop optimize
        stop_conf = conf.stop_conf
        if it_id == 0:
            stop_conf.last_error = 10000
            stop_conf.loss_stop_down_count_now = 0
            stop_conf.loss_stop_up_count_now = 0
            loss_dict['loss_error'] = 0
        else:
            loss_error = stop_conf.last_error - loss

            if loss_error >= 0:
                if loss_error < stop_conf.stop_error:
                    stop_conf.loss_stop_down_count_now += 1
                    stop_conf.loss_stop_up_count_now = 0

                    if stop_conf.loss_stop_down_count_now >= \
                            stop_conf.loss_total_down_count:

                        print('===> stop down complete.')
                        break
                else:
                    stop_conf.loss_stop_up_count_now = 0
                    stop_conf.loss_stop_down_count_now = 0
            else:
                stop_conf.loss_stop_up_count_now += 1
                stop_conf.loss_stop_down_count_now = 0

                if stop_conf.loss_stop_up_count_now >=\
                        stop_conf.loss_total_up_count:
                    stop_conf.loss_stop_down_count_now = 0
                    stop_conf.loss_stop_up_count_now = 0
                    print('===> stop up complete.')
                    break

            stop_conf.last_error = loss
            loss_dict['loss_error'] = loss_error


        # update grad
        if loss.requires_grad == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print('warning: loss.requires_grad == False')


        # submit
        if it_id % opt.submit_scalar_iter == 0:
            # eval mpjpe
            eval = True
            if eval:
                pred_dict = {
                    "vertices": vertices_two_person_batch.detach().cpu().numpy(),
                    "faces": faces_two_person_batch.detach().cpu().numpy(),
                    'kp3d': opt.camera_sequence.world_2_camera(kp3d_batch).detach().cpu().numpy(),
                    'intrinsic': intrinsic.detach().cpu().numpy()
                }
                 loss_dict = eval_mpjpe(loss_dict,
                                       pred_dict['kp3d'],
                                       pred_dict['vertices'].reshape(
                                           vertices_two_person_batch.shape[0], 2, -1,
                                           vertices_two_person_batch.shape[2]),
                                       opt.image_id_range,
                                       opt.eval_gt_file_dir)

            opt.logger.update_summary_id(step_id_root + it_id)
            opt.logger.scalar_summary_dict(loss_dict)

    # save img
    pred_dict = {
        "vertices": vertices_two_person_batch.detach().cpu().numpy(),
        "faces": faces_two_person_batch.detach().cpu().numpy(),
        'kp3d': opt.camera_sequence.world_2_camera(kp3d_batch).detach().cpu().numpy(),
        'intrinsic': intrinsic.detach().cpu().numpy()
    }

    save_data_dict = {
        "pred_dict": pred_dict,
        "iter_cycle": iter_cycle,
        "stage_name": stage_name,
        "it_id": it_id
    }

    save_data(opt, save_data_dict)

    # return
    return update_para, it_id, save_data_dict


def optimize(opt):

    opt.dataset = dataset(opt)
    opt.update_para, opt.other_para = init_para(opt)

    last_LG_iter = -1
    last_LP_iter = -1

    break_iter_diff_count = 0

    iter_cycle = 1

    ## LG, update -> pose_0, shape, intrinsic, extrinsic
    if opt.use_once_LG:
        opt.update_para, iter_LG, _ = \
            optimize_stage(opt, 'LG', iter_cycle)

    while True:
        ## LG, update -> pose_0, shape, intrinsic, extrinsic
        if opt.use_LG:
            opt.update_para, iter_LG, _ = \
                optimize_stage(opt, 'LG', iter_cycle)

        ## LP, update -> {pose_0, ..., pose_21}, shape, intrinsic, extrinsic
        opt.update_para, iter_LP, save_data_dict = \
            optimize_stage(opt, 'LP', iter_cycle)


        ## submit
        if iter_cycle > 1:
            ## submit
            if opt.use_LG:
                iter_LG_diff = last_LG_iter - iter_LG
            iter_LP_diff = last_LP_iter - iter_LP

            if opt.use_LG:
                iter_dict = {
                    'iter_LG_diff': iter_LG_diff,
                    'iter_LP_diff': iter_LP_diff,
                }
            else:
                iter_dict = {
                    'iter_LP_diff': iter_LP_diff,
                }

            opt.logger.update_summary_id(iter_cycle)
            opt.logger.scalar_summary_dict(iter_dict)

            ## break counter
            if abs(iter_LP_diff) < opt.break_diff_iter_thresh:
                break_iter_diff_count += 1
                if break_iter_diff_count >= opt.break_diff_iter:
                    break
            else:
                break_iter_diff_count=0

        if opt.use_LG:
            last_LG_iter = iter_LG
        last_LP_iter = iter_LP

        iter_cycle += 1


    save_data(opt, save_data_dict)

    # LC, update -> {pose_0, ..., pose_21} , shape, intrinsic, extrinsic
    if opt.use_LC:
        opt.update_para, iter_LP, save_data_dict = \
            optimize_stage(opt, 'LC', iter_cycle)

        save_data(opt, save_data_dict)


def main(conf):
    # signal handle
    def handler(sig, argv):
        force_exit_thread()
        sys.exit(0)
        # os.kill(os.getpid(),signal.SIGKILL)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    # signal.signal(signal.SIGQUIT, handler)

    # init opt
    opt = init_opt(conf)


    # optimize
    optimize(opt)


if __name__ == '__main__':

    sequence_name = [
        'downtown_warmWelcome_00',
        'courtyard_dancing_00',
        'courtyard_dancing_01',
        'courtyard_basketball_00',
        'courtyard_captureSelfies_00',
        'courtyard_giveDirections_00',
        'courtyard_shakeHands_00',
        'courtyard_warmWelcome_00',

        # 'courtyard_hug_00',
        # 'courtyard_capoeira_00'
    ]


    for seq_name in sequence_name: vb

        conf = {
            'sequence_name': seq_name,
            'LG': {
                # 'stop_error': 0.035
                'stop_error': 0.025
                # 'stop_error': 0.020
                # 'stop_error': 0.015
            },
            'LP': {
                # 'stop_error': 0.00085
                # 'stop_error': 0.00095
                # 'stop_error': 0.00105
                # 'stop_error': 0.00125
                # 'stop_error': 0.00165
                # 'stop_error': 0.00195
                # 'stop_error': 0.00225
                # 'stop_error': 0.00255
                # 'stop_error': 0.00285
                # 'stop_error': 0.00315
                # 'stop_error': 0.00345
                'stop_error': 0.00375
                # 'stop_error': 0.00395
                # 'stop_error': 0.00415
            },
            'LC': {
                # 'stop_error': 0.004
                # 'stop_error': 0.003
                'stop_error': 0.002
                # 'stop_error': 0.0015
            },

            'break_diff_iter': 10,
            'break_diff_iter_thresh': 10
        }

        main(conf)
