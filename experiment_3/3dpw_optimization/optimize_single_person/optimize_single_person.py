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
from common.loss import l1_loss, l2_loss, mask_loss, part_mask_loss
from common.log import Logger
from common.camera import CameraPerspective, CameraPerspectiveTorch
from common.render import PerspectivePyrender, PerspectiveNeuralRender
from common.smpl_uv import smplx_part_label

from config import opt

def crop_img(opt, label):
    # crop
    index = label['mask'].nonzero()
    x1, x2, y1, y2 = index[1].min(), index[1].max(), index[0].min(), index[0].max()

    t = opt.side_expand
    x1 -= t
    y1 -= t
    x2 += t
    y2 += t

    img_crop = label['img'][y1:y2, x1:x2].copy()
    mask_crop = label['mask'][y1:y2, x1:x2].copy()
    
    part_segmentation_crop_dict = {}
    for part_name, part_segmentation in label['part_segmentation'].items():
        part_segmentation_crop_dict[part_name] = part_segmentation[y1:y2, x1:x2].copy()

    kp2d_crop = label['kp2d'] - np.array([[x1, y1]])

    intrinsic_crop = label['intrinsic'].copy()
    intrinsic_crop[0, 2] -= x1
    intrinsic_crop[1, 2] -= y1


    # resize
    height, width = img_crop.shape[:2]
    new_size = (int(opt.image_scale * width),
                int(opt.image_scale * height))

    label['img_crop'] = cv2.resize(img_crop, new_size, interpolation=cv2.INTER_CUBIC)
    label['mask_crop'] = cv2.resize((mask_crop*255).astype(np.uint8),
                                     new_size, interpolation=cv2.INTER_CUBIC) / 255.0

    part_segmentation_crop_resize_dict = {}

    for part_name, part_segmentation in part_segmentation_crop_dict.items():
        part_segmentation_crop_resize_dict[part_name] = \
            cv2.resize((part_segmentation*255).astype(np.uint8),
                       new_size, interpolation=cv2.INTER_CUBIC) / 255.0

    label['part_segmentation_crop'] = part_segmentation_crop_resize_dict

    intrinsic_crop[:2] = intrinsic_crop[:2] * opt.image_scale
    label['intrinsic_crop'] = intrinsic_crop

    label['kp2d_crop'] = kp2d_crop * opt.image_scale

    return label


def get_part_segmentation_mask(label, gender, img_id):
    part_dir = os.path.join(abspath, "../data_prepare/3DPW/courtyard_dancing_00_mask/part_segmentation/")
    part_name_list = label['smplx_faces']['part_name_list']

    part_segmentation_dict = {}
    for part_name in part_name_list:
        mask = cv2.imread(os.path.join(part_dir, str(gender), part_name,
                                       'image_%s.png' % str(img_id).zfill(5)))
        part_segmentation_dict[part_name] = mask[:,:,0] / 255

    label['part_segmentation'] = part_segmentation_dict

    return label


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
    mask = cv2.imread(os.path.join(abspath, "../data_prepare/3DPW/courtyard_dancing_00_mask/instance/",
                                       str(gender),'image_%s.png' % str(img_id).zfill(5)))

    label['img'] = img
    label['mask'] = mask[:,:,0] / 255

    # load part segmentation
    label['smplx_faces'] = smplx_part_label()
    label = get_part_segmentation_mask(label, gender, img_id)

    # crop img
    label = crop_img(opt, label)
    label['img'] = label['img_crop']
    label['mask'] = label['mask_crop']
    label['intrinsic'] = label['intrinsic_crop']
    label['kp2d'] = label['kp2d_crop']
    label['part_segmentation'] = label['part_segmentation_crop']

    # show
    if show_label:
        I = draw_kp2d(label['img'].copy(), label['kp2d'])
        I = draw_mask(I, label['mask'][:, :, None], color=(0, 255, 0))

        cv2.namedWindow('mask_kp2d', 0)
        cv2.imshow('mask_kp2d', I)

        cv2.namedWindow('segmentation_mask', 0)
        I = label['img'].copy()
        for part_name, seg_mask in label['part_segmentation'].items():
            I = draw_mask(I, seg_mask[:, :, None], color=(0, 255, 0))
        cv2.imshow('segmentation_mask', I)

        cv2.waitKey(0)

    return label


def get_smpl_x(gender='female', device='cpu'):
    smplx_model_path = os.path.join(abspath, '../../../data/')
    return SMPL_X(model_path=smplx_model_path, model_type='smplx', gender=gender).to(device)


def create_log(exp_name):
    log_dir = os.path.join(abspath, 'output', exp_name)
    config_path = os.path.join(abspath, 'config.py')
    return Logger(log_dir, config_path, save_obj=True, save_img=True)


def submit(opt, id, loss_dict, pre_dict):
    logger = opt.logger
    label = opt.label
    render = opt.pyrender

    opt.logger.update_summary_id(id)

    if id % 9 == 0:
        logger.scalar_summary_dict(loss_dict)

    if id % 299 == 0:
        # kp2d
        img_kp2d = draw_kp2d(label['img'].copy(),
                        pre_dict['kp2d_body_pre'].detach().cpu().numpy())
        img_kp2d = draw_kp2d(img_kp2d.copy(),
                        label['kp2d'], color=(0,255,0))

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
        mask = np.zeros((label['mask'].shape[0], label['mask'].shape[1], 3), dtype=np.uint8)
        mask = draw_mask(mask, label['mask'][:, :, None], color=(0, 255, 0))
        mask = draw_mask(mask, pre_dict['mask_pre'][:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        logger.add_image('mask_'+str(id), cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        logger.save_image('mask_%s.png' % str(id).zfill(5), mask)

        # add part mask
        part_full_mask = np.zeros((label['mask'].shape[0], label['mask'].shape[1], 3), dtype=np.uint8)
        for part_name, seg_mask in label['part_segmentation'].items():
            part_full_mask = draw_mask(part_full_mask, seg_mask[:, :, None], color=(0, 255, 0))
        # for part_mask in pre_dict['part_mask_pre']:
        #     part_full_mask = draw_mask(part_full_mask, part_mask[:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        part_full_mask = draw_mask(part_full_mask, pre_dict['mask_pre'][:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        logger.add_image('part_full_mask_'+str(id), cv2.cvtColor(part_full_mask, cv2.COLOR_BGR2RGB))
        logger.save_image('part_full_mask_%s.png' % str(id).zfill(5), part_full_mask)


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
    pose_reg = torch.tensor(label['pose'][1:], dtype=torch.float32).to(opt.device)
    shape_reg = torch.tensor(label['shape'], dtype=torch.float32).to(opt.device)

    # mask
    mask_gt = torch.tensor(label['mask'], dtype=torch.float32).to(opt.device)

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

    return {
        'kp2d': kp2d_gt,
        'pose_reg': pose_reg,
        'shape_reg': shape_reg,
        'mask': mask_gt,
        'part_mask': part_mask_gt,
        'full_and_part_mask': full_and_part_mask_gt,
        'part_faces': part_faces_gt
    }


def optimize(opt):
    ##
    logger = opt.logger
    label = opt.label
    camera = opt.camera
    pyrender = opt.pyrender
    neural_render = opt.neural_render
    smpl = opt.smpl

    ## gt
    opt.dataset = dataset(opt)

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
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)

    tqdm_iter = tqdm(range(opt.total_iter), leave=True)
    for it_id in tqdm_iter:
        # forword
        vertices, kp3d_pre, faces = smpl(global_orient=global_orient,
                                         body_pose=body_pose,
                                         betas=shape_iter)
        vertices = vertices.squeeze(0) + transl_iter
        kp3d_pre = kp3d_pre.squeeze(0) + transl_iter

        # kp2d pred
        kp2d_body_pre = camera.perspective(kp3d_pre[:22])

        # mask and part mask pred
        vertices = camera.world_2_camera(vertices).unsqueeze(0)
        faces = torch.tensor(faces[None, :, :].astype(np.int32),
                             dtype=torch.int32).to(opt.device)

        vertices_batch = vertices.expand(len(opt.dataset['full_and_part_mask']),
                                         vertices.shape[1], vertices.shape[2])
        faces_batch = torch.cat((faces, opt.dataset['part_faces']), dim=0)

        mask_pre = []
        for i in range(len(vertices_batch)):
            mask_pre.append(neural_render.render_mask(vertices_batch[i].unsqueeze(0),
                                                      faces_batch[i].unsqueeze(0)))
        mask_pre = torch.cat(mask_pre, dim=0)

        # loss
        loss_mask = mask_loss(mask_pre[0], opt.dataset['mask'])
        loss_part_mask = part_mask_loss(mask_pre[1:], opt.dataset['part_mask'])
        loss_kp2d = l2_loss(kp2d_body_pre, opt.dataset['kp2d'])
        loss_pose_reg = l1_loss(pose_iter[1:22], opt.dataset['pose_reg'])
        loss_shape_reg = l1_loss(shape_iter, opt.dataset['shape_reg'])
        loss =  loss_mask * opt.mask_weight + \
                loss_part_mask * opt.part_mask_weight + \
                loss_kp2d * opt.kp2d_weight + \
                loss_pose_reg * opt.pose_weight + \
                loss_shape_reg * opt.shape_weight

        # update grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        # submit
        tqdm_iter.set_postfix_str(s='transl_iter=(%.4f,%.4f,%.4f)' % \
                                    (transl_iter[0][0].item(),
                                     transl_iter[0][1].item(),
                                     transl_iter[0][1].item()))

        loss_dict = {
            "loss_mask": loss_mask,
            "loss_part_mask": loss_part_mask,
            "loss_kp2d": loss_kp2d,
            "loss_pose_reg": loss_pose_reg,
            "loss_shape_reg": loss_shape_reg,
            "loss": loss
        }
        pre_dict = {
            "part_mask_pre": mask_pre[1:],
            "mask_pre": mask_pre[0],
            "kp2d_body_pre": kp2d_body_pre,
            "vertices": vertices,
            "faces": faces
        }

        submit(opt, it_id, loss_dict, pre_dict)


def main():
    # submit
    opt.logger = create_log(opt.exp_name)

    # label
    opt.label = get_label(img_id=0, gender=opt.gender, show_label=False)

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
    opt.smpl = get_smpl_x(gender=opt.label['gender'], device=opt.device)

    # optimize
    optimize(opt)


if __name__ == '__main__':
    main()
