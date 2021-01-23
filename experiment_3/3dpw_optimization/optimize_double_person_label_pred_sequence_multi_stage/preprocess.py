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
from common.log import Logger
from common.camera import CameraPerspective, CameraPerspectiveTorch, CameraPerspectiveTorchMultiImage
from common.render import PerspectivePyrender, PerspectiveNeuralRender
from common.smpl_uv import smplx_part_label
from common.pose_prior import PosePrior

from config import opt


def crop_img(opt, label):
    # crop
    index = label['mask'][0].nonzero()
    x1, x2, y1, y2 = index[1].min(), index[1].max(), index[0].min(), index[0].max()

    t = opt.side_expand
    x1 -= t
    y1 -= t
    x2 += t
    y2 += t

    img_crop = label['img'][:, y1:y2, x1:x2].copy()
    mask_crop = label['mask'][:, y1:y2, x1:x2].copy()
    # instance_a_crop = label['instance_a'][y1:y2, x1:x2].copy()
    # instance_b_crop = label['instance_b'][y1:y2, x1:x2].copy()

    if 'part_segmentation' in label:
        part_segmentation_crop_dict = {}
        for part_name, part_segmentation in label['part_segmentation'].items():
            part_segmentation_crop_dict[part_name] = part_segmentation[y1:y2, x1:x2].copy()

    kp2d_crop = label['kp2d'] - np.array([x1, y1]).reshape(1, 1, 1, 2)

    intrinsic_crop = label['intrinsic'].copy()
    intrinsic_crop[0, 2] -= x1
    intrinsic_crop[1, 2] -= y1


    # resize
    height, width = img_crop.shape[1:3]
    new_size = (int(opt.image_scale * width),
                int(opt.image_scale * height))

    label['img_crop'] = np.stack([cv2.resize(im, new_size,
                                  interpolation=cv2.INTER_CUBIC) for im in img_crop], axis=0)
    label['mask_crop'] = np.stack([(cv2.resize((im * 255).astype(np.uint8), new_size,
                                  interpolation=cv2.INTER_CUBIC) > 0).astype(np.float32) for im in mask_crop], axis=0)

    if 'part_segmentation' in label:
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


def get_label(img_id_range=[0,1], kp2d_conf=0.1, visualize=False):
    label = {}
    img_id_start = img_id_range[0]
    img_id_end = img_id_range[1]
    num_img = img_id_end - img_id_start

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

        label["shape"] = shape[img_id_start:img_id_end].reshape(num_img, 2, 1, -1)
        label["pose"] = pose[img_id_start:img_id_end].reshape(num_img, 2, -1, 3)[:, :, :22, :]
        label["kp2d_3dpw"] = gt2d[img_id_start:img_id_end].reshape(num_img, 2, -1, 3)
        label["kp3d_3dpw_smpl"] = gt3d[img_id_start:img_id_end].reshape(num_img, 2, -1, 3)[:, :22]
        label["trans"] = trans[img_id_start:img_id_end].reshape(num_img, 2, -1, 3)
        label["extrinsic"] = pose_world_2_camera[img_id_start:img_id_end]
        label["intrinsic"] = camera_intrinsic
        label['pyrender_camera_pose'] = pyrender_camera_pose


    # kp2d pred
    kp2d_label_path = os.path.join(abspath, '../data_prepare_pred/3dpw/kp2d_pred/3dpw_kp2d_pred_tracked.pkl')
    with open(kp2d_label_path, 'rb') as f:
        kp2d_label = pkl.load(f, encoding='iso-8859-1')

    label['kp2d_mask'] = kp2d_label['kp2d'][img_id_start:img_id_end, :, :, 2:3] > kp2d_conf
    label["kp2d"] = label['kp2d_mask'] * kp2d_label['kp2d'][img_id_start:img_id_end, :, :, :2]
    label["joint_smplx_2_coco"] = [55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


    # load img and mask
    img_dir = os.path.join(abspath, "../data_prepare/3DPW")
    mask_dir = os.path.join(abspath, '../data_prepare_pred/3dpw/mask_pred')

    img_list = []
    mask_list = []
    for img_id in range(img_id_start, img_id_end):
        img = cv2.imread(os.path.join(img_dir, "courtyard_dancing_00",
                                      'image_%s.jpg' % str(img_id).zfill(5)))
        mask = cv2.imread(os.path.join(mask_dir,
                                      'image_%s.jpg' % str(img_id).zfill(5)))
        img_list.append(img)
        mask_list.append(mask[:, :, 0] / 255)

    label['img'] = np.stack(img_list, axis=0)
    label['mask'] = np.stack(mask_list, axis=0)


    # pose shape mean parameters
    smpl_mean_para_file = os.path.join(abspath, '../../../data/neutral_smpl_mean_params.h5')
    with h5py.File(smpl_mean_para_file, 'r') as fp:
        mean_pose = np.array(fp['pose']).reshape(24, 3)[:22]
        mean_shape = np.array(fp['shape']).reshape(1, 10)
    label['mean_pose'] = mean_pose
    label['mean_pose'][0, 0] += np.pi
    label['mean_shape'] = mean_shape


    # load part segmentation
    label['smplx_faces'] = smplx_part_label()
    # label = get_part_segmentation_mask(label, gender, img_id)

    # crop img
    label = crop_img(opt, label)
    label['img'] = label['img_crop']
    label['mask'] = label['mask_crop']
    label['intrinsic'] = label['intrinsic_crop']
    label['kp2d'] = label['kp2d_crop']

    # label['part_segmentation'] = label['part_segmentation_crop']

    # show
    if visualize:
        I = label['img'].copy()
        I = draw_kp2d(I, label['kp2d'][0], radius=3, color=(255, 0, 0))
        I = draw_kp2d(I, label['kp2d'][1], radius=3, color=(0, 0, 255))
        I = draw_mask(I, label['mask'][:, :, None], color=(0, 255, 0))

        cv2.namedWindow('mask_kp2d', 0)
        cv2.imshow('mask_kp2d', I)


        # cv2.namedWindow('segmentation_mask', 0)
        # I = label['img'].copy()
        # for part_name, seg_mask in label['part_segmentation'].items():
        #     I = draw_mask(I, seg_mask[:, :, None], color=(0, 255, 0))
        # cv2.imshow('segmentation_mask', I)

        cv2.waitKey(0)

    return label


def get_smpl_x(gender='female', device='cpu'):
    smplx_model_path = os.path.join(abspath, '../../../data/')
    return SMPL_X(model_path=smplx_model_path, model_type='smplx', gender=gender).to(device)


def create_log(exp_name, image_id_range, submit_step_id_list):
    log_dir = os.path.join(abspath, 'output', exp_name)
    config_path = os.path.join(abspath, 'config.py')
    return Logger(log_dir, config_path,
                  save_obj=True, save_img=True,
                  save_img_sequence=True,
                  save_obj_sequence=True,
                  image_id_range=image_id_range,
                  submit_step_id_list=submit_step_id_list)


def init_opt():
    # submit
    opt.logger = create_log(opt.exp_name,
                            opt.image_id_range,
                            opt.submit_step_id_list)

    # label
    opt.label = get_label(img_id_range=opt.image_id_range,
                          kp2d_conf=opt.kp2d_conf,
                          visualize=False)

    # render and camera
    height, width = opt.label['img'].shape[1:3]
    opt.pyrender = PerspectivePyrender(opt.label['intrinsic'],
                                       opt.label['pyrender_camera_pose'],
                                       width=width, height=height)

    if opt.mask_weight > 0:
        K = torch.tensor(opt.label['intrinsic'][None, :, :], dtype=torch.float32).to(opt.device)
        R = torch.tensor(np.eye(3)[None, :, :], dtype=torch.float32).to(opt.device)
        t = torch.tensor(np.zeros((1, 3))[None, :, :], dtype=torch.float32).to(opt.device)

        opt.neural_render = PerspectiveNeuralRender(K, R, t, height=height, width=width)

    opt.camera = CameraPerspectiveTorch(opt.label['intrinsic'], opt.label['extrinsic'], opt.device)
    opt.camera_sequence = CameraPerspectiveTorchMultiImage(opt.label['intrinsic'],
                                                           opt.label['extrinsic'], opt.device)

    # smplx
    opt.smpl_male = get_smpl_x(gender='male', device=opt.device)
    opt.smpl_female = get_smpl_x(gender='female', device=opt.device)
    opt.smpl_neutral = get_smpl_x(gender='neutral', device=opt.device)

    # pose prior
    opt.pose_prior = PosePrior().to(opt.device)

    return opt

