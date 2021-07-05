import os
import sys
import pickle as pkl
import cv2
import h5py
import torch
import numpy as np
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")

from common.debug import draw_kp2d, draw_mask, add_blend_smpl
from common.smpl_x import SMPL_X
from common.camera import CameraPerspective, CameraPerspectiveTorch, CameraPerspectiveTorchMultiImage
from common.render import PerspectivePyrender, PerspectiveNeuralRender
from common.smpl_uv import smplx_part_label
from common.pose_prior import PosePrior

from config import option
from logger import Logger


def crop_img_with_true_camera_para(opt, label):
    # crop
    index = label['mask'][0].nonzero()
    x1, x2, y1, y2 = index[1].min(), index[1].max(), index[0].min(), index[0].max()

    t = opt.side_expand
    x1 -= t
    y1 -= t
    x2 += t
    y2 += t

    img_crop = label['img'][:, y1:y2, x1:x2]
    mask_crop = label['mask'][:, y1:y2, x1:x2]
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


def crop_img_with_fake_camera_para(opt, label):
    # crop
    y1, y2, x1, x2 = opt.crop_range
    if y2 == -1:
        y2 = opt.img_size[1]
    if x2 == -1:
        x2 = opt.img_size[0]

    opt.img_size = (int(opt.img_size[0] * opt.image_scale),
                    int(opt.img_size[1] * opt.image_scale))


    kp2d_crop = label['kp2d'] - np.array([x1, y1]).reshape(1, 1, 1, 2)

    # intrinsic_crop = label['intrinsic'].copy()
    # intrinsic_crop[0, 2] -= x1
    # intrinsic_crop[1, 2] -= y1

    intrinsic_crop = np.eye(3)
    intrinsic_crop[0][0] = opt.init_camera_para['f']
    intrinsic_crop[1][1] = opt.init_camera_para['f']
    intrinsic_crop[0][2] = (x2 - x1) / 2.0
    intrinsic_crop[1][2] = (y2 - y1) / 2.0


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


def get_label(opt, kp2d_conf=0.1, visualize=False):
    label = {}

    if opt.use_ground_truth:
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
        del img_list
        label['mask'] = np.stack(mask_list, axis=0)
        del mask_list


        # pose shape mean parameters
        smpl_mean_para_file = os.path.join(abspath, '../../../data/neutral_smpl_mean_params.h5')
        with h5py.File(smpl_mean_para_file, 'r') as fp:
            mean_pose = np.array(fp['pose']).reshape(24, 3)[:22]
            mean_shape = np.array(fp['shape']).reshape(1, 10)
        label['mean_pose'] = mean_pose
        # label['mean_pose'][0, 0] += np.pi
        label['mean_shape'] = mean_shape


        # load part segmentation
        label['smplx_misc'] = smplx_part_label()
        # label = get_part_segmentation_mask(label, gender, img_id)

        # crop img
        if opt.use_ture_camera_para:
            label = crop_img_with_true_camera_para(opt, label)
        else:
            label = crop_img_with_fake_camera_para(opt, label)
            label['extrinsic'] = np.eye(4)[None, ...].repeat(opt.num_img, axis=0)
            label['trans'] = np.zeros_like(label['trans'])
            label['trans'][:, :, :, 2] = opt.init_camera_para['dz']

        label['img'] = label['img_crop']
        label['mask'] = label['mask_crop']
        label['intrinsic'] = label['intrinsic_crop']
        label['kp2d'] = label['kp2d_crop']

        # label['part_segmentation'] = label['part_segmentation_crop']

    else:
        # camera parameters
        label['pyrender_camera_pose'] = np.array([[1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, -1, 0],
                                                [0, 0, 0, 1]])  # transform camera coordinate
        label['extrinsic'] = np.eye(4)[None, ...].repeat(opt.num_img, axis=0)
        label['trans'] = np.zeros((opt.num_img, 2, 1, 3))
        label['trans'][:, :, :, 2] = opt.init_camera_para['dz']


        # kp2d pred
        kp2d_path = os.path.join(opt.input_dataset['kp2d_dir'],
                                 opt.input_dataset['sequence_name'],
                                 'kp2d_pred/kp2d_tracked.pkl')
        with open(kp2d_path, 'rb') as f:
            kp2d_label = pkl.load(f, encoding='iso-8859-1')

        label['kp2d_mask'] = kp2d_label['kp2d'][opt.image_id_range[0]:opt.image_id_range[1], :, :, 2:3] > kp2d_conf
        label["kp2d"] = label['kp2d_mask'] * kp2d_label['kp2d'][opt.image_id_range[0]:opt.image_id_range[1], :, :, :2]
        label["joint_smplx_2_coco"] = [55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


        # load img and mask
        opt.img_dir = os.path.join(opt.input_dataset['img_dir'], opt.input_dataset['sequence_name'])
        img = cv2.imread(os.path.join(opt.img_dir, 'image_%s.jpg' % str(0).zfill(5)))
        opt.img_size = [img.shape[1], img.shape[0]]


        # pose shape mean parameters
        smpl_mean_para_file = os.path.join(abspath, '../../../data/neutral_smpl_mean_params.h5')
        with h5py.File(smpl_mean_para_file, 'r') as fp:
            mean_pose = np.array(fp['pose']).reshape(24, 3)[:22]
            mean_shape = np.array(fp['shape']).reshape(1, 10)
        label['mean_pose'] = mean_pose
        # label['mean_pose'][0, 0] += np.pi
        label['mean_shape'] = mean_shape


        # load part segmentation
        label['smplx_misc'] = smplx_part_label()
        # label = get_part_segmentation_mask(label, gender, img_id)

        # crop img
        label = crop_img_with_fake_camera_para(opt, label)

        # label['img'] = label['img_crop']
        # if opt.mask_weight > 0:
        #     label['mask'] = label['mask_crop']
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


def create_log(exp_name, save_img):
    log_dir = os.path.join(abspath, 'output', exp_name)
    config_path = os.path.join(abspath, 'config.py')
    return Logger(log_dir, config_path, save_img=save_img)


class renderSequence(object):

    def __init__(self, opt):
        self.list_now = 0
        self.total_list = opt.num_render_list
        self.sequence_list = [[] for i in range(self.total_list)]


        integer = opt.num_img // self.total_list
        remainder = opt.num_img % self.total_list

        for i in range(integer):
            for j in range(self.total_list):
                id = self.total_list * i + j
                self.sequence_list[j].append(id)

        for j in range(remainder):
            id = self.total_list * integer + j
            self.sequence_list[j].append(id)


    def get(self):
        seq = self.sequence_list[self.list_now]

        return seq


    def update(self):
        if (self.list_now+1) == self.total_list:
            self.list_now = 0
        else:
            self.list_now += 1


def init_opt(conf):
    # preprocess config
    opt = preprocess_opt(conf)

    # submit
    opt.logger = create_log(opt.exp_name,
                            save_img=True)

    # label
    opt.label = get_label(opt,
                          kp2d_conf=opt.kp2d_conf,
                          visualize=False)

    # render and camera
    width, height = opt.img_size
    opt.pyrender = PerspectivePyrender(opt.label['intrinsic'],
                                       opt.label['pyrender_camera_pose'],
                                       width=width, height=height)

    # if opt.mask_weight > 0 or \
    #    opt.texture_render_weight > 0 or \
    #    opt.texture_temporal_consistency_weight > 0 or \
    #    opt.texture_part_consistency_weight > 0:
    #     K = torch.tensor(opt.label['intrinsic'][None, :, :], dtype=torch.float32).to(opt.device)
    #     R = torch.tensor(np.eye(3)[None, :, :], dtype=torch.float32).to(opt.device)
    #     t = torch.tensor(np.zeros((1, 3))[None, :, :], dtype=torch.float32).to(opt.device)
    #
    #     opt.neural_render = PerspectiveNeuralRender(K, R, t, height=height, width=width)

    # opt.render_sequence = renderSequence(opt)

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


def preprocess_opt(conf):
    opt = option()
    opt.exp_name = opt.exp_name_back

    ## set conf
    opt.input_dataset['sequence_name'] = conf['sequence_name']
    opt.conf['LG'].stop_conf.stop_error = conf['LG']['stop_error']
    opt.conf['LP'].stop_conf.stop_error = conf['LP']['stop_error']
    opt.conf['LC'].stop_conf.stop_error = conf['LC']['stop_error']
    opt.break_diff_iter = conf['break_diff_iter']
    opt.break_diff_iter_thresh = conf['break_diff_iter_thresh']

    ##
    opt.eval_gt_file_dir = os.path.join(opt.eval_3dpw['gt_dir'],
                                        opt.input_dataset['sequence_name'])
    opt.model_gender = opt.eval_3dpw['gt_sequence'][opt.input_dataset['sequence_name']]['model_gender']
    opt.image_id_range = opt.eval_3dpw['gt_sequence'][opt.input_dataset['sequence_name']]['image_id_range']

    ## preprocess
    assert opt.image_id_range[0] < opt.image_id_range[1]
    opt.num_img = opt.image_id_range[1] - opt.image_id_range[0]

    opt.exp_name = ('[' + opt.input_dataset['sequence_name'] + ']_') + opt.exp_name
    opt.exp_name += '_id_[%g,%g]' % (opt.image_id_range[0], opt.image_id_range[1])

    opt.exp_name += '_sca_%g' % opt.image_scale

    opt.exp_name += '_se'
    for k, v in opt.conf.items():
        opt.exp_name += '_' + str(v.stop_conf.stop_error)

    opt.exp_name += '_break'
    opt.exp_name += '_' + str(opt.break_diff_iter)
    opt.exp_name += '_' + str(opt.break_diff_iter_thresh)


    # cuda
    torch.backends.cudnn.benchmark = opt.cuda_benchmark
    opt.gpus_list = [int(i) for i in opt.gpus.split(',')]
    opt.device = 'cuda:{}'.format(opt.gpus_list[0]) if -1 not in opt.gpus_list else 'cpu'


    # submit_step_id_list
    opt.submit_step_id_list = \
        [id for id in range(0, opt.total_iter, opt.submit_other_iter)]


    # mask, kp2d weight
    for k, v in opt.conf.items():
        opt.conf[k].loss_weight.kp2d_weight_list = \
            torch.tensor(opt.conf[k].loss_weight.kp2d_weight_list,
                         dtype=torch.float32).to(opt.device)

        opt.conf[k].loss_weight.ground_normal = \
            torch.tensor([[0, 1, 0]],
                         dtype=torch.float32).to(opt.device)

    return opt


def load_check_point(opt, para):
    path = opt.check_point
    load_data_selection = opt.load_data_selection

    with open(path, 'rb') as f:
        load_para = pkl.load(f, encoding='iso-8859-1')

        if "pose_1_9" in load_para:
            para['pose_1'] = load_para["pose_1_9"][:, :, 0, :][:, :, None, :]
            para['pose_2'] = load_para["pose_1_9"][:, :, 1, :][:, :, None, :]
            para['pose_3'] = load_para["pose_1_9"][:, :, 2, :][:, :, None, :]
            para['pose_4'] = load_para["pose_1_9"][:, :, 3, :][:, :, None, :]
            para['pose_5'] = load_para["pose_1_9"][:, :, 4, :][:, :, None, :]
            para['pose_6'] = load_para["pose_1_9"][:, :, 5, :][:, :, None, :]
            para['pose_7'] = load_para["pose_1_9"][:, :, 6, :][:, :, None, :]
            para['pose_8'] = load_para["pose_1_9"][:, :, 7, :][:, :, None, :]
            para['pose_9'] = load_para["pose_1_9"][:, :, 8, :][:, :, None, :]

        if "pose_12_21" in load_para:
            para['pose_12'] = load_para["pose_12_21"][:, :, 0, :][:, :, None, :]
            para['pose_13'] = load_para["pose_12_21"][:, :, 1, :][:, :, None, :]
            para['pose_14'] = load_para["pose_12_21"][:, :, 2, :][:, :, None, :]
            para['pose_15'] = load_para["pose_12_21"][:, :, 3, :][:, :, None, :]
            para['pose_16'] = load_para["pose_12_21"][:, :, 4, :][:, :, None, :]
            para['pose_17'] = load_para["pose_12_21"][:, :, 5, :][:, :, None, :]
            para['pose_18'] = load_para["pose_12_21"][:, :, 6, :][:, :, None, :]
            para['pose_19'] = load_para["pose_12_21"][:, :, 7, :][:, :, None, :]
            para['pose_20'] = load_para["pose_12_21"][:, :, 8, :][:, :, None, :]
            para['pose_21'] = load_para["pose_12_21"][:, :, 9, :][:, :, None, :]

        for k, v in load_para.items():
            # if k in para:
            if k in load_data_selection and \
                load_data_selection[k] == True:
                    para[k] = v[:para[k].shape[0]]

    return para
