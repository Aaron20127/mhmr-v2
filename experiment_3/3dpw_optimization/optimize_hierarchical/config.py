import os
abspath = os.path.abspath(os.path.dirname(__file__))


# learning parameter grad
requires_grad_para_dict_type_0 = {
    'camera_f': True,
    'camera_cx': False,
    'camera_cy': False,
    'textures': False,
    'pose_0': True,
    'pose_1': False,
    'pose_2': False,
    'pose_3': False,
    'pose_4': False,
    'pose_5': False,
    'pose_6': False,
    'pose_7': False,
    'pose_8': False,
    'pose_9': False,
    'pose_12': False,
    'pose_13': False,
    'pose_14': False,
    'pose_15': False,
    'pose_16': False,
    'pose_17': False,
    'pose_18': False,
    'pose_19': False,
    'pose_20': False,
    'pose_21': False,
    'shape': False,
    'transl': True,
    'left_hand_pose': False,
    'right_hand_pose': False
}

requires_grad_para_dict_type_1 = {
    'camera_f': True,
    'camera_cx': False,
    'camera_cy': False,
    'textures': False,
    'pose_0': True,
    'pose_1': True,
    'pose_2': True,
    'pose_3': True,
    'pose_4': True,
    'pose_5': True,
    'pose_6': True,
    'pose_7': True,
    'pose_8': True,
    'pose_9': True,
    'pose_12': True,
    'pose_13': True,
    'pose_14': True,
    'pose_15': True,
    'pose_16': True,
    'pose_17': True,
    'pose_18': True,
    'pose_19': True,
    'pose_20': True,
    'pose_21': True,
    'shape': True,
    'transl': True,
    'left_hand_pose': False,
    'right_hand_pose': False
}

requires_grad_para_dict_type_2 = {
    'camera_f': False,
    'camera_cx': False,
    'camera_cy': False,
    'textures': False,
    'pose_0': False,
    'pose_1': True,
    'pose_2': True,
    'pose_3': True,
    'pose_4': True,
    'pose_5': True,
    'pose_6': True,
    'pose_7': True,
    'pose_8': True,
    'pose_9': True,
    'pose_12': True,
    'pose_13': True,
    'pose_14': True,
    'pose_15': True,
    'pose_16': True,
    'pose_17': True,
    'pose_18': True,
    'pose_19': True,
    'pose_20': True,
    'pose_21': True,
    'shape': True,
    'transl': False,
    'left_hand_pose': False,
    'right_hand_pose': False
}

requires_grad_para_dict_type_3 = {
    'camera_f': False,
    'camera_cx': False,
    'camera_cy': False,
    'textures': True,
    'pose_0': False,
    'pose_1': False,
    'pose_2': False,
    'pose_3': False,
    'pose_4': False,
    'pose_5': False,
    'pose_6': False,
    'pose_7': False,
    'pose_8': False,
    'pose_9': False,
    'pose_12': False,
    'pose_13': False,
    'pose_14': False,
    'pose_15': False,
    'pose_16': False,
    'pose_17': False,
    'pose_18': False,
    'pose_19': False,
    'pose_20': False,
    'pose_21': False,
    'shape': False,
    'transl': False,
    'left_hand_pose': False,
    'right_hand_pose': False
}


# loss weight
class loss_weight_LG():
    num_sample_touch_face = 20  # for touch loss

    mask_weight = 0
    mask_weight_list = [0, 1]  # in to out, out to in

    part_mask_weight = 0

    kp2d_weight = 1
    kp2d_weight_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    pose_reg_weight = 0
    shape_reg_weight = 0


    collision_batch_size = 2
    collision_weight = 0

    touch_weight = 0
    pose_prior_weight = 0

    global_pose_consistency_weight = 0
    transl_consistency_weight = 0
    body_pose_consistency_weight = 0

    shape_consistency_weight = 0
    kp3d_consistency_weight = 10000

    num_render_list = 1
    texture_render_weight = 0
    texture_temporal_consistency_weight = 0
    texture_part_consistency_weight = 0

    ground_weight = 0

class loss_weight_LP():
    num_sample_touch_face = 20  # for touch loss

    mask_weight = 0
    mask_weight_list = [0, 1]  # in to out, out to in

    part_mask_weight = 0

    kp2d_weight = 0.01
    kp2d_weight_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    pose_reg_weight = 0
    shape_reg_weight = 0.005


    collision_batch_size = 2
    collision_weight = 0

    touch_weight = 0
    pose_prior_weight = 0

    global_pose_consistency_weight = 0
    transl_consistency_weight = 0
    body_pose_consistency_weight = 1000

    shape_consistency_weight = 10000
    kp3d_consistency_weight = 1000

    num_render_list = 1
    texture_render_weight = 0
    texture_temporal_consistency_weight = 0
    texture_part_consistency_weight = 0

    ground_weight = 0

class loss_weight_LC():
    num_sample_touch_face = 20  # for touch loss

    mask_weight = 0
    mask_weight_list = [0, 1]  # in to out, out to in

    part_mask_weight = 0

    kp2d_weight = 0.01
    kp2d_weight_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    pose_reg_weight = 0
    shape_reg_weight = 0.005


    collision_batch_size = 2
    collision_weight = 1

    touch_weight = 0
    pose_prior_weight = 0

    global_pose_consistency_weight = 0
    transl_consistency_weight = 0
    body_pose_consistency_weight = 1000

    shape_consistency_weight = 10000
    kp3d_consistency_weight = 1000

    num_render_list = 1
    texture_render_weight = 0
    texture_temporal_consistency_weight = 0
    texture_part_consistency_weight = 0

    ground_weight = 0


# stop config
class stop_conf_LG():
    last_error = 1000
    stop_error = 0.025

    loss_total_up_count = 50
    loss_stop_up_count_now = 0

    loss_total_down_count = 100
    loss_stop_down_count_now = 0


class stop_conf_LP():
    last_error = 1000
    stop_error = 0.00075

    loss_total_up_count = 50
    loss_stop_up_count_now = 0

    loss_total_down_count = 100
    loss_stop_down_count_now = 0


class stop_conf_LC():
    last_error = 1000
    stop_error = 0.001

    loss_total_up_count = 50
    loss_stop_up_count_now = 0

    loss_total_down_count = 100
    loss_stop_down_count_now = 0


# config in different optimize stage
class conf_LG():
    requires_grad_para_dict = requires_grad_para_dict_type_0
    init_para_dict = {
        'mean_pose': False,
        'mean_shape': False
    }
    loss_weight = loss_weight_LG
    lr = 20e-4

    stop_conf = stop_conf_LG

    stage_name = 'LG'


class conf_LP():
    requires_grad_para_dict = requires_grad_para_dict_type_1
    init_para_dict = {
        'mean_pose': True,
        'mean_shape': True
    }
    loss_weight = loss_weight_LP
    lr = 20e-4
    stop_conf = stop_conf_LP

    stage_name = 'LP'


class conf_LC():
    requires_grad_para_dict = requires_grad_para_dict_type_1
    init_para_dict = {
        'mean_pose': False,
        'mean_shape': False
    }
    loss_weight = loss_weight_LC
    lr = 20e-4
    stop_conf = stop_conf_LC

    stage_name = 'LC'


conf = {
    'LG': conf_LG,
    'LP': conf_LP,
    'LC': conf_LC
}

eval_3dpw = {
    'smpl_type': 'smplx',
    'gt_sequence': {
        'courtyard_dancing_00': {'image_id_range': [30, 273], 'model_gender': ['female', 'male']},
        'courtyard_dancing_01': {'image_id_range': [0, 300], 'model_gender': ['male', 'female']},
        'courtyard_basketball_00': {'image_id_range': [215, 420], 'model_gender': ['female', 'male']},
        'courtyard_captureSelfies_00': {'image_id_range': [148, 600], 'model_gender': ['male', 'female']},
        'courtyard_giveDirections_00': {'image_id_range': [216, 650], 'model_gender': ['male', 'female']},
        'downtown_warmWelcome_00': {'image_id_range': [240, 588], 'model_gender': ['male', 'male']},
        'courtyard_shakeHands_00': {'image_id_range': [0, 320], 'model_gender': ['male', 'female']},
        'courtyard_warmWelcome_00': {'image_id_range': [187, 340], 'model_gender': ['male', 'female']},

        'courtyard_hug_00': {'image_id_range': [30, 540], 'model_gender': ['male', 'female']},
        'courtyard_capoeira_00': {'image_id_range': [0, 360], 'model_gender': ['male', 'female']},
    },
    'gt_dir': '/opt/LIWEI/handover/src/3dpw_optimization/data_eval/gt/3DPW'
}


dataset = {
    'dataset_name': '3dpw',
    'sequence_name': 'courtyard_warmWelcome_00',
    'kp2d_dir': '/opt/LIWEI/handover/src/3dpw_optimization/data_prepare_pred/3dpw',
    'img_dir': '/opt/LIWEI/datasets/3DPW/imageFiles',
}


class option(object):
    ## neural render
    texture_size = 4

    ## cuda
    gpus = '0'              # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True    # accelerate non-dynamic networks

    ## data preprocess
    eval_3dpw = eval_3dpw
    input_dataset = dataset
    use_ground_truth = False

    # image_id_range = [216, 650]    # attention: [0, 2] only use img 0 and 1, total img == 2
    kp2d_conf = 0.3                  # min kp2d confidence
    image_scale = 1
    crop_range = [0, -1, 0, -1] # y1, y2, x1, x2

    ## camera para
    init_camera_para = {
        'f': 2000,
        'dz': 4
    }

    ## load parameter
    conf = conf
    resume = False
    check_point = abspath + "/output/" + \
                  "[courtyard_dancing_00]_M3_9_1_stop_0.005_50_100_id_[30,273]_lr_0.002_kp2_10_j3c_10000_sca_0.1_cnf_0.3" + \
                  "/check_point/" + \
                  "00150.pkl"



    ######################################
    ## log
    save_image_interval = 100
    exp_name_back = 'while_LP+LC'

    submit_scalar_iter = 20 # for loss error
    submit_other_iter = 20  # for eval
    total_iter = 10001      # max iteration

    ## optimized flow control
    use_once_LG = False
    use_LG = False
    use_LC = True

    """
    Optmize LG              # use_once_LG (True/False)
    
    while not stable:
        Optmize LG          # use_LG (True/False) 
        Optmize LP
        
    Optmize LC              # use_LC (True/False) 
    """

    # break LG and LP iteration
    break_diff_iter = 10
    break_diff_iter_thresh = 10
