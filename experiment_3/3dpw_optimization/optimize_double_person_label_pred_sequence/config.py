import os
abspath = os.path.abspath(os.path.dirname(__file__))


class opt(object):
    ## neural render
    texture_size = 4

    ## cuda
    gpus = '0'              # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True    # accelerate non-dynamic networks

    ## data preprocess
    image_id_range = [30, 32]     # attention: [0, 2] only use img 0 and 1, total img == 2
    gender_list = ['female', 'male']
    kp2d_conf = 0.3               # min kp2d confidence
    image_scale = 0.25
    side_expand = 100

    ## load parameter
    resume = True
    check_point = abspath + "/output/" + \
                  "M2_id_[30,32]_lr_0.002_sha_0.5_kp2_0.01_sc_10000_j3c_1000_bpc_1000_sca_0.25_cnf_0.3" + \
                  "/check_point/" + \
                  "00100.pkl"

    ## log
    exp_name = 'M3_texture'
    submit_scalar_iter = 20
    submit_other_iter = 100

    use_save_server = True  # use save server to save
    server_ip_port_list = [['127.0.0.1', 40034],
                           ['127.0.0.1', 40035],
                           ['127.0.0.1', 40036],
                           ['127.0.0.1', 40037]]

    ## optimize
    total_iter = 2801

    ## learning rate
    lr = 20e-4

    ## learning parameter grad
    requires_grad_para_dict = {
        'textures': True,
        'pose_0': False,
        'pose_1_9': False,
        'pose_12_21': False,
        'shape': False,
        'transl': False,
        'left_hand_pose': False,
        'right_hand_pose': False
    }

    ## loss
    num_sample_touch_face = 20  # for touch loss

    mask_weight = 0
    mask_weight_list = [0, 1]  # in to out, out to in

    part_mask_weight = 0

    kp2d_weight = 0.01
    # kp2d_weight_list = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2]
    # kp2d_weight_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    kp2d_weight_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    pose_reg_weight = 0
    shape_reg_weight = 0.5
    collision_weight = 0
    touch_weight = 0
    pose_prior_weight = 0

    global_pose_consistency_weight = 0
    transl_consistency_weight = 0
    body_pose_consistency_weight = 0

    shape_consistency_weight = 0
    kp3d_consistency_weight = 0

    texture_render_weight = 100000
    texture_temporal_consistency_weight = 0




