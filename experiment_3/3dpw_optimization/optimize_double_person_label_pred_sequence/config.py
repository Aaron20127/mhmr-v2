import os
abspath = os.path.abspath(os.path.dirname(__file__))


class opt(object):
    ## cuda
    gpus = '-1'              # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True    # accelerate non-dynamic networks

    ## data preprocess
    image_id_range = [40, 42]     # attention: [0, 2] only use img 0 and 1, total img == 2
    gender_list = ['female', 'male']
    kp2d_conf = 0.3               # min kp2d confidence
    image_scale = 0.25
    side_expand = 10

    ## load parameter
    resume = True
    check_point = abspath + "/output/" + \
                            "T1_id_[40,42]_lr_0.002_sha_5_kp2_1_pri_1000_sc_10000_gpc_1_bpc_1_sca_0.25_cnf_0.3/" + \
                            "check_point/" + \
                            "02000.pkl"

    ## log
    exp_name = 'T0'
    submit_scalar_iter = 20
    submit_other_iter = 500

    use_save_server = True  # use save server to save
    server_ip_port_list = [['127.0.0.1', 40030]]
    # server_ip_port_list = [['127.0.0.1', 40030],
    #                        ['127.0.0.1', 40031],
    #                        ['127.0.0.1', 40032],
    #                        ['127.0.0.1', 40033],
    #                        ['127.0.0.1', 40034],
    #                        ['127.0.0.1', 40035]]

    ## optimize
    total_iter = 15000

    ## learning rate
    lr = 20e-4


    ## stage list
    stage_list = []


    ## loss
    num_sample_touch_face = 20  # for touch loss

    mask_weight = 0
    part_mask_weight = 0
    kp2d_weight = 1
    pose_reg_weight = 0
    shape_reg_weight = 5
    collision_weight = 0
    touch_weight = 0
    pose_prior_weight = 1000

    global_pose_consistency_weight = 1
    transl_consistency_weight = 0
    body_pose_consistency_weight = 1

    shape_consistency_weight = 10000
    kp3d_consistency_weight = 0




