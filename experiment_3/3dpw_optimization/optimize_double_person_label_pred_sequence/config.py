import torch

class opt(object):
    # data preprocess
    image_id_range = [40, 61]    # attention: [0, 2] only use img 0 and 1, total img == 2
    gender_list = ['female', 'male']
    kp2d_conf = 0.3              # min kp2d confidence
    image_scale = 0.25
    side_expand = 10

    ## log
    exp_name = 'T0'
    submit_scalar_iter = 20
    submit_other_iter = 500

    use_save_server = True  # use save server to save
    server_ip_port_list = [['127.0.0.1', 40030],
                           ['127.0.0.1', 40031],
                           ['127.0.0.1', 40032],
                           ['127.0.0.1', 40033],
                           ['127.0.0.1', 40034],
                           ['127.0.0.1', 40035]]

    ## optimize
    total_iter = 15000

    ## learning rate
    lr = 20e-4

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

    # global_pose_consistency_weight = 0
    # body_pose_consistency_weight = 0
    # transl_consistency_weight = 0
    pose_consistency_weight = 0
    shape_consistency_weight = 10000
    kp3d_consistency_weight = 0

    ## cuda
    gpus = '0'              # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True   # accelerate non-dynamic networks


    ## preprocess
    assert image_id_range[0] < image_id_range[1]
    num_img = image_id_range[1] - image_id_range[0]

    exp_name = exp_name + '_id_[%g,%g]' % (image_id_range[0], image_id_range[1])
    exp_name = exp_name + '_lr_%g' % lr
    if shape_reg_weight > 0:
        exp_name += '_sha_%g' % shape_reg_weight
    if pose_reg_weight > 0:
        exp_name += '_pos_%g' % pose_reg_weight
    if kp2d_weight > 0:
        exp_name += '_kp2_%g' % kp2d_weight
    if collision_weight > 0:
        exp_name += '_col_%g' % collision_weight
    if touch_weight > 0:
        exp_name += '_tou_%g' % touch_weight
    if pose_prior_weight > 0:
        exp_name += '_pri_%g' % pose_prior_weight
    if pose_consistency_weight > 0 and num_img > 1:
        exp_name += '_pc_%g' % pose_consistency_weight
    if shape_consistency_weight > 0 and num_img > 1:
        exp_name += '_sc_%g' % shape_consistency_weight
    if kp3d_consistency_weight > 0 and num_img > 1:
        exp_name += '_j3c_%g' % kp3d_consistency_weight

    exp_name += '_sca_%g' % image_scale
    exp_name += '_cnf_%g' % kp2d_conf


    # cuda
    torch.backends.cudnn.benchmark = cuda_benchmark
    gpus_list = [int(i) for i in gpus.split(',')]
    device = 'cuda' if -1 not in gpus_list else 'cpu'


    # submit_step_id_list
    submit_step_id_list = \
        [id for id in range(0, total_iter, submit_other_iter)]
