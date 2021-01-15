import torch

class opt(object):
    # data preprocess
    image_id = 140
    gender_list = ['female', 'male']
    # gender_list = ['male', 'female']
    image_scale = 0.25
    side_expand = 10

    ## log
    exp_name = 'test_6'
    sub_scalar_iter = 5
    sub_other_iter = 50

    ## optimize
    total_iter = 30000

    ## learning rate
    lr = 20e-4

    ## loss
    num_sample_touch_face = 20  # for touch loss

    mask_weight = 0
    part_mask_weight = 0
    kp2d_weight = 1
    pose_reg_weight = 0
    shape_reg_weight = 20
    collision_weight = 0
    touch_weight = 0
    pose_prior_weight = 3000


    ## cuda
    gpus = '0'              # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True   # accelerate non-dynamic networks

    ## preprocess
    exp_name = exp_name + '_id_%g' % image_id
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

    torch.backends.cudnn.benchmark = cuda_benchmark
    gpus_list = [int(i) for i in gpus.split(',')]
    device = 'cuda' if -1 not in gpus_list else 'cpu'

