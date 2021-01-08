import torch

class opt(object):
    # data preprocess
    image_scale = 0.2
    side_expand = 10

    ## log
    exp_name = 'test_25_collision'
    sub_scalar_iter = 10
    sub_other_iter = 50

    ## optimize
    total_iter = 2400

    ## learning rate
    lr = 2e-4

    ## loss
    num_sample_touch_face = 10  # for touch loss

    mask_weight = 0
    part_mask_weight = 0
    kp2d_weight = 0
    pose_weight = 0.0001
    shape_weight = 0.01
    collision_weight = 0
    touch_weight = 1
    # mask_weight = 1
    # part_mask_weight = 1
    # kp2d_weight = 0.5
    # pose_weight = 100
    # shape_weight = 0.001
    # collision_weight = 100

    ## cuda
    gpus = '0'             # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True   # accelerate non-dynamic networks

    ## preprocess
    torch.backends.cudnn.benchmark = cuda_benchmark
    gpus_list = [int(i) for i in gpus.split(',')]
    device = 'cuda' if -1 not in gpus_list else 'cpu'

