import torch

class opt(object):
    # data preprocess
    image_scale = 0.2
    side_expand = 20

    ## log
    exp_name = 'test_5'

    ## optimize
    total_iter = 2400

    ## learning rate
    lr = 10e-4

    ## loss
    mask_weight = 1
    part_mask_weight = 1
    kp2d_weight = 0.5
    pose_weight = 70
    shape_weight = 0.001

    ## cuda
    gpus = '0'             # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True   # accelerate non-dynamic networks

    ## preprocess
    torch.backends.cudnn.benchmark = cuda_benchmark
    gpus_list = [int(i) for i in gpus.split(',')]
    device = 'cuda' if -1 not in gpus_list else 'cpu'

