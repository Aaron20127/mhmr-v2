import torch

class opt(object):
    ## log
    exp_name = 'test_2'

    ## optimize
    total_iter = 10000
    gender = 'female'

    ## learning rate
    lr = 1e-4

    ## loss
    kp2d_weight = 0.05
    pose_weight = 100
    shape_weight = 1

    ## cuda
    gpus = '-1'             # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True   # accelerate non-dynamic networks

    ## preprocess
    torch.backends.cudnn.benchmark = cuda_benchmark
    gpus_list = [int(i) for i in gpus.split(',')]
    device = 'cuda' if -1 not in gpus_list else 'cpu'

