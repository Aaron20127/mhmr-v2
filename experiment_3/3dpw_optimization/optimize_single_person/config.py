
class opt(object):
    ## log
    exp_name = 'test_6'

    ## optimize
    total_iter = 10000

    ## learning rate
    lr = 1e-4

    ## loss
    kp2d_weight = 0.1
    pose_weight = 100
    shape_weight = 1

    ## preprocess
    device = 'cpu'
    cuda_benchmark = True   # accelerate non-dynamic networks

