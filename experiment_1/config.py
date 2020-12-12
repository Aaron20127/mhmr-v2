
class opt(object):
    ## checkpoint
    save_epoch_interval = -1
    checkpoint_path = ''
    resume = True

    ## log
    exp_name = 'test_get_average'       # experiment name
    val_iter_interval = -1
    train_iter_interval = 1

    ## train
    train = True
    val = False
    seed = 223
    gpus = '0'             # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True   # accelerate non-dynamic networks

    num_epoch = 10
    batch_size = 2

    lr = 1e-4
    lr_scheduler_factor = 0.99999999
    lr_scheduler_patience = 200
    lr_scheduler_threshold = 1e-4
    lr_verbose = True

    ## loss
    pose_weight = 1.0
    shape_weight = 1.0

    ## dataloader
    num_workers = 0

    ## preprocess
    gpus_list = [int(i) for i in gpus.split(',')]
    device = 'cuda' if -1 not in gpus_list else 'cpu'

