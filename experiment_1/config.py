
class opt(object):
    ## checkpoint
    save_epoch_interval = 1
    checkpoint_path = ''

    # checkpoint_path = 'D:\\paper\\human_body_reconstruction\\code\\mhmr-v2\\experiment_1\\exp' +\
    #                   '\\test_1\\model_epoch_1.pth'
    resume = True

    ## log
    exp_name = 'test_1'       # experiment name
    val_epoch = False
    val_iter_interval = -1
    train_iter_interval = 1

    ## train
    train = True
    seed = 223
    gpus = '-1'             # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True   # accelerate non-dynamic networks

    num_epoch = 1
    batch_size = 2

    lr = 1e-4
    lr_scheduler_factor = 0.97
    lr_scheduler_patience = 4
    lr_scheduler_threshold = 1.0
    lr_verbose = True

    ## loss
    pose_weight = 1.0
    shape_weight = 1.0

    ## dataloader
    num_workers = 0
    shuffle = False
    drop_last = True
    max_train_data = -1
    max_val_data = 5000

    ## preprocess
    gpus_list = [int(i) for i in gpus.split(',')]
    device = 'cuda' if -1 not in gpus_list else 'cpu'

