
class opt(object):
    ## checkpoint
    save_epoch_interval = 1
    checkpoint_path = ''

    # checkpoint_path = '/opt/LIWEI/mhmr-v2/experiment_1/exp/test_1/model_epoch_41.pth'
    resume = True

    ## log
    exp_name = 'test_1'       # experiment name
    val_epoch = False
    val_iter_interval = 1000
    train_iter_interval = 10

    ## train
    train = True
    seed = 223
    gpus = '-1'             # -1 cpu, 0,1,2 ... gpu
    cuda_benchmark = True   # accelerate non-dynamic networks

    num_epoch = 100
    batch_size_train = 16
    batch_size_val = 64

    lr = 1e-4
    lr_scheduler_factor = 0.9999999
    lr_scheduler_patience = 200
    lr_scheduler_threshold = 1e-4
    lr_verbose = True

    ## loss
    pose_weight = 1.0
    shape_weight = 1.0

    ## dataset and dataloader
    num_workers = 4
    shuffle = True
    drop_last = True

    render_img = True
    mask_img = False

    ## preprocess
    gpus_list = [int(i) for i in gpus.split(',')]
    device = 'cuda' if -1 not in gpus_list else 'cpu'

