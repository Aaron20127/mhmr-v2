import json
import torch
import os
import sys
import pickle as pkl
from tqdm import tqdm
import numpy as np

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../")
sys.path.append(abspath + "/../../../")
sys.path.append(abspath + "/../../../../")

from common.utils import save_obj

SMPLX_2_J17 = [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]


def all_matched_min_distance_matching(kp3d_gt_batch, kp3d_pred_batch):
    kp3d_gt_batch = torch.tensor(kp3d_gt_batch[:, None, :, :])
    kp3d_pred_batch = torch.tensor(kp3d_pred_batch[None, :, :, :]). \
        expand(kp3d_gt_batch.shape[0], -1, -1, -1)
    mean_distance = ((kp3d_gt_batch - kp3d_pred_batch)**2). \
        sum(dim=3).sqrt().mean(dim=2)
    pairs_index = torch.argmin(mean_distance, dim=1)

    return pairs_index.numpy()


def all_matched_error(kp3d_gt, kp3d_pred):
    ## gt
    gt_pelvis_smpl = kp3d_gt[:, [0]]
    gt_kp3d = kp3d_gt - gt_pelvis_smpl

    # pred
    pred_pelvis_smpl = kp3d_pred[:, [0]]
    pred_kp3d = kp3d_pred - pred_pelvis_smpl


    # match gt
    paired_idxs = all_matched_min_distance_matching(gt_kp3d, pred_kp3d)
    pred_selected_kp3d = pred_kp3d[paired_idxs]


    # comput MPJPE
    error_smpl = np.sqrt(np.sum((pred_selected_kp3d - gt_kp3d)**2, axis=-1))
    mpjpe = float(error_smpl.mean() * 1000)

    num_person = error_smpl.shape[0]
    total_mpjpe_error = mpjpe * num_person

    # comput MSPE
    pred_selected_pelvis = pred_pelvis_smpl[paired_idxs]

    error_trans = np.sqrt(np.sum((pred_selected_pelvis - gt_pelvis_smpl)**2, axis=-1))
    mspe = float(error_trans.mean() * 1000)

    num_person = error_trans.shape[0]
    total_mspe_error = mspe * num_person


    return num_person, total_mpjpe_error, total_mspe_error



def eval_mpjpe(data_gt, data_pre, seq):
    total_mpjpe_error = 0.0
    total_mspe_error = 0.0
    total_person = 0

    # tqdm_iter = tqdm(seq, leave=True)
    not_eval_person_img_num = []
    for img_num in seq:
        if img_num in data_pre:
            ## matched
            kp3d_gt = data_gt[img_num]['joints'][:, SMPLX_2_J17]
            kp3d_pre =data_pre[img_num]['joints'][:, SMPLX_2_J17]

            num_person, error_mpjpe, error_mspe = all_matched_error(kp3d_gt, kp3d_pre)

            total_mpjpe_error += error_mpjpe
            total_mspe_error += error_mspe
            total_person += num_person


            ## all matched

            # tqdm_iter.set_postfix_str('mpjpe: %.4f' % (total_error/total_person))
            # print('img: %d, mpjpe: %.4f' % (img_num, total_error/total_person))
        else:
            not_eval_person_img_num.append(img_num)


    return total_mpjpe_error/total_person, total_mspe_error/total_person, not_eval_person_img_num


def pack_data(pred_label, start=30):
    data_dict = {}

    for i, joints in enumerate(pred_label['kp3d']):
        data_dict[start+i] = {
            'joints': joints
        }

    return data_dict



def eval_multiperson_mpjpe_mspe_3dpw():

    seq_name = 'courtyard_warmWelcome_00'
    image_id_range = [30, 273]

    gt_file = os.path.join(abspath, '3DPW', 'gt', seq_name, 'kp3d_smpl.pkl')
    pred_file = os.path.join(abspath, '3DPW', 'pred', seq_name, 'kp3d_smpl.pkl')


    with open(gt_file, 'rb') as f:
        gt_label = pkl.load(f, encoding='iso-8859-1')
    with open(pred_file, 'rb') as f:
        pred_label = pkl.load(f, encoding='iso-8859-1')


    gt_data = {}
    for i, joints in enumerate(gt_label['kp3d']):
        gt_data[i] = {
            'joints': joints
        }

    pred_data = {}
    for i, joints in pred_label.items():
        pred_data[i] = {
            'joints': joints.detach().cpu().numpy()
        }

    seq = list(range(image_id_range[0], image_id_range[1]))

    # eval
    mpjpe, mspe, not_eval_person_img_num = \
        eval_mpjpe(gt_data, pred_data, seq)


    # save
    save_file = os.path.join(abspath, '3DPW', seq_name + '.json')
    save_data = {
        'sequences': image_id_range,
        'mpjpe': mpjpe,
        'mspe': mspe,
        'imageSequences': seq_name,
        'not_eval_person_img_num': not_eval_person_img_num
    }

    with open(save_file, 'w') as f:
        f.write(json.dumps(save_data))

    print({'mpjpe': mpjpe, 'mspe': mspe})
    return {'mpjpe': mpjpe, 'mspe': mspe}


if __name__ == '__main__':
    eval_multiperson_mpjpe_mspe_3dpw()


