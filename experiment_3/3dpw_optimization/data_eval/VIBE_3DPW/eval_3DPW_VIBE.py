import json
import torch
import os
import sys
import pickle as pkl
from tqdm import tqdm
import numpy as np
import joblib

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


def all_matched_error(kp3d_gt, kp3d_pred, vertices_gt, vertices_pred):
    ## gt
    gt_pelvis_smpl = kp3d_gt[:, [0]]
    gt_kp3d = kp3d_gt - gt_pelvis_smpl
    gt_vertices = vertices_gt - gt_pelvis_smpl

    # pred
    pred_pelvis_smpl = kp3d_pred[:, [0]]
    pred_kp3d = kp3d_pred - pred_pelvis_smpl
    pred_vertices = vertices_pred - pred_pelvis_smpl



    ## match gt
    paired_idxs = all_matched_min_distance_matching(gt_kp3d, pred_kp3d)
    pred_selected_kp3d = pred_kp3d[paired_idxs]


    ## comput MPJPE
    error_smpl = np.sqrt(np.sum((pred_selected_kp3d - gt_kp3d)**2, axis=-1))
    mpjpe = float(error_smpl.mean() * 1000)

    num_person = error_smpl.shape[0]
    total_mpjpe_error = mpjpe * num_person

    ## comput MSPE
    pred_selected_pelvis = pred_pelvis_smpl[paired_idxs]

    error_trans = np.sqrt(np.sum((pred_selected_pelvis - gt_pelvis_smpl)**2, axis=-1))
    mspe = float(error_trans.mean() * 1000)

    num_person = error_trans.shape[0]
    total_mspe_error = mspe * num_person


    # comput MAVE
    pred_selected_vertices = pred_vertices[paired_idxs]

    error_vertices = np.sqrt(np.sum((pred_selected_vertices - gt_vertices)**2, axis=-1))
    mave = float(error_vertices.mean() * 1000)

    num_person = error_vertices.shape[0]
    total_mave_error = mave * num_person


    return num_person, total_mpjpe_error, total_mspe_error, total_mave_error



def eval_mpjpe(data_gt, data_pre, seq):
    total_mpjpe_error = 0.0
    total_mspe_error = 0.0
    total_mave_error = 0.0
    total_person = 0

    # tqdm_iter = tqdm(seq, leave=True)
    not_eval_person_img_num = []
    for img_num in seq:
        if img_num in data_pre:

            ## matched
            # kp3d_gt = data_gt[img_num]['joints'][:, SMPLX_2_J17]
            kp3d_gt = data_gt['kp3d'][img_num][:, SMPLX_2_J17]
            kp3d_pre =data_pre[img_num]['joints'][:, SMPLX_2_J17]

            # vertices_gt = data_gt[img_num]['vertices']
            vertices_gt = data_gt['vertices'][img_num]
            vertices_pre = data_pre[img_num]['vertices']

            num_person, error_mpjpe, error_mspe, error_mave = \
                all_matched_error(kp3d_gt, kp3d_pre, vertices_gt, vertices_pre)

            total_mpjpe_error += error_mpjpe
            total_mspe_error += error_mspe
            total_mave_error += error_mave

            total_person += num_person


            ## all matched

            # tqdm_iter.set_postfix_str('mpjpe: %.4f' % (total_error/total_person))
            # print('img: %d, mpjpe: %.4f' % (img_num, total_error/total_person))
        else:
            not_eval_person_img_num.append(img_num)


    mpjpe = total_mpjpe_error/total_person
    mspe = total_mspe_error/total_person
    mave = total_mave_error/total_person

    return mpjpe, mspe, mave, not_eval_person_img_num


""" rotation """
def Rx_mat(theta):
    """绕x轴旋转
        batch x theta
    """
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    M = torch.zeros((theta.size(0), 3, 3), requires_grad=False).to(theta.device)
    M[:, 0, 0]=1
    M[:, 1, 1]=cos
    M[:, 1, 2]=-sin
    M[:, 2, 1]=sin
    M[:, 2, 2]=cos

    return M


def preprocess_vibe(sequence, image_id_range):

    gt_dir = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_eval/VIBE_3DPW'
    gt_file = os.path.join(gt_dir, sequence, 'vibe_output.pkl')


    gt_label = joblib.load(gt_file)

    pred_dict = {}
    for _, seq in gt_label.items():
        for i, frame_id in enumerate(seq['frame_ids']):
            if frame_id not in pred_dict:
                pred_dict[frame_id] = {
                    'vertices': seq['verts'][[i]],
                    'joints': seq['joints3d'][[i]],
                }
            else:
                pred_dict[frame_id]['vertices'] = \
                    np.concatenate((pred_dict[frame_id]['vertices'], seq['verts'][[i]]), axis=0)
                pred_dict[frame_id]['joints'] = \
                    np.concatenate((pred_dict[frame_id]['joints'], seq['joints3d'][[i]]), axis=0)


    #
    for frame_id in range(image_id_range[0], image_id_range[1]):
        if frame_id in pred_dict:
            rot_x = Rx_mat(torch.tensor([np.pi])).numpy()[0]

            # save obj

            pred_dict[frame_id]['vertices'] = np.dot(pred_dict[frame_id]['vertices'], rot_x.T)
            pred_dict[frame_id]['joints'] = np.dot(pred_dict[frame_id]['joints'], rot_x.T)

    return pred_dict


def eval_mpjpe_3dpw(pred_data, sequence, image_id_range, save=True):
    """
        for_save_server
    """

    gt_dir = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_eval/gt/3DPW/'
    gt_file = os.path.join(gt_dir, sequence, 'kp3d_vertices_smpl.pkl')


    with open(gt_file, 'rb') as f:
        gt_label = pkl.load(f, encoding='iso-8859-1')

    # pred_data = {}
    # for i in range(len(kp3d_pred)):
    #     pred_data[image_id_range[0]+i] = {
    #         'joints': kp3d_pred[i],
    #         'vertices': vertices_pred[i],
    #     }

    seq = list(range(image_id_range[0], image_id_range[1]))

    # eval
    mpjpe, mspe, mave, not_eval_person_img_num = \
        eval_mpjpe(gt_label, pred_data, seq)


    # save
    if save:
        save_data = {
            'sequences': image_id_range,
            'mpjpe': mpjpe,
            'mspe': mspe,
            'mave': mave,
            'imageSequences': sequence,
            'not_eval_person_img_num': not_eval_person_img_num
        }

        save_file = sequence + '.json'
        with open(save_file, 'w') as f:
            f.write(json.dumps(save_data))

    return {'mpjpe': mpjpe, 'mspe': mspe, 'mave': mave}


if __name__ == '__main__':
    image_id_range = [30, 273]
    sequence = 'courtyard_basketball_00'

    pred_data = preprocess_vibe(sequence, image_id_range)
    eval_mpjpe_3dpw(pred_data, sequence, image_id_range, save=True)


