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

from common.utils import save_obj

SMPLX_2_J17 = [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]


def save_gt():
    sequence_name = 'courtyard_dancing_01'

    src_root_dir = '/opt/LIWEI/tools/smplx/transfer_model/output/3DPW/'
    dst_root_dir = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_eval/gt/3DPW'

    src_dir = os.path.join(src_root_dir, sequence_name)
    dst_dir = os.path.join(dst_root_dir, sequence_name)


    dst_obj_0_dir = dst_dir + '/obj/0'
    dst_obj_1_dir = dst_dir + '/obj/1'

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_obj_0_dir, exist_ok=True)
    os.makedirs(dst_obj_1_dir, exist_ok=True)

    source_dir_0 = os.path.join(src_dir, '0')
    source_dir_1 = os.path.join(src_dir, '1')

    lable_list_0 = os.listdir(source_dir_0)
    lable_list_1 = os.listdir(source_dir_1)

    lable_list_0.sort()
    lable_list_1.sort()

    data_dict = {}

    faces = 0

    tqdm_iter = tqdm(range(len(lable_list_0)), leave=True)
    for i in tqdm_iter:
        image_name = lable_list_0[i].split('.')[0]
        image_name = image_name.split('_')[1]
        image_name = int(image_name)


        with open(source_dir_0 + '/' + lable_list_0[i], 'rb') as f:
            label_0 = pkl.load(f, encoding='iso-8859-1')
            save_obj(dst_obj_0_dir + '/{}.obj'.format(image_name),
                     label_0['vertices'][0].detach().numpy(),
                     label_0['faces'])

        with open(source_dir_1 + '/' + lable_list_1[i], 'rb') as f:
            label_1 = pkl.load(f, encoding='iso-8859-1')
            save_obj(dst_obj_1_dir + '/{}.obj'.format(image_name),
                     label_1['vertices'][0].detach().numpy(),
                     label_1['faces'])

        faces = label_0['faces']
        data_dict[image_name] = {
                'vertices': np.concatenate((label_0['vertices'].detach().numpy(),
                                      label_1['vertices'].detach().numpy()),
                                      axis=0),
                'joints': np.concatenate((label_0['joints'].detach().numpy(),
                                     label_1['joints'].detach().numpy()),
                                     axis=0),
            }

    # save
    save_path = os.path.join(dst_dir, 'annot_smplx.pkl')
    with open(save_path, 'wb') as f:
        pkl.dump({'faces': faces, 'data': data_dict}, f)


def save_multiperson_eval():
    src_dir = 'H:/paper/code/tools/smplx/transfer_model/output_multiperson'
    dst_dir = 'H:/paper/code/mhmr-v2/experiment_3/3dpw_optimization/data_eval/multiperson'
    dst_obj_dir = os.path.join(dst_dir, 'obj')


    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_obj_dir, exist_ok=True)

    lable_list = os.listdir(src_dir)
    lable_list.sort()

    label_dict = {}
    for filename in lable_list:
        image_num = filename.split('_')[1]

        if image_num not in label_dict:
            label_dict[image_num] = []
        label_dict[image_num].append(filename)


    data_dict = {}
    faces = 0

    tqdm_iter = tqdm(label_dict.keys(), leave=True)
    for key in tqdm_iter:
        for i, filename_pkl in enumerate(label_dict[key]):
            image_name = filename_pkl.split('_')[:2]
            image_name = int(image_name[1])

            with open(src_dir + '/' + filename_pkl, 'rb') as f:
                label_0 = pkl.load(f, encoding='iso-8859-1')
                save_obj(dst_obj_dir + '/{}.obj'.format(filename_pkl),
                         label_0['vertices'][0].detach().numpy(),
                         label_0['faces'])

            faces = label_0['faces']
            if image_name not in data_dict:
                data_dict[image_name] = {
                    'vertices': label_0['vertices'].detach().numpy(),
                    'joints': label_0['joints'].detach().numpy(),
                }
            else:
                data_dict[image_name] = {
                    'vertices': np.concatenate((data_dict[image_name]['vertices'],
                                                label_0['vertices'].detach().numpy()), axis=0),
                    'joints': np.concatenate((data_dict[image_name]['joints'],
                                              label_0['joints'].detach().numpy()), axis=0),
                }


    # save
    with open(dst_dir+'/annotation_smplx.pkl', 'wb') as f:
        pkl.dump({'faces': faces, 'data': data_dict}, f)




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
            kp3d_gt = data_gt[img_num]['joints'][:, SMPLX_2_J17]
            kp3d_pre =data_pre[img_num]['joints'][:, SMPLX_2_J17]

            vertices_gt = data_gt[img_num]['vertices']
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



def eval_mpjpe_multiperson():
    gt_file = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_eval/gt/3dpw_courtyard_dancing_00_smplx.pkl'
    pred_file = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_eval/multiperson/3dpw_courtyard_dancing_00_smplx.pkl'
    # sequences = [30, 273]
    sequences = [30, 273]

    with open(gt_file, 'rb') as f:
        gt_label = pkl.load(f, encoding='iso-8859-1')

    with open(pred_file, 'rb') as f:
        pred_label = pkl.load(f, encoding='iso-8859-1')

    seq = list(range(sequences[0], sequences[1]))

    # eval
    mpjpe, not_eval_person_img_num = eval_mpjpe(gt_label['data'], pred_label['data'], seq)
    print('\nnot eval person img num:')
    print(not_eval_person_img_num)

    # save
    save_data = {
        'sequences': sequences,
        'mpjpe': mpjpe,
        'imageSequences': 'courtyard_dancing_00'
    }
    save_file = os.path.join(abspath, '3dpw_courtyard_dancing_00_multiperson_mpjpe.json')
    with open(save_file, 'w') as f:
        f.write(json.dumps(save_data))



def pack_data(pred_label, start=30):
    data_dict = {}

    for i, joints in enumerate(pred_label['kp3d']):
        data_dict[start+i] = {
            'joints': joints
        }

    return data_dict



def eval_mpjpe_my():
    gt_file = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_eval/gt/3dpw_courtyard_dancing_00_smplx.pkl'
    pred_file = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/optimize_double_person_label_pred_sequence/output/P-3-4_no_kp3d_consistency_III_id_[30,273]_lr_0.002_sha_0.5_kp2_0.01_col_1_sc_10000_j3c_1000_bpc_1000_sca_0.25_cnf_0.3/check_point/00200.pkl'
    save_file_name = '3dpw_courtyard_dancing_00_no_kp3d_consistency_III_mpjpe.json'
    # sequences = [30, 273]
    sequences = [30, 273]

    with open(gt_file, 'rb') as f:
        gt_label = pkl.load(f, encoding='iso-8859-1')

    with open(pred_file, 'rb') as f:
        pred_label = pkl.load(f, encoding='iso-8859-1')
        pred_data = pack_data(pred_label, start=sequences[0])

    seq = list(range(sequences[0], sequences[1]))

    # eval
    mpjpe, not_eval_person_img_num = eval_mpjpe(gt_label['data'], pred_data, seq)
    print('\nnot eval person img num:')
    print(not_eval_person_img_num)

    # save
    save_data = {
        'sequences': sequences,
        'mpjpe': mpjpe,
        'imageSequences': 'courtyard_dancing_00'
    }
    save_file = os.path.join(abspath, save_file_name)
    with open(save_file, 'w') as f:
        f.write(json.dumps(save_data))



def eval_mpjpe_3dpw(kp3d_pred, vertices_pred, image_id_range, gt_dir, save_file='mpjpe.pkl', save=False):
    """
        for_save_server
    """
    gt_file = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_eval/gt/3dpw_courtyard_dancing_00_smplx.pkl'
    # pred_file = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/optimize_double_person_label_pred_sequence/output/P-3-4_no_kp3d_consistency_III_id_[30,273]_lr_0.002_sha_0.5_kp2_0.01_col_1_sc_10000_j3c_1000_bpc_1000_sca_0.25_cnf_0.3/check_point/00200.pkl'
    # save_file_name = '3dpw_courtyard_dancing_00_no_kp3d_consistency_III_mpjpe.json'
    #
    # sequences = [30, 273]


    # print('save_file:', save_file)
    # print('image_id_range:', image_id_range)

    gt_file = os.path.join(gt_dir, 'annot_smplx.pkl')

    image_id_range = [int(image_id_range[0]),
                      int(image_id_range[1])]

    with open(gt_file, 'rb') as f:
        gt_label = pkl.load(f, encoding='iso-8859-1')

    pred_data = {}
    for i in range(len(kp3d_pred)):
        pred_data[image_id_range[0]+i] = {
            'joints': kp3d_pred[i],
            'vertices': vertices_pred[i],
        }

    seq = list(range(image_id_range[0], image_id_range[1]))

    # eval
    mpjpe, mspe, mave, not_eval_person_img_num = \
        eval_mpjpe(gt_label['data'], pred_data, seq)


    # save
    if save:
        save_data = {
            'sequences': image_id_range,
            'mpjpe': mpjpe,
            'mspe': mspe,
            'mave': mave,
            'imageSequences': gt_file.split('/')[-1].split('.')[0],
            'not_eval_person_img_num': not_eval_person_img_num
        }

        with open(save_file, 'w') as f:
            f.write(json.dumps(save_data))

    return {'mpjpe': mpjpe, 'mspe': mspe, 'mave': mave}


if __name__ == '__main__':
    save_gt()

    # save_multiperson_eval()
    # eval_mpjpe_multiperson()

    # eval_mpjpe_my()


