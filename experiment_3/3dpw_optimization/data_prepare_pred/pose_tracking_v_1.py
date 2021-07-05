from tqdm import tqdm
import cv2
import sys
import os
import pickle as pkl
import numpy as np

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")

from common.debug import draw_kp2d

def kp2d_distance(kp2d_a, kp2d_b, conf_thresh):
    vis = ((kp2d_b[:, 2] > conf_thresh).astype(np.float32) + \
           (kp2d_a[:, 2] > conf_thresh).astype(np.float32)) > 1
    dis = np.sum(np.sqrt(np.sum((kp2d_a - kp2d_b)**2, axis=1)) * vis) / (np.sum(vis)+1e-6)

    return dis


def best_conf_kp2d_id(kp2d_list):
    # max conf
    conf_max = 0
    best_kp2d_id = 0

    for i, kp2d in enumerate(kp2d_list):
        conf = np.sum(kp2d[:, 2])
        if conf > conf_max:
            conf_max = conf
            best_kp2d_id = i

    return best_kp2d_id


def best_kp2d(kp2d_list, last_kp2d=None, odd_thresh=None, conf_thresh=None):

    if len(kp2d_list) == 0:
        return 0

    if len(kp2d_list) == 1:
        return kp2d_list[0]

    if last_kp2d is None:
        best_id = best_conf_kp2d_id(kp2d_list)
        return kp2d_list[best_id]


    best_id = best_conf_kp2d_id(kp2d_list)
    while kp2d_distance(kp2d_list[best_id], last_kp2d, conf_thresh) > odd_thresh:
        del kp2d_list[best_id]

        if len(kp2d_list) == 1:
            best_id = 0
            break

        best_id = best_conf_kp2d_id(kp2d_list)

    return kp2d_list[best_id]



def cluster_kp2d_distance(kp2d_a_list, kp2d_b, thresh):
    dis_total = 0.0
    dis_len = 0

    for kp2d_a in kp2d_a_list:
        dis = kp2d_distance(kp2d_a, kp2d_b, thresh)
        dis_total += dis
        dis_len += 1

    dis_mean = dis_total / dis_len

    return dis_mean


def cluster(kp2d_label, conf_thresh=0.3, dis_thresh=10):
    kp2d_class_a_list = [[] for i in range(kp2d_label.shape[0])]
    kp2d_class_b_list = [[] for i in range(kp2d_label.shape[0])]

    init_class_a = kp2d_label[0][0]
    init_class_b = kp2d_label[0][1]

    for i, kp2ds in enumerate(kp2d_label):
        for j, kp2d in enumerate(kp2ds):
            if np.sum(kp2d[:, 2]) <= 0:
                continue

            ## filter some person
            # vis = kp2d[:, 2] > conf_thresh
            # if np.sum(kp2d[vis, 1] < 600) == np.sum(vis):
            #     continue

            # if i == 148 and j == 1:
            #     kp2d_class_b_list[i].append(kp2d)
            #     continue

            if i == 0:
                dis_a = kp2d_distance(init_class_a, kp2d, conf_thresh)
                dis_b = kp2d_distance(init_class_b, kp2d, conf_thresh)
                if dis_a < dis_b:
                    kp2d_class_a_list[i].append(kp2d)
                else:
                    kp2d_class_b_list[i].append(kp2d)
            else:

                class_a = kp2d_class_a_list[i-1]
                k = i-1
                while len(class_a) == 0:
                    class_a = kp2d_class_a_list[k-1]
                    k -= 1

                class_b = kp2d_class_b_list[i-1]
                k = i-1
                while len(class_b) == 0:
                    class_b = kp2d_class_b_list[k-1]
                    k -= 1

                len_a = len(class_a)
                len_b = len(class_b)

                if len_a > 0 and len_b > 0:
                    dis_a = cluster_kp2d_distance(class_a, kp2d, conf_thresh)
                    dis_b = cluster_kp2d_distance(class_b, kp2d, conf_thresh)

                    if dis_a <= 0.0 or dis_b <= 0.0:
                        continue

                    # if i == 116 and j == 0:
                    #     kp2d_class_a_list[i].append(kp2d)
                    #     continue

                    if dis_a < dis_b:
                        kp2d_class_a_list[i].append(kp2d)
                    else:
                        kp2d_class_b_list[i].append(kp2d)


                elif len_a > 0 and len_b == 0:
                    dis_a = cluster_kp2d_distance(class_a, kp2d, conf_thresh)

                    if dis_a < dis_thresh:
                        kp2d_class_a_list[i].append(kp2d)
                    else:
                        kp2d_class_b_list[i].append(kp2d)
                elif len_a == 0 and len_b > 0:
                    dis_b = cluster_kp2d_distance(class_b, kp2d, conf_thresh)
                    if dis_b < dis_thresh:
                        kp2d_class_b_list[i].append(kp2d)
                    else:
                        kp2d_class_a_list[i].append(kp2d)

    return [kp2d_class_a_list, kp2d_class_b_list]


def main(img_dir, pkl_dir, src_file, dst_file):
    conf_thresh = 0.3
    dis_thresh = 12.0
    odd_thresh = 40.0
    output_tracked_dir = os.path.join(pkl_dir, 'tracked_kpt_thr_%g' % conf_thresh)
    output_clustered_dir = os.path.join(pkl_dir, 'clustered_kpt_thr_%g' % conf_thresh)
    os.makedirs(output_tracked_dir, exist_ok=True)
    os.makedirs(output_clustered_dir, exist_ok=True)


    with open(src_file, 'rb') as f:
            kp2d_label = pkl.load(f, encoding='iso-8859-1')

    kp2d_label = kp2d_label['kp2d']

    # 1.cluster
    kp2d_clustered = cluster(kp2d_label, conf_thresh=conf_thresh, dis_thresh=dis_thresh)

    for i in range(len(kp2d_clustered[0])):
        img_name = 'image_%s.jpg' % str(i).zfill(5)
        # img_name = 'img_%s.jpg' % str(i).zfill(6)
        I = cv2.imread(os.path.join(img_dir, img_name))

        for kp2d in kp2d_clustered[0][i]:
            I = draw_kp2d(I, kp2d, radius=8, color=(255, 0, 0), kp_thresh=conf_thresh)

        for kp2d in kp2d_clustered[1][i]:
            I = draw_kp2d(I, kp2d, radius=8, color=(0, 0, 255), kp_thresh=conf_thresh)

        cv2.imwrite(os.path.join(output_clustered_dir, img_name), I)


    # 2.choosing best kp2d
    kp2d_label_new = np.zeros((kp2d_label.shape[0], 2, kp2d_label.shape[2], kp2d_label.shape[3]))
    for i in range(len(kp2d_clustered[0])):
        # if i == 0:
        if len(kp2d_clustered[0][i]) > 0:
            kp2d_label_new[i][0] = best_kp2d(kp2d_clustered[0][i])
        if len(kp2d_clustered[1][i]) > 0:
            kp2d_label_new[i][1] = best_kp2d(kp2d_clustered[1][i])
        # else:
        #     if len(kp2d_clustered[0][i]) > 0:
        #         last_kp2d = kp2d_label_new[i-1][0]
        #         k = i-1
        #         while np.sum(last_kp2d[:, 2] > conf_thresh) == 0:
        #             last_kp2d = kp2d_label_new[k-1][0]
        #             k -= 1
        #
        #         kp2d_label_new[i][0] = best_kp2d(kp2d_clustered[0][i], last_kp2d, odd_thresh, conf_thresh)
        #
        #     if len(kp2d_clustered[1][i]) > 0:
        #         last_kp2d = kp2d_label_new[i-1][1]
        #         k = i-1
        #         while np.sum(last_kp2d[:, 2] > conf_thresh) == 0:
        #             last_kp2d = kp2d_label_new[k-1][1]
        #             k -= 1
        #         kp2d_label_new[i][1] = best_kp2d(kp2d_clustered[1][i], last_kp2d, odd_thresh, conf_thresh)


    # save image
    for i, kp2d_ in enumerate(kp2d_label_new):
        img_name = 'image_%s.jpg' % str(i).zfill(5)
        # img_name = 'img_%s.jpg' % str(i).zfill(6)

        I = cv2.imread(os.path.join(img_dir, img_name))

        I = draw_kp2d(I, kp2d_[0], radius=8, color=(255, 0, 0), kp_thresh=conf_thresh)
        I = draw_kp2d(I, kp2d_[1], radius=8, color=(0, 0, 255), kp_thresh=conf_thresh)

        cv2.imwrite(os.path.join(output_tracked_dir, img_name), I)


    # save pkl
    output = {
        'kp2d': kp2d_label_new
    }
    with open(dst_file, 'wb') as f:
        pkl.dump(output, f)


if __name__ == '__main__':
    ## 3dpw
    """
        courtyard_dancing_01, courtyard_basketball_00,
        courtyard_capoeira_00, courtyard_captureSelfies_00
        courtyard_giveDirections_00, courtyard_hug_00,
        courtyard_shakeHands_00, courtyard_warmWelcome_00
        downtown_warmWelcome_00, courtyard_dancing_00
    """

    ## mupots-3d
    """
        TS11, TS12
    """

    # img_dir = '/opt/LIWEI/datasets/mupots-3d/mupots-3d-eval/MultiPersonTestSet/TS12'
    # pkl_dir = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_prepare_pred/mupots-3d/TS12/kp2d_pred'

    seq_name = 'courtyard_dancing_01'

    img_dir = '/opt/LIWEI/datasets/3DPW/imageFiles/' + seq_name
    pkl_dir = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_prepare_pred/3dpw/' + seq_name + '/kp2d_pred'

    src_file = os.path.join(pkl_dir, 'kp2d.pkl')
    dst_file = os.path.join(pkl_dir, 'kp2d_tracked.pkl')
    main(img_dir, pkl_dir, src_file, dst_file)