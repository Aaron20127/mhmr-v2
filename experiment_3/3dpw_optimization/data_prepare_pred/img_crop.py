from tqdm import tqdm
import cv2
import sys
import os
import pickle as pkl
import numpy as np

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")


def crop_img(kp2d, img):
    num_kp2d = np.sum(kp2d[:, 2] > 0.3)
    if num_kp2d == 0:
        return None

    kp2d_n = kp2d[kp2d[:, 2] > 0.3, :]

    x1 = min(kp2d_n[:, 0])
    x2 = max(kp2d_n[:, 0])

    y1 = min(kp2d_n[:, 1])
    y2 = max(kp2d_n[:, 1])

    expand = 70

    x1 = x1 - expand if x1 - expand > 0 else 0
    x2 = x2 + expand if x2 + expand < img.shape[1] else img.shape[1]
    y1 = y1 - expand if y1 - expand > 0 else 0
    y2 = y2 + expand if y2 + expand < img.shape[0] else img.shape[0]

    return img[int(y1):int(y2), int(x1):int(x2)].copy()




def crop_3dpw():
    img_root = '/opt/LIWEI/datasets/3DPW/imageFiles/'
    kp2d_root = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_prepare_pred/3dpw'
    save_root = '/opt/LIWEI/mhmr-v2-gitee/mhmr-v2/experiment_3/3dpw_optimization/data_prepare_pred/3dpw_crop'

    img_seq = [
        'courtyard_dancing_00',
        'courtyard_basketball_00',
        'courtyard_capoeira_00',
        'courtyard_captureSelfies_00',
        'courtyard_giveDirections_00',
        'courtyard_hug_00',
        'courtyard_shakeHands_00',
        'downtown_warmWelcome_00'
    ]

    img_num_list = [
        [30, 273],
        [215, 420],
        [0, 360],
        [148, 600],
        [216, 650],
        [30, 540],
        [0, 320],
        [240, 588],
    ]

    for seq_id, seq in enumerate(img_seq):
        save_path_0 = os.path.join(save_root, seq, '0')
        save_path_1 = os.path.join(save_root, seq, '1')

        os.makedirs(save_path_0, exist_ok=True)
        os.makedirs(save_path_1, exist_ok=True)

        img_dir = os.path.join(img_root, seq)
        kp2d_file = os.path.join(kp2d_root, seq, 'kp2d_pred', 'kp2d_tracked.pkl')

        img_names = os.listdir(img_dir)
        img_names.sort()

        print(seq)
        with open(kp2d_file, 'rb') as f:
            kp2d = pkl.load(f, encoding='iso-8859-1')

        tqdm_iter = tqdm(img_names, leave=True)
        for img_id, img_name in enumerate(tqdm_iter):

            if img_id < img_num_list[seq_id][0] or \
               img_id >= img_num_list[seq_id][1]:
               continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)

            img_cropped = crop_img(kp2d['kp2d'][img_id][0], img)
            if img_cropped is not None:
                img_save_path = os.path.join(save_path_0, img_name)
                cv2.imwrite(img_save_path, img_cropped)

            img_cropped = crop_img(kp2d['kp2d'][img_id][1], img)
            if img_cropped is not None:
                img_save_path = os.path.join(save_path_1, img_name)
                cv2.imwrite(img_save_path, img_cropped)








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


    crop_3dpw()




