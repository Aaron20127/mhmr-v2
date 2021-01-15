from tqdm import tqdm
import cv2
import sys
import os
import pickle as pkl
import numpy as np

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")

from common.debug import draw_kp2d



def match_kp2d(kp2d, kp2d_list):
    min_d = 10000000
    min_index = 0

    for i, kp2d_i in enumerate(kp2d_list):
        d = np.sqrt(np.sum((kp2d[:, :2] - kp2d_i[:, :2]) ** 2, axis=1)).mean()
        if d < min_d:
            min_d = d
            min_index = i

    return kp2d_list[min_index].copy()


def main():
    kp2d_label_path = os.path.join(abspath, '3dpw/kp2d_pred/3dpw_kp2d_pred.pkl')
    with open(kp2d_label_path, 'rb') as f:
        kp2d_label = pkl.load(f, encoding='iso-8859-1')

    kp2d_label = kp2d_label['kp2d']

    for current_id in range(1, len(kp2d_label)):
        last_id = current_id - 1

        kp2d_current = kp2d_label[current_id]
        kp2d_last = kp2d_label[last_id]

        match_kp2d_a = match_kp2d(kp2d_last[0], kp2d_current)
        match_kp2d_b = match_kp2d(kp2d_last[1], kp2d_current)

        kp2d_label[current_id][0] = match_kp2d_a
        kp2d_label[current_id][1] = match_kp2d_b


    # save image
    kp_thresh = 0.3
    img_dir = os.path.join(abspath, '../data_prepare/3DPW/courtyard_dancing_00')

    output_root = os.path.join(abspath, '3dpw/kp2d_pred')
    output_dir = os.path.join(output_root, 'tracked_kpt_thr_%g' % kp_thresh)
    os.makedirs(output_dir, exist_ok=True)

    tqdm_iter = tqdm(kp2d_label, leave=True)
    tqdm_iter.set_description('')

    for i, kp2d_ in enumerate(tqdm_iter):
        img_name = 'image_%s.jpg' % str(i).zfill(5)
        I = cv2.imread(os.path.join(img_dir, img_name))

        I = draw_kp2d(I, kp2d_[0], radius=8, color=(255, 0, 0), kp_thresh=kp_thresh)
        I = draw_kp2d(I, kp2d_[1], radius=8, color=(0, 0, 255), kp_thresh=kp_thresh)

        cv2.imwrite(os.path.join(output_dir, img_name), I)


    # save pkl
    save_path = os.path.join(output_root, '3dpw_kp2d_pred_tracked.pkl')
    os.makedirs(output_root, exist_ok=True)

    output = {
            'kp2d': kp2d_label[:, :2]
        }
    with open(save_path, 'wb') as f:
        pkl.dump(output, f)


if __name__ == '__main__':
    main()