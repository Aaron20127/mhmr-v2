import os
import sys
import h5py
import numpy as np
import torch

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../')


def collect_3dpw_smpl_para(src_path, save_dir):
    pass


def collect_human36_smpl_para():

    src_file = os.path.join('D:\\paper\\human_body_reconstruction\\datasets\\human_reconstruction\\hum36m-toy', 'annot.h5')
    dst_dir = os.path.join(abspath, 'smpl_para')
    dst_file = os.path.join(dst_dir, 'human36m.h5')

    os.makedirs(dst_dir, exist_ok=True)
    _shape = np.zeros((312188, 10))

    with h5py.File(dst_file, 'w') as dst_f:
        with h5py.File(src_file) as fp:
            # kp2ds = np.array(fp['gt2d']).reshape(-1, 14, 3)
            # self.kp3ds = np.array(fp['gt3d']).reshape(-1, 14, 3)
            shape_raw = np.array(fp['shape'])
            pose_raw = np.array(fp['pose'])
            # imagename = np.array(fp['imagename'])

            shape_total = 0

            for shape_id, shape_i in enumerate(shape_raw):
                find_same = False
                for i in range(shape_total):
                    if np.sum(_shape[i] == shape_i) == 10:
                        find_same = True
                        break

                if not find_same:
                    print(" shape_id / shape_total: (%d / %d)" % (shape_id, shape_total))

                    _shape[shape_total] = shape_i
                    shape_total += 1

            # for pose_id, pose_i in enumerate(pose_raw):
            #     find_same = False
            #     for i in range(pose_total):
            #         if np.sum(_pose[i] == pose_i) == 72:
            #             find_same = True
            #             break
            #
            #     if not find_same:
            #         print(" pose_id / pose_total: (%d / %d)" % (pose_id, pose_total))
            #
            #         _pose[pose_total] = pose_i
            #         pose_total += 1


        # dst_f.create_dataset('gt2d', data=kp2ds)
        # dst_f.create_dataset('gt3d', data=kp3ds)
        dst_f.create_dataset('shape', data=_shape[:shape_total])
        dst_f.create_dataset('pose', data=pose_raw)
        # dst_f.create_dataset('imagename', data=imagename)

    print('shape / pose: (%d, %d)' % (shape_total, len(pose_raw)))
    print('done.')


if __name__ == '__main__':
    collect_human36_smpl_para()
