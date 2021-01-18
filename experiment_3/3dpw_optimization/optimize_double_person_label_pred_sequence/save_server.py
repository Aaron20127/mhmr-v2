import zlib
import os
import sys
import time
import cv2
import threading
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")


from common.debug import draw_kp2d, draw_mask, add_blend_smpl
from common.render import PerspectivePyrender
from common.utils import save_obj

from gevent.server import StreamServer
from mprpc import RPCServer


g_save_thread_on = True
g_lock = threading.Lock()
g_data_list = []

"""
g_data_list [
    {
        'has_init': False,
        'opt': {
            'exp_name': 'test',
            'img_dir_list': [],
            'obj_dir_list': []
         }
        'render': {
            'width': width, {'data': string, 'shape': ()}
            'height': height, {'data': string, 'shape': ()}
            'intrinsic': intrinsic, {'data': string, 'shape': ()}
            'camera_pose': camera_pose, {'data': string, 'shape': ()}
        }
        'gt': {
            'img': img, {'data': string, 'shape': ()}
            'mask': mask, {'data': string, 'shape': ()}
            'faces': faces {'data': string, 'shape': ()}
         },
        'pred_list': [
            {
                'step_id': step_id,
                'pred_dict': {
                    'kp2d': body_kp2d, {'data': string, 'shape': ()}
                    'vertices': vertices, {'data': string, 'shape': ()}
                    'faces': faces, {'data': string, 'shape': ()}
                    'mask': mask {'data': string, 'shape': ()}
                }
            }
        ]

    }
]
"""

def submit(opt, render, gt, pred_dict):
    step_id = pred_dict['step_id']

    # kp2d
    if 'kp2d' in pred_dict:
        imgs = gt['img'].copy()
        for img_id, img in enumerate(imgs):
            kp2d_gt = gt['kp2d'][img_id].reshape(-1, 2)
            kp2d = pred_dict['kp2d'][img_id].reshape(-1, 2)

            img_kp2d = draw_kp2d(img, kp2d,
                                 radius=np.int(8 * opt['image_scale']))
            img_kp2d = draw_kp2d(img_kp2d, kp2d_gt, color=(0, 255, 0),
                                 radius=np.int(8 * opt['image_scale']))

            save_path = os.path.join(opt['img_dir_list'][img_id],
                                     'kp2d_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_kp2d.astype(np.uint8))

        print('kp2d ok')

    # render
    if 'vertices' in pred_dict and \
       'faces' in pred_dict:

        imgs = gt['img'].copy()
        for img_id, img in enumerate(imgs):
            img_render, img_depth = render.render_obj(pred_dict['vertices'][img_id],
                                                      pred_dict['faces'][img_id],
                                                      show_viewer=False)
            img_add_smpl = add_blend_smpl(img_render, img_depth > 0, img)

            save_path = os.path.join(opt['img_dir_list'][img_id],
                                     'img_add_smpl_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_add_smpl.astype(np.uint8))

        print('render ok')

    # add mask
    if 'mask' in pred_dict:
        for img_id, mask_pre in enumerate(pred_dict['mask']):
            mask_gt = gt['mask'][img_id]

            mask = np.zeros((mask_gt.shape[0], mask_gt.shape[1], 3), dtype=np.uint8)
            mask = draw_mask(mask, mask_gt[:, :, None], color=(0, 255, 0))
            mask = draw_mask(mask, mask_pre[:, :, None], color=(0, 0, 255))

            save_path = os.path.join(opt['img_dir_list'][img_id],
                                     'mask_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, mask.astype(np.uint8))

        print('mask ok')

    # save obj
    if 'vertices' in pred_dict and \
       'faces' in pred_dict:

        for img_id in range(len(gt['img'])):
            save_path = os.path.join(opt['obj_dir_list'][img_id],
                                     '%s.obj' % step_id)
            save_obj(save_path,
                     pred_dict['vertices'][img_id],
                     pred_dict['faces'][img_id])

        print('obj ok')


def save_thread():
    global g_save_thread_on, g_lock, g_data_list

    # render for every experiment
    render_dict = {}


    while g_save_thread_on:

        ## init, create render
        g_lock.acquire()
        for old_data in g_data_list:
            if not old_data['has_init']:
                old_data['opt']['render'] = \
                    PerspectivePyrender(old_data['render']['intrinsic'],
                                        old_data['render']['camera_pose'],
                                        width=old_data['render']['width'],
                                        height=old_data['render']['height'])
                old_data['has_init'] = True
                break
        g_lock.release()


        ## get save data
        save_data = None
        g_lock.acquire()
        for old_data in g_data_list:
            if len(old_data['pred_list']) > 0:
                save_data = {
                    'opt': old_data['opt'],
                    'gt': old_data['gt'],
                    'pred_dict': old_data['pred_list'].pop(0),
                    'render': old_data['opt']['render'],
                }
        g_lock.release()


        # save data
        if save_data is not None:
            submit(save_data['opt'],
                   save_data['render'],
                   save_data['gt'],
                   save_data['pred_dict'])



class SaveServer(RPCServer):
    def __init__(self):
        super(SaveServer, self).__init__()
        pass


    def register(self, opt, gt_data, render_data):
        global g_lock, g_data_list

        # check
        new_data = True
        g_lock.acquire()
        for old_data in g_data_list:
            if opt['exp_name'] == old_data['opt']['exp_name']:
                print('duplicated register!')
                new_data = False
                break
        g_lock.release()

        # register
        if not new_data:
            return -1

        new_data = {
            'has_init': False,
            'opt': opt,
            'gt': {},
            'render': {},
            'pred_list': []
        }

        for k, v in gt_data.items():
            z_str = v['data']
            new_shape = v['shape']
            np_data = np.frombuffer(zlib.decompress(z_str)).reshape(new_shape)
            new_data['gt'][k] = np_data

        for k, v in render_data.items():
            z_str = v['data']
            new_shape = v['shape']
            np_data = np.frombuffer(zlib.decompress(z_str)).reshape(new_shape)
            new_data['render'][k] = np_data

        g_lock.acquire()
        g_data_list.append(new_data)
        g_lock.release()

        print('g_data_list len: %d' % len(g_data_list))
        return 0
        

    def update(self, exp_name, step_id, pred_dict):
        global g_lock, g_data_list
        ret = -1

        new_data = {
            'step_id': step_id,
        }

        for k, v in pred_dict.items():
            z_str = v['data']
            new_shape = v['shape']
            np_data = np.frombuffer(zlib.decompress(z_str)).reshape(new_shape)

            new_data[k] = np_data


        g_lock.acquire()
        for old_data in g_data_list:
            if exp_name == old_data['opt']['exp_name']:
                old_data['pred_list'].append(new_data)
                ret = 0
                break
        g_lock.release()

        print('g_data_list len: %d' % len(g_data_list))
        return ret


if __name__ == '__main__':
    # save thread
    sub_thread = threading.Thread(target=save_thread, args=())
    sub_thread.setDaemon = True
    sub_thread.start()

    # sever
    server = StreamServer(('127.0.0.1', 6000), SaveServer())
    server.serve_forever()

