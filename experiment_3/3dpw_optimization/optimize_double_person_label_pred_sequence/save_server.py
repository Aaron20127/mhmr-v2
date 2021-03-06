import zlib
import os
import sys
import time
import cv2
import threading
import numpy as np
import pickle as pkl
import signal
from argparse import ArgumentParser

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
ip = None
port = None
g_debug_on = True

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
    global port, g_debug_on

    step_id = pred_dict['step_id']

    if 'para' in pred_dict:
        save_path = os.path.join(opt['check_point_dir'],
                                 '%s.pkl' % str(step_id).zfill(5))
        output = pred_dict['para']
        with open(save_path, 'wb') as f:
            pkl.dump(output, f)

    # kp2d
    if 'kp2d' in pred_dict:
        imgs = gt['img'].copy()
        for img_id, img in enumerate(imgs):
            kp2d_gt = gt['kp2d'][img_id].reshape(-1, 2)
            kp2d = pred_dict['kp2d'][img_id].reshape(-1, 2)

            img_kp2d = draw_kp2d(img, kp2d,
                                 radius=int(8 * opt['image_scale']))
            img_kp2d = draw_kp2d(img_kp2d, kp2d_gt,
                                 color=(0, 255, 0), draw_num=True,
                                 radius=int(8 * opt['image_scale']))

            save_path = os.path.join(opt['img_dir_list'][img_id],
                                     'kp2d_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_kp2d.astype(np.uint8))


            dir_id = step_id // opt['submit_other_iter']
            save_path = os.path.join(opt['img_sequence_dir_list'][dir_id],
                                     'kp2d_%s.png' % str(img_id).zfill(5))
            cv2.imwrite(save_path, img_kp2d.astype(np.uint8))
        # print('kp2d ok')


    # neural render texture
    if "img" in pred_dict and "depth" in pred_dict:
        imgs = gt['img'].copy()
        for img_id, img in enumerate(imgs):
            img_texture = (pred_dict['img'][img_id] * 255.0).astype(np.uint8)
            img_depth = pred_dict['depth'][img_id]
            img_add_texture = add_blend_smpl(img_texture, img_depth < 100, img)

            save_path = os.path.join(opt['img_dir_list'][img_id],
                                     'img_add_texture_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_add_texture.astype(np.uint8))

            save_path = os.path.join(opt['img_dir_list'][img_id],
                                     'img_texture_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_texture.astype(np.uint8))


            dir_id = step_id // opt['submit_other_iter']
            save_path = os.path.join(opt['img_sequence_dir_list'][dir_id],
                                     'img_add_texture_%s.png' % str(img_id).zfill(5))
            cv2.imwrite(save_path, img_add_texture.astype(np.uint8))

            save_path = os.path.join(opt['img_sequence_dir_list'][dir_id],
                                     'img_texture_%s.png' % str(img_id).zfill(5))
            cv2.imwrite(save_path, img_texture.astype(np.uint8))

        # print('neural render texture ok')

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


            dir_id = step_id // opt['submit_other_iter']
            save_path = os.path.join(opt['img_sequence_dir_list'][dir_id],
                                     'img_add_smpl_%s.png' % str(img_id).zfill(5))
            cv2.imwrite(save_path, img_add_smpl.astype(np.uint8))
        # print('render ok')

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


            dir_id = step_id // opt['submit_other_iter']
            save_path = os.path.join(opt['img_sequence_dir_list'][dir_id],
                                     'mask_%s.png' % str(img_id).zfill(5))
            cv2.imwrite(save_path, mask.astype(np.uint8))
        # print('mask ok')

    # save obj
    if 'vertices' in pred_dict and \
       'faces' in pred_dict:

        vertices_sequence = None
        faces_sequence = None
        for img_id in range(len(gt['img'])):
            save_path = os.path.join(opt['obj_dir_list'][img_id],
                                     '%s.obj' % step_id)
            save_obj(save_path,
                     pred_dict['vertices'][img_id],
                     pred_dict['faces'][img_id])

            # sequence
            dir_id = step_id // opt['submit_other_iter']
            save_path = os.path.join(opt['obj_sequence_dir_list'][dir_id],
                                     '%s.obj' % str(img_id).zfill(5))
            save_obj(save_path,
                     pred_dict['vertices'][img_id],
                     pred_dict['faces'][img_id])

            # full sequence
            if vertices_sequence is None and \
               faces_sequence is None:
                vertices_sequence = pred_dict['vertices'][img_id]
                faces_sequence = pred_dict['faces'][img_id]
            else:
                faces_sequence = np.concatenate((faces_sequence,
                                                pred_dict['faces'][img_id] +
                                                len(vertices_sequence)),
                                                axis=0)
                vertices_sequence = np.concatenate((vertices_sequence,
                                                   pred_dict['vertices'][img_id]),
                                                   axis=0)


        save_path = os.path.join(opt['obj_sequence_full_dir'],
                                 '%s.obj' % str(step_id))
        save_obj(save_path,
                 vertices_sequence,
                 faces_sequence)

        # print('obj ok')

    if g_debug_on:
        print('[submit %d] %s, step_id %d' % (port, opt['exp_name'], step_id))


def subtract_data(exp_name):
    global g_lock, g_data_list

    g_lock.acquire()
    for old_data in g_data_list:
        if exp_name == old_data['opt']['exp_name']:
            old_data['data_len'] -= 1
            break
    g_lock.release()


def destory():
    global g_lock, g_data_list, port

    g_lock.acquire()
    new_g_data_list = []
    for i, old_data in enumerate(g_data_list):
        if not ((old_data['close'] == True) and \
                (old_data['data_len'] <= 0)):
            new_g_data_list.append(old_data)
        else:
            print('[destory %d] %s' % (port, old_data['opt']['exp_name']))

    g_data_list = new_g_data_list
    g_lock.release()


def save_thread():
    global g_save_thread_on, g_lock, g_data_list


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
            if old_data['data_len'] > 0:
                save_data = {
                    'opt': old_data['opt'],
                    'gt': old_data['gt'],
                    'pred_dict': old_data['pred_list'].pop(0),
                    'render': old_data['opt']['render']
                }
                break
        g_lock.release()


        # save data
        if save_data is not None:
            submit(save_data['opt'],
                   save_data['render'],
                   save_data['gt'],
                   save_data['pred_dict'])


            subtract_data(save_data['opt']['exp_name'])
        else:
            time.sleep(0.1)


        # destory session
        destory()


class SaveServer(RPCServer):
    def __init__(self):
        super(SaveServer, self).__init__()
        pass


    def register(self, opt, gt_data, render_data):
        global g_lock, g_data_list, port, g_debug_on

        # clear same experiment
        g_lock.acquire()
        new_g_data_list = []
        for old_data in g_data_list:
            if opt['exp_name'] != old_data['opt']['exp_name']:
                new_g_data_list.append(old_data)
        g_data_list = new_g_data_list
        g_lock.release()

        # register
        new_data = {
            'has_init': False,
            'opt': opt,
            'gt': {},
            'render': {},
            'pred_list': [],
            'data_len': 0,
            'close': False
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

        if g_debug_on:
            print('[register %d] %s' % (port, opt['exp_name']))

        return 0


    def force_unregister(self, exp_name):
        global g_lock, g_data_list, port, g_debug_on

        # clear same experiment
        g_lock.acquire()
        for old_data in g_data_list:
            if exp_name == old_data['opt']['exp_name']:
                old_data['close'] = True
                old_data['data_len'] = 0
                break
        g_lock.release()

        if g_debug_on:
            print('[force_unregister %d] %s' % (port, exp_name))

        return 0


    def unregister(self, exp_name):
        global g_lock, g_data_list, port, g_debug_on

        # clear same experiment
        g_lock.acquire()
        for old_data in g_data_list:
            if exp_name == old_data['opt']['exp_name']:
                old_data['close'] = True
                break
        g_lock.release()

        if g_debug_on:
            print('[unregister %d] %s' % (port, exp_name))

        return 0


    def update(self, exp_name, step_id, pred_dict, para_dict):
        global g_lock, g_data_list, port, g_debug_on
        ret = -1

        new_data = {
            'step_id': step_id,
            'para': {}
        }

        for k, v in pred_dict.items():
            z_str = v['data']
            new_shape = v['shape']
            np_data = np.frombuffer(zlib.decompress(z_str)).reshape(new_shape)
            new_data[k] = np_data

        for k, v in para_dict.items():
            z_str = v['data']
            new_shape = v['shape']
            np_data = np.frombuffer(zlib.decompress(z_str)).reshape(new_shape)
            new_data['para'][k] = np_data


        g_lock.acquire()
        for old_data in g_data_list:
            if exp_name == old_data['opt']['exp_name']:
                old_data['pred_list'].append(new_data)
                old_data['data_len'] += 1
                ret = 0
                break
        g_lock.release()

        if g_debug_on:
            print('[update %d] %s, step_id %d' % (port, exp_name, step_id))

        return ret


    def remain(self, exp_name):
        global g_lock, g_data_list, ip, port, g_debug_on
        ret = 0
        g_lock.acquire()
        for old_data in g_data_list:
            if exp_name == old_data['opt']['exp_name']:
                ret = old_data['data_len']
                break
        g_lock.release()

        if g_debug_on:
            print('[remain %d] %s, len %d' % (port, exp_name, ret))

        return ret


if __name__ == '__main__':
    # ip, port
    parser = ArgumentParser()
    parser.add_argument('--ip', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=6000, type=int)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    # debug
    g_debug_on = args.debug

    # signal handle
    def handler(sig, argv):
        global g_save_thread_on
        g_save_thread_on = False
        sys.exit(0)
        # os.kill(os.getpid(),signal.SIGKILL)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    # signal.signal(signal.SIGQUIT, handler)

    # save thread
    sub_thread = threading.Thread(target=save_thread, args=())
    sub_thread.setDaemon = True
    sub_thread.start()

    # sever
    ip = args.ip
    port = args.port
    server = StreamServer((ip, port), SaveServer())
    server.serve_forever()

