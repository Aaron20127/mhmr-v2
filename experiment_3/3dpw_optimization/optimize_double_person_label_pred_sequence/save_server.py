
import os
import sys
import time
import threading
import numpy as np

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")

from common.render import PerspectivePyrender
from preprocess import create_log

from gevent.server import StreamServer
from mprpc import RPCServer


g_save_thread_on = True
g_lock = threading.Lock()
g_data_list = []

g_data_list [
    {
        'opt': {
            'exp_name': 'test'
            'image_id_range': (1,2)
         }
        'gt': {
            'img': img, {'data': string, 'shape': ()}
            'mask': mask, {'data': string, 'shape': ()}
            'faces': faces {'data': string, 'shape': ()}
         },
        'pred_list': [
            {
                'step_id': step_id,
                'body_kp2d': body_kp2d, {'data': string, 'shape': ()}
                'vertices': vertices, {'data': string, 'shape': ()}
                'mask': mask {'data': string, 'shape': ()}
            }
        ]

    }
]



def save_thread():
    global g_save_thread_on, g_lock, g_data_list

    while g_save_thread_on:

        # creat logger
        g_lock.acquire()
        for old_data in g_data_list:
            if 'logger' is not in old_data:
                old_data['logger'] = \
                    create_log(old_data['opt']['exp_name'],
                               old_data['opt']['image_id_range'])
                break
        g_lock.release()


        # get save data
        save_data = None
        g_lock.acquire()
        for old_data in g_data_list:
            if len(old_data['pred_list']) > 0:
                save_data = {
                    'opt': old_data['opt'],
                    'gt': old_data['gt'],
                    'pred': old_data['pred_list'].pop(0)
                }
        g_lock.release()

        if data is not None:

        # kp2d
        imgs = label['img'].copy()
        for img_id, img in enumerate(imgs):
            kp2d_gt = label['kp2d'][img_id].reshape(-1, 2)
            kp2d = pre_dict['body_kp2d'][img_id].reshape(-1, 2)

            img_kp2d = draw_kp2d(img, kp2d,
                                 radius=np.int(8*opt.image_scale))
            img_kp2d = draw_kp2d(img_kp2d, kp2d_gt, color=(0, 255, 0),
                                 radius=np.int(8*opt.image_scale))

            if opt.save_submit_other:
                logger.add_image('img_%s/kp2d_%s' % (str(img_id).zfill(5), str(id)),
                                 cv2.cvtColor(img_kp2d, cv2.COLOR_BGR2RGB))
            if opt.save_local_other:
                logger.save_image('kp2d_%s.png' % str(id).zfill(5), img_kp2d, img_id=img_id)


        # render
        imgs = label['img'].copy()
        for img_id, img in enumerate(imgs):
            img_render, img_depth = render.render_obj(pre_dict['vertices'][img_id],
                                                      pre_dict['faces'][img_id],
                                                      show_viewer=False)
            img_add_smpl = add_blend_smpl(img_render, img_depth > 0, img)

            if opt.save_submit_other:
                logger.add_image('img_%s/img_add_smpl_%s' % (str(img_id).zfill(5), str(id)),
                                 cv2.cvtColor(img_add_smpl, cv2.COLOR_BGR2RGB))
            if opt.save_local_other:
                logger.save_image('img_add_smpl_%s.png' % str(id).zfill(5), img_add_smpl, img_id=img_id)


        # add mask
        if 'mask' in pre_dict:
            for img_id, mask_pre in enumerate(pre_dict['mask']):
                mask_gt = label['mask'][img_id]

                mask = np.zeros((mask_gt.shape[0], mask_gt.shape[1], 3), dtype=np.uint8)
                mask = draw_mask(mask, mask_gt[:, :, None], color=(0, 255, 0))
                mask = draw_mask(mask, mask_pre[:, :, None], color=(0, 0, 255))

                if opt.save_submit_other:
                    logger.add_image('img_%s/mask_%s' % (str(img_id).zfill(5), str(id)),
                                     cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                if opt.save_local_other:
                    logger.save_image('mask_%s.png' % str(id).zfill(5), mask, img_id=img_id)



        # save obj
        if opt.save_local_other:
            for img_id in range(opt.num_img):
                logger.save_obj('%s.obj' % img_id,
                                pre_dict['vertices'][img_id],
                                pre_dict['faces'][img_id],
                                img_id=img_id)



class SaveServer(RPCServer):
    def __init__(self):
        super(SaveServer, self).__init__()
        pass


    def register(self, opt, gt_data):
        global g_lock, g_data_list

        new_data = {
            'opt': opt,
            'gt': {},
            'pred_list': []
        }

        for k, v in gt_data.items():
            np_str = v['data']
            new_shape = v['shape']
            np_data = np.fromstring(np_str).reshape(new_shape)

            new_data['gt'][k] = np_data

        g_lock.acquire()
        g_data_list.append(new_data)
        g_lock.release()
    
        return 0
        

    def update(self, exp_name, step_id, pred_data):
        global g_lock, g_data_list
        ret = -1

        new_data = {
            'step_id': step_id
        }

        for k, v in pred_data.items():
            np_str = v['data']
            new_shape = v['shape']
            np_data = np.fromstring(np_str).reshape(new_shape)

            new_data[k] = np_data


        g_lock.acquire()
        for old_data in g_data_list
            if exp_name in old_data['opt']:
                old_data['pred_list'].append(new_data)
                ret = 0
                break
        g_lock.release()

        return ret


if __name__ == '__main__':
    # save thread
    sub_thread = threading.Thread(target=save_thread, args=())
    sub_thread.setDaemon = True
    sub_thread.start()

    # sever
    server = StreamServer(('127.0.0.1', 6000), SaveServer())
    server.serve_forever()

