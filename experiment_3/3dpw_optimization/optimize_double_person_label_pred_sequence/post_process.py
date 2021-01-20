import os
import sys
import numpy as np
import time
import threading
from mprpc import RPCClient
import zlib
import cv2

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")

from common.debug import draw_kp2d, draw_mask, add_blend_smpl


g_save_thread_on = True
g_lock = threading.Lock()
g_opt = None
g_save_list = []


def pack(data):
    c_data = zlib.compress(np.array(data). \
                           astype(np.float64).tostring())
    return c_data


def update_server(opt, client_index, client_list, pred_dict):
    client = client_list[client_index]

    # is busy ?
    ret = client.call('remain', opt.exp_name)
    if ret != 0:
        return ret

    # send data
    exp_name = opt.exp_name
    step_id = pred_dict['step_id']

    new_pred_dict = {}
    if 'mask' in pred_dict:
        mask = pred_dict['mask']
        new_pred_dict['mask'] = {'data': pack(mask), 'shape': mask.shape}
    if 'kp2d' in pred_dict:
        kp2d = pred_dict['kp2d']
        new_pred_dict['kp2d'] = {'data': pack(pred_dict['kp2d']), 'shape': kp2d.shape}
    if 'vertices' in pred_dict:
        vertices = pred_dict['vertices']
        new_pred_dict['vertices'] = {'data': pack(pred_dict['vertices']), 'shape': vertices.shape}
    if 'faces' in pred_dict:
        faces = pred_dict['faces']
        new_pred_dict['faces'] = {'data': pack(pred_dict['faces']), 'shape': faces.shape}


    ret = client.call('update', exp_name, step_id, new_pred_dict)
    if ret != 0:
        print('register server %d failed, ret %d' % (client_index, ret))

    return ret



def register_server(opt, client_list):
    height, width = opt.label['img'].shape[1:3]
    intrinsic = opt.label['intrinsic']
    camera_pose = opt.label['pyrender_camera_pose']

    img = opt.label['img']
    mask = opt.label['mask']
    kp2d = opt.label['kp2d']

    opt = {
        'exp_name': g_opt.exp_name,
        'image_scale': g_opt.image_scale,
        'img_dir_list': g_opt.logger.save_img_dir_list,
        'obj_dir_list': g_opt.logger.save_obj_dir_list
    }
    render = {
        'width': {'data': pack(width), 'shape': (1)},
        'height':  {'data': pack(height), 'shape': (1)},
        'intrinsic':  {'data': pack(intrinsic), 'shape': intrinsic.shape},
        'camera_pose':  {'data': pack(camera_pose), 'shape': camera_pose.shape},
    }
    gt = {
        'img': {'data': pack(img), 'shape': img.shape},
        'mask': {'data': pack(mask), 'shape': mask.shape},
        'kp2d': {'data': pack(kp2d), 'shape': kp2d.shape},
    }

    for i, client in enumerate(client_list):
        ret = client.call('register', opt, gt, render)
        if ret != 0:
            print('register server %d failed, ret %d' % (i, ret))


def unregister_server(opt, client_list):
    for i, client in enumerate(client_list):
        ret = client.call('unregister', opt.exp_name)
        if ret != 0:
            print('unregister server %d failed, ret %d' % (i, ret))


def remain_server(opt, client_list):
    ret = 0
    for i, client in enumerate(client_list):
        ret += client.call('remain', opt.exp_name)

    return ret


def submit_thread():
    global g_save_thread_on, g_lock, g_opt, g_save_list

    client_list = [RPCClient(ip_port[0], ip_port[1]) for \
                   ip_port in g_opt.server_ip_port_list]
    client_index = 0

    # register
    register_server(g_opt, client_list)

    while g_save_thread_on or \
          len(g_save_list) > 0:
        # get new save data
        save_dict = None
        g_lock.acquire()
        if len(g_save_list) > 0:
            save_dict = g_save_list.pop(0)
        g_lock.release()

        # save data
        if save_dict is not None:
            while True:
                ret = update_server(g_opt,
                                    client_index,
                                    client_list,
                                    save_dict)
                client_index += 1
                if client_index == len(client_list):
                    client_index = 0

                if ret == 0:
                    break
        else:
            time.sleep(0.1)


    # ## waiting
    print('waiting to save data ...')
    total = remain_server(g_opt, client_list)
    while total > 0:
        print('remain %d' % total)
        total = remain_server(g_opt, client_list)
        time.sleep(5)

    print('done.')
    unregister_server(g_opt, client_list)


def submit(opt, render, id, loss_dict, pre_dict):
    logger = opt.logger
    label = opt.label

    opt.logger.update_summary_id(id)

    # submit scalar
    if id % opt.submit_scalar_iter == 0:
        logger.scalar_summary_dict(loss_dict)

    if id % opt.submit_other_iter == 0:
        # kp2d
        if 'kp2d' in pre_dict:
            imgs = label['img'].copy()
            for img_id, img in enumerate(imgs):
                kp2d_gt = label['kp2d'][img_id].reshape(-1, 2)
                kp2d = pre_dict['kp2d'][img_id].reshape(-1, 2)

                img_kp2d = draw_kp2d(img, kp2d,
                                     radius=np.int(8*opt.image_scale))
                img_kp2d = draw_kp2d(img_kp2d, kp2d_gt, color=(0, 255, 0),
                                     radius=np.int(8*opt.image_scale))


                logger.add_image('img_%s/kp2d_%s' % (str(img_id).zfill(5), str(id)),
                                 cv2.cvtColor(img_kp2d, cv2.COLOR_BGR2RGB))

                logger.save_image('kp2d_%s.png' % str(id).zfill(5), img_kp2d, img_id=img_id)


        # render
        if 'faces' in pre_dict and \
           'vertices' in pre_dict:

            imgs = label['img'].copy()
            for img_id, img in enumerate(imgs):
                img_render, img_depth = render.render_obj(pre_dict['vertices'][img_id],
                                                          pre_dict['faces'][img_id],
                                                          show_viewer=False)
                img_add_smpl = add_blend_smpl(img_render, img_depth > 0, img)

                logger.add_image('img_%s/img_add_smpl_%s' % (str(img_id).zfill(5), str(id)),
                                 cv2.cvtColor(img_add_smpl, cv2.COLOR_BGR2RGB))
                logger.save_image('img_add_smpl_%s.png' % str(id).zfill(5), img_add_smpl, img_id=img_id)


        # add mask
        if 'mask' in pre_dict:
            for img_id, mask_pre in enumerate(pre_dict['mask']):
                mask_gt = label['mask'][img_id]

                mask = np.zeros((mask_gt.shape[0], mask_gt.shape[1], 3), dtype=np.uint8)
                mask = draw_mask(mask, mask_gt[:, :, None], color=(0, 255, 0))
                mask = draw_mask(mask, mask_pre[:, :, None], color=(0, 0, 255))

                logger.add_image('img_%s/mask_%s' % (str(img_id).zfill(5), str(id)),
                                 cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                logger.save_image('mask_%s.png' % str(id).zfill(5), mask, img_id=img_id)


        # save obj
        if 'faces' in pre_dict and \
           'vertices' in pre_dict:

            for img_id in range(opt.num_img):
                logger.save_obj('%s.obj' % id,
                                pre_dict['vertices'][img_id],
                                pre_dict['faces'][img_id],
                                img_id=img_id)

        # add part mask
        # part_full_mask = np.zeros((label['mask'].shape[0], label['mask'].shape[1], 3), dtype=np.uint8)
        # for part_name, seg_mask in label['part_segmentation'].items():
        #     part_full_mask = draw_mask(part_full_mask, seg_mask[:, :, None], color=(0, 255, 0))
        # # for part_mask in pre_dict['part_mask_pre']:
        # #     part_full_mask = draw_mask(part_full_mask, part_mask[:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        # part_full_mask = draw_mask(part_full_mask, pre_dict['mask_pre'][:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        # logger.add_image('part_full_mask_'+str(id), cv2.cvtColor(part_full_mask, cv2.COLOR_BGR2RGB))
        # logger.save_image('part_full_mask_%s.png' % str(id).zfill(5), part_full_mask)


        # add mesh
        # logger.add_mesh('mesh_'+str(id),
        #                  pre_dict['vertices'].detach().cpu().numpy()[np.newaxis,:],
        #                  pre_dict['faces'][np.newaxis,:])


def save_data(opt, it_id, loss_dict, pred_dict):
    if opt.use_save_server:
        # submit scalar
        if it_id % opt.submit_scalar_iter == 0:
            opt.logger.update_summary_id(it_id)
            opt.logger.scalar_summary_dict(loss_dict)

        # update data
        if it_id % opt.submit_other_iter == 0:
            global g_lock, g_save_list

            g_lock.acquire()
            pred_dict['step_id'] = it_id
            g_save_list.append(pred_dict)
            g_lock.release()

    else:
        submit(opt, opt.pyrender, it_id, loss_dict, pred_dict)


def post_process(opt):
    global g_save_thread_on, g_save_list

    # waiting to send
    print('waiting to send data ...')
    while len(g_save_list) > 0:
        print('remain %d' % len(g_save_list))
        time.sleep(5)
    print('done.')
    g_save_thread_on = False


    # waiting to save
    # print('waiting to save data ...')
    # total = remain_server(g_opt, client_list)
    # while total > 0:
    #     print('remain %d' % total)
    #     total = remain_server(g_opt, client_list)
    #     time.sleep(5)
    #
    # print('done.')
    # unregister_server(g_opt, client_list)


def init_submit_thread(opt):
    global g_opt
    g_opt = opt
    sub_thread = threading.Thread(target=submit_thread, args=())
    sub_thread.setDaemon = True
    sub_thread.start()