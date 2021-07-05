import os
import sys
import numpy as np
import time
import threading
from mprpc import RPCClient
import zlib
import cv2
import pickle as pkl


abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")
sys.path.append(abspath + "/../")

from common.utils import save_obj
from common.debug import draw_kp2d, draw_mask, add_blend_smpl
from data_eval.eval_3DPW import eval_mpjpe_3dpw



g_run_thread = True
g_force_exit_thread = False
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

    # pack data
    exp_name = opt.exp_name
    step_id = pred_dict['step_id']


    visual_data_dict = {}
    if 'visual_data' in pred_dict:
        for k, v in pred_dict['visual_data'].items():
            visual_data_dict[k] = {'data': pack(v), 'shape': v.shape}

    # learning parameters
    save_data_dict = {}
    if 'save_data' in pred_dict:
        for k, v in pred_dict['save_data'].items():
            save_data_dict[k] = {'data': pack(v), 'shape': v.shape}

    # send
    ret = client.call('update', exp_name, step_id, visual_data_dict, save_data_dict)
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
        'submit_other_iter': g_opt.submit_other_iter,
        'img_dir_list': g_opt.logger.save_img_dir_list,
        'obj_dir_list': g_opt.logger.save_obj_dir_list,
        'img_sequence_dir_list': g_opt.logger.save_img_sequence_dir_list,
        'obj_sequence_dir_list': g_opt.logger.save_obj_sequence_dir_list,
        'obj_sequence_full_dir': g_opt.logger.save_obj_sequence_full_dir,
        'check_point_dir': g_opt.logger.save_check_point_dir,
        'eval_gt_file_dir': g_opt.eval_gt_file_dir
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


def force_unregister_server(opt, client_list):
    for i, client in enumerate(client_list):
        ret = client.call('force_unregister', opt.exp_name)
        if ret != 0:
            print('force_unregister server %d failed, ret %d' % (i, ret))


def remain_server(opt, client_list):
    ret = 0
    for i, client in enumerate(client_list):
        ret += client.call('remain', opt.exp_name)

    return ret


def submit_thread():
    global g_run_thread, g_force_exit_thread, g_lock, g_opt, g_save_list

    client_list = [RPCClient(ip_port[0], ip_port[1]) for \
                   ip_port in g_opt.server_ip_port_list]
    client_index = 0

    # register
    # print('start register')
    register_server(g_opt, client_list)
    # print('end register')


    while g_run_thread or \
          len(g_save_list) > 0:

        # force exit
        if g_force_exit_thread:
            force_unregister_server(g_opt, client_list)
            sys.exit(0)

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


    # waiting
    print('waiting to save data ...')
    total = remain_server(g_opt, client_list)
    while total > 0:
        if g_force_exit_thread:
            force_unregister_server(g_opt, client_list)
            sys.exit(0)

        print('remain %d' % total)
        total = remain_server(g_opt, client_list)
        time.sleep(5)

    if g_force_exit_thread:
        force_unregister_server(g_opt, client_list)
    else:
        unregister_server(g_opt, client_list)
    print('done.')


def submit(opt, render, gt, pred_dict):
    render.update_intrinsic(pred_dict['para']['intrinsic'])

    step_id = pred_dict['step_id']
    image_id_range = [int(pred_dict['para']['image_id_range'][0]),
                      int(pred_dict['para']['image_id_range'][1])]
    start_image_num = image_id_range[0]

    if 'para' in pred_dict:
        # save parameter
        save_path = os.path.join(opt.logger.save_check_point_dir,
                                 '%s.pkl' % str(step_id).zfill(5))
        output = pred_dict['para']
        with open(save_path, 'wb') as f:
            pkl.dump(output, f)

        if 'kp3d' in output:
            # eval
            save_path = os.path.join(opt.logger.save_check_point_dir,
                                     'eval_mpjpe_%s.json' % str(step_id).zfill(5))

            eval_mpjpe_3dpw(output['kp3d'],
                            image_id_range,
                            save_file=save_path,
                            save=True)

    # kp2d
    # if 'kp2d' in pred_dict:
    if 'kp2d' in pred_dict:
        imgs = gt['img'].copy()
        for img_id, img in enumerate(imgs):
            kp2d_gt = gt['kp2d'][img_id].reshape(-1, 2)
            kp2d = pred_dict['kp2d'][img_id].reshape(-1, 2)

            img_kp2d = draw_kp2d(img, kp2d,
                                 radius=int(8 * opt.image_scale))
            img_kp2d = draw_kp2d(img_kp2d, kp2d_gt,
                                 color=(0, 255, 0), draw_num=True,
                                 radius=int(8 * opt.image_scale))

            save_path = os.path.join(opt.logger.save_img_dir_list[img_id],
                                     'kp2d_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_kp2d.astype(np.uint8))


            dir_id = step_id // opt.submit_other_iter
            save_path = os.path.join(opt.logger.save_img_sequence_dir_list[dir_id],
                                     'kp2d_%s.png' % str(img_id + start_image_num).zfill(5))
            cv2.imwrite(save_path, img_kp2d.astype(np.uint8))
            # cv2.imwrite(save_path, gt['img'][img_id].astype(np.uint8))
        # print('kp2d ok')


    # neural render texture
    if "img" in pred_dict and "depth" in pred_dict:
        imgs = gt['img'].copy()
        for img_id, img in enumerate(imgs):
            img_texture = (pred_dict['img'][img_id] * 255.0).astype(np.uint8)
            img_depth = pred_dict['depth'][img_id]
            img_add_texture = add_blend_smpl(img_texture, img_depth < 100, img)

            save_path = os.path.join(opt.logger.save_img_dir_list[img_id],
                                     'img_add_texture_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_add_texture.astype(np.uint8))

            save_path = os.path.join(opt.logger.save_img_dir_list[img_id],
                                     'img_texture_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_texture.astype(np.uint8))


            dir_id = step_id // opt.submit_other_iter
            save_path = os.path.join(opt.logger.save_img_sequence_dir_list[dir_id],
                                     'img_add_texture_%s.png' % str(img_id + start_image_num).zfill(5))
            cv2.imwrite(save_path, img_add_texture.astype(np.uint8))

            save_path = os.path.join(opt.logger.save_img_sequence_dir_list[dir_id],
                                     'img_texture_%s.png' % str(img_id + start_image_num).zfill(5))
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

            save_path = os.path.join(opt.logger.save_img_dir_list[img_id],
                                     'img_add_smpl_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, img_add_smpl.astype(np.uint8))


            dir_id = step_id // opt.submit_other_iter
            save_path = os.path.join(opt.logger.save_img_sequence_dir_list[dir_id],
                                     'img_add_smpl_%s.png' % str(img_id + start_image_num).zfill(5))
            cv2.imwrite(save_path, img_add_smpl.astype(np.uint8))
        # print('render ok')

    # add mask
    if 'mask' in pred_dict:
        for img_id, mask_pre in enumerate(pred_dict['mask']):
            mask_gt = gt['mask'][img_id]

            mask = np.zeros((mask_gt.shape[0], mask_gt.shape[1], 3), dtype=np.uint8)
            mask = draw_mask(mask, mask_gt[:, :, None], color=(0, 255, 0))
            mask = draw_mask(mask, mask_pre[:, :, None], color=(0, 0, 255))

            save_path = os.path.join(opt.logger.save_img_dir_list[img_id],
                                     'mask_%s.png' % str(step_id).zfill(5))
            cv2.imwrite(save_path, mask.astype(np.uint8))


            dir_id = step_id // opt.submit_other_iter
            save_path = os.path.join(opt.logger.save_img_sequence_dir_list[dir_id],
                                     'mask_%s.png' % str(img_id + start_image_num).zfill(5))
            cv2.imwrite(save_path, mask.astype(np.uint8))
        # print('mask ok')

    # save obj
    if 'vertices' in pred_dict and \
       'faces' in pred_dict:

        vertices_sequence = None
        faces_sequence = None
        for img_id in range(len(gt['img'])):
            save_path = os.path.join(opt.logger.save_obj_dir_list[img_id],
                                     '%s.obj' % step_id)
            save_obj(save_path,
                     pred_dict['vertices'][img_id],
                     pred_dict['faces'][img_id])

            # sequence
            dir_id = step_id // opt.submit_other_iter
            save_path = os.path.join(opt.logger.save_obj_sequence_dir_list[dir_id],
                                     '%s.obj' % str(img_id + start_image_num).zfill(5))
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


        save_path = os.path.join(opt.logger.save_obj_sequence_full_dir,
                                 '%s.obj' % str(step_id))
        save_obj(save_path,
                 vertices_sequence,
                 faces_sequence)



def post_process(opt):
    global g_lock, g_run_thread, g_save_list

    # waiting to send
    print('waiting to send data ...')
    while len(g_save_list) > 0:
        g_lock.acquire()
        print('remain %d' % len(g_save_list))
        g_lock.release()
        time.sleep(5)
    print('done.')

    g_lock.acquire()
    g_run_thread = False
    g_lock.release()


def force_exit_thread():
    global g_lock, g_force_exit_thread

    g_lock.acquire()
    g_force_exit_thread = True
    g_lock.release()


def init_submit_thread(opt):
    global g_opt
    g_opt = opt

    sub_thread = threading.Thread(target=submit_thread, args=())
    sub_thread.setDaemon = True
    sub_thread.start()

