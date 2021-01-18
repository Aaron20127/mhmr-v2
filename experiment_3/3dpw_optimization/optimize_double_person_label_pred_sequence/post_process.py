import os
import numpy as np
import time

abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath + "/../../../")

from common.render import PerspectivePyrender
from common.debug import draw_kp2d, draw_mask, add_blend_smpl


def submit_thread():
    global g_save_thread_on, g_lock, g_opt, g_save_list

    height, width = g_opt.label['img'].shape[1:3]
    render = PerspectivePyrender(g_opt.label['intrinsic'],
                                 g_opt.label['pyrender_camera_pose'],
                                 width=width, height=height)

    while g_save_thread_on:
        # get new save data
        save_dict = None
        g_lock.acquire()
        if len(g_save_list) > 0:
            save_dict = g_save_list.pop(0)
        g_lock.release()

        # save data
        if save_dict is not None:
            id = save_dict['id']
            loss_dict = save_dict['loss_dict']
            pre_dict = save_dict['pre_dict']
            submit(g_opt, render, id, loss_dict, pre_dict)
            print('saved id %d' % id)
        else:
            time.sleep(0.1)


def submit(opt, render, id, loss_dict, pre_dict):
    logger = opt.logger
    label = opt.label

    opt.logger.update_summary_id(id)

    if id % opt.submit_scalar_iter == 0:
        logger.scalar_summary_dict(loss_dict)

    if id % opt.submit_other_iter == 0:
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


        # add part mask
        # part_full_mask = np.zeros((label['mask'].shape[0], label['mask'].shape[1], 3), dtype=np.uint8)
        # for part_name, seg_mask in label['part_segmentation'].items():
        #     part_full_mask = draw_mask(part_full_mask, seg_mask[:, :, None], color=(0, 255, 0))
        # # for part_mask in pre_dict['part_mask_pre']:
        # #     part_full_mask = draw_mask(part_full_mask, part_mask[:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        # part_full_mask = draw_mask(part_full_mask, pre_dict['mask_pre'][:, :, None].detach().cpu().numpy(), color=(0, 0, 255))
        # logger.add_image('part_full_mask_'+str(id), cv2.cvtColor(part_full_mask, cv2.COLOR_BGR2RGB))
        # logger.save_image('part_full_mask_%s.png' % str(id).zfill(5), part_full_mask)


        # save obj
        if opt.save_local_other:
            for img_id in range(opt.num_img):
                logger.save_obj('%s.obj' % img_id,
                                pre_dict['vertices'][img_id],
                                pre_dict['faces'][img_id],
                                img_id=img_id)

        # add mesh
        # logger.add_mesh('mesh_'+str(id),
        #                  pre_dict['vertices'].detach().cpu().numpy()[np.newaxis,:],
        #                  pre_dict['faces'][np.newaxis,:])


def save(opt, it_id, loss_dict, pre_dict):
    if it_id % opt.submit_scalar_iter == 0 or \
            it_id % opt.submit_other_iter == 0:

        if opt.save_method == 'thread':

            global g_lock, g_save_list

            g_lock.acquire()
            g_save_list.append({
                'id': it_id,
                'loss_dict': loss_dict,
                'pre_dict': pre_dict
            })
            g_lock.release()

        elif opt.save_method == 'process':
            if it_id % opt.submit_scalar_iter == 0 and \
                    it_id % opt.submit_other_iter == 0:
                p = Process(target=submit_process,
                            args=(opt, it_id, loss_dict, pre_dict,))
                p.start()
                # p.join()

        else:
            submit(opt, opt.pyrender, it_id, loss_dict, pre_dict)


def post_process(opt):
    if opt.save_method == 'thread':
        global g_save_thread_on
        g_save_thread_on = False