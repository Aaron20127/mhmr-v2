

import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../../')

import h5py
from tqdm import tqdm
import math
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import numpy as np

from common.debug import draw_coco_mask, draw_kp2d, draw_bbox, draw_mask
from common.utils import iou


def show_person_keypoints():
    dataDir = '/opt/LIWEI/datasets/coco/coco2017'
    dataType = 'train2017'
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    img_dir = '{}/{}'.format(dataDir, dataType)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)
    image_ids = coco.getImgIds(catIds=[1])

    for img_id in image_ids:
        # load img
        img = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img['file_name'])
        I = cv2.imread(img_path)

        # load and display instance annotations
        idxs = coco.getAnnIds(imgIds=[img_id], catIds=1, iscrowd=0)
        anns = coco.loadAnns(ids=idxs)

        for ann in anns:
            # instance
            I = draw_coco_mask(I, ann)

            # kp2ds
            kp2d = np.array(ann['keypoints']).reshape(-1, 3)
            I = draw_kp2d(I, kp2d, radius=3)

            # bbox
            bbox = ann['bbox']
            bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2],
                             bbox[1] + bbox[3]], dtype=np.int32)
            I = draw_bbox(I, bbox)


        cv2.namedWindow('I', 0)
        cv2.imshow('I', I)
        cv2.waitKey(0)


def decode_coco_mask(src, ann):
    if type(ann['segmentation']) == list:
        # polygon
        mask = np.zeros_like(src)
        for seg in ann['segmentation']:
            outline_pts = np.array(seg).reshape((int(len(seg) / 2), 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [outline_pts], (0, 255, 0))
        mask = (mask[:,:,1] > 0).astype(np.uint8)
    else:
        # mask
        if type(ann['segmentation']['counts']) == list:
            rle = maskUtils.frPyObjects([ann['segmentation']], src.shape[0], src.shape[1])
        else:
            rle = [ann['segmentation']]
        mask = maskUtils.decode(rle).astype(np.uint8)

    mask = mask.reshape(mask.shape[0], mask.shape[1], 1) * 255
    return np.concatenate((mask, mask, mask), axis=2)


def pack_data(annA, annB, img):
    boxA = annA['bbox']
    boxB = annB['bbox']

    crop_box = [
        int(min(boxA[0], boxB[0])),
        int(min(boxA[1], boxB[1])),
        math.ceil(max(boxA[2], boxB[2])),
        math.ceil(max(boxA[3], boxB[3]))
    ]

    crop_img = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

    origin = np.array([[crop_box[0],crop_box[1],0]])
    kp2d = np.stack((annA['kp2d'] - origin,
                     annB['kp2d'] - origin))

    mask = np.stack((annA['mask'][crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]],
                     annB['mask'][crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]))

    origin = np.array([crop_box[0],crop_box[1],crop_box[0],crop_box[1]])
    bbox = np.stack((boxA - origin,
                     boxB - origin))

    mask_full = ((((mask[0] / 255) + (mask[1] / 255)) > 0) * 255).astype(np.uint8)

    return {
        'img': crop_img,
        'kp2d': kp2d,
        'mask': mask,
        'bbox': bbox,
        'mask_full': mask_full
    }


def save_visual_label(data, visual_label_name):
    I = data['img']

    I = draw_mask(I, (data['mask'][0] / 255).astype(np.uint8), color=(0, 255, 255))
    I = draw_mask(I, (data['mask'][1] / 255).astype(np.uint8), color=(255, 0, 255))

    # kp2ds
    I = draw_kp2d(I, data['kp2d'][0], radius=3)
    I = draw_kp2d(I, data['kp2d'][1], radius=3)

    # bbox
    I = draw_bbox(I, data['bbox'][0])
    I = draw_bbox(I, data['bbox'][1])

    cv2.imwrite(visual_label_name, I)


def main(annotation_name, dataType='train2017', min_iou=0.1, min_area=4000, min_kp2d=6):
    dataDir = '/opt/LIWEI/datasets/coco/coco2017'
    # dataType = 'train2017'
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    full_img_dir = '{}/{}'.format(dataDir, dataType)

    # dataset dir
    save_dir = 'dataset'
    save_img_dir = os.path.join(save_dir, 'image')
    save_mask_a_dir = os.path.join(save_dir, 'mask_a')
    save_mask_b_dir = os.path.join(save_dir, 'mask_b')
    save_mask_full_dir = os.path.join(save_dir, 'mask_full')
    save_visual_label_dir = os.path.join(save_dir, 'visual_label')
    dst_file = os.path.join(save_dir, annotation_name)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_a_dir, exist_ok=True)
    os.makedirs(save_mask_b_dir, exist_ok=True)
    os.makedirs(save_mask_full_dir, exist_ok=True)
    os.makedirs(save_visual_label_dir, exist_ok=True)

    _kp2d = np.zeros((100000, 2, 17, 3))
    _bbox = np.zeros((100000, 2, 4))
    _img_name = []
    _mask_a_name = []
    _mask_b_name = []
    _mask_full_name = []

    # initialize COCO api for instance annotations
    coco = COCO(annFile)
    image_ids = coco.getImgIds(catIds=[1])

    save_id = 0
    for img_id in tqdm(image_ids):
        # load img
        img = coco.loadImgs(img_id)[0]
        img_path = os.path.join(full_img_dir, img['file_name'])
        I = cv2.imread(img_path)

        # load annotations
        idxs = coco.getAnnIds(imgIds=[img_id], catIds=1, iscrowd=0)
        anns = coco.loadAnns(ids=idxs)

        # get mask, kp2d, bbox
        anns_list = []
        for iter_id, ann in enumerate(anns):
            if 'segmentation' in ann:
                # instance
                mask = decode_coco_mask(I, ann)

                # kp2ds
                kp2d = np.array(ann['keypoints']).reshape(-1, 3)

                # bbox
                bbox = ann['bbox']
                bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2],
                                 bbox[1] + bbox[3]], dtype=np.int32)

                anns_list.append({
                    'mask': mask,
                    'kp2d': kp2d,
                    'bbox': bbox,
                    'num_keypoints': ann['num_keypoints'],
                    'area': ann['area']
                })

        # data_generater
        data_list = []
        len_anns = len(anns_list)
        for i in range(len_anns):
            for j in range(i+1, len_anns):
                annA = anns_list[i]
                annB = anns_list[j]
                if iou(annA['bbox'], annB['bbox']) > min_iou and\
                    annA['area'] > min_area and\
                    annB['area'] > min_area and\
                    annA['num_keypoints'] > min_kp2d and\
                    annB['num_keypoints'] > min_kp2d:

                    data_list.append(pack_data(annA, annB, I))

        # save data
        for i, data in enumerate(data_list):
            _kp2d[i] = data['kp2d']
            _bbox[i] = data['bbox']

            img_name = os.path.join(save_img_dir, 'img_%s.png' % str(save_id).zfill(8))
            mask_a_name = os.path.join(save_mask_a_dir, 'img_%s.png' % str(save_id).zfill(8))
            mask_b_name = os.path.join(save_mask_b_dir, 'img_%s.png' % str(save_id).zfill(8))
            mask_full_name = os.path.join(save_mask_full_dir, 'img_%s.png' % str(save_id).zfill(8))
            visual_label_name = os.path.join(save_visual_label_dir, 'img_%s.png' % str(save_id).zfill(8))

            _img_name.append(img_name)
            _mask_a_name.append(mask_a_name)
            _mask_b_name.append(mask_b_name)
            _mask_full_name.append(mask_b_name)

            cv2.imwrite(img_name, data['img'])
            cv2.imwrite(mask_a_name, data['mask'][0])
            cv2.imwrite(mask_b_name, data['mask'][1])
            cv2.imwrite(mask_full_name, data['mask_full'])

            save_visual_label(data, visual_label_name)

            save_id += 1


    dst_fp = h5py.File(dst_file, 'w')
    dst_fp.create_dataset('kp2d', data=_kp2d[:save_id])
    dst_fp.create_dataset('bbox', data=_bbox[:save_id])
    dst_fp.create_dataset('img_name', data=np.array(_img_name, dtype='S'))
    dst_fp.create_dataset('mask_a_name', data=np.array(_mask_a_name, dtype='S'))
    dst_fp.create_dataset('mask_b_name', data=np.array(_mask_b_name, dtype='S'))
    dst_fp.create_dataset('mask_full_name', data=np.array(_mask_full_name, dtype='S'))
    dst_fp.close()

        # cv2.namedWindow('I', 0)
        # cv2.imshow('I', I)
        # cv2.waitKey(0)


if __name__ == '__main__':
    main(annotation_name='train2017.h5',
                          dataType='train2017')