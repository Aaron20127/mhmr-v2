
import os
import sys
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, abspath + '/../../')

import cv2
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt

from common.debug import draw_coco_mask, draw_kp2d, draw_bbox


def show_instances():
    dataDir='D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2017'
    dataType='val2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)

    img_dir = '{}/{}'.format(dataDir, dataType)

    # initialize COCO api for instance annotations
    coco = coco.COCO(annFile)
    image_ids = coco.getImgIds()

    # load img
    img = coco.loadImgs(image_ids[0])[0]
    img_path = os.path.join(img_dir, img['file_name'])
    I = cv2.imread(img_path)

    # load and display instance annotations
    idxs = coco.getAnnIds(imgIds=[image_ids[0]], catIds=1, iscrowd=None)
    anns = coco.loadAnns(ids=idxs)

    # coco.showAnns(anns)
    np.array(ann['keypoints']).reshape(-1, 3)
    I_instance = draw_mask(I, anns[0]['segmentation'][0])

    # draw bbox
    cv2.namedWindow('I_instance', 0)
    cv2.imshow('I_instance', I_instance)
    cv2.waitKey(0)


def show_person_keypoints():
    dataDir = 'D:/paper/human_body_reconstruction/datasets/human_reconstruction/coco/coco2017'
    dataType = 'val2017'
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    img_dir = '{}/{}'.format(dataDir, dataType)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)
    image_ids = coco.getImgIds()

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


if __name__ == '__main__':
    show_person_keypoints()