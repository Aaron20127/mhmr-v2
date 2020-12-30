import cv2
import numpy as np
import pycocotools.mask as maskUtils

def draw_kp2d(img, kp2d, draw_conf=False, draw_num=False, radius=8, color=(255,0,0)):
    for j in range(len(kp2d)):
        if ((kp2d.shape[1] == 3) and kp2d[j, 2] > 0):
            p = (int(kp2d[j, 0]), int(kp2d[j, 1]))
            conf = kp2d[j, 2]
            cv2.circle(img, p, radius, color, -1)

            if draw_conf:
                cv2.putText(img, "%.1f" % conf, (p[0], p[1]+30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, 4)

            if draw_num:
                cv2.putText(img, "%d" % j, (p[0], p[1]+30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2, 4)

        elif kp2d.shape[1] == 2:
            p = (int(kp2d[j, 0]), int(kp2d[j, 1]))
            cv2.circle(img, p, radius, color, -1)

            if draw_num:
                cv2.putText(img, "%d" % j, (p[0], p[1]+30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2, 4)

    return img


def draw_outline_mask(src, pts):
    """
    @param pts(narray int Nx1x2): coco segmentation outline points
    """
    mask = np.zeros_like(src)
    cv2.fillPoly(mask, [pts], (0, 255, 0))

    mask_src = (mask > 0) * src
    mask_img = cv2.addWeighted(mask_src, 0.6, mask, 0.4, 0) # transparency

    dst = mask_img + (mask == 0) * src

    return dst


def draw_mask(src, mask, color=(0, 255, 0)):
    """
    @param mask(narray uint8 hxwx1 or hxwx3): mask
    """
    color_mask = np.zeros_like(src)
    color_mask[:, :] = color
    color_mask = color_mask * (mask > 0)

    mask_src = (mask > 0) * src
    mask_img = cv2.addWeighted(mask_src, 0.6, color_mask, 0.4, 0) # transparency

    dst = mask_img + (mask == 0) * src

    return dst


def draw_bbox(src, bbox):
    dst = cv2.rectangle(src, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
    return  dst



####################### coco ############################
def draw_coco_mask(src, ann):
    if type(ann['segmentation']) == list:
        # polygon
        for seg in ann['segmentation']:
            outline_pts = np.array(seg).reshape((int(len(seg) / 2), 1, 2)).astype(np.int32)
            src = draw_outline_mask(src, outline_pts)
        return src
    else:
        # mask
        if type(ann['segmentation']['counts']) == list:
            rle = maskUtils.frPyObjects([ann['segmentation']], src.shape[0], src.shape[1])
        else:
            rle = [ann['segmentation']]
        mask = maskUtils.decode(rle)
        dst = draw_mask(src, mask)

        return dst



########################### smpl #########################
def add_blend_smpl(render_img, mask, img_raw):
    new_mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    color = render_img * new_mask + img_raw * (1 - new_mask)

    return color.astype(np.uint8)