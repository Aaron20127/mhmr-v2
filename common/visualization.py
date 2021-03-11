import cv2
from PIL import Image
import numpy as np


def normalization(data):
    _range = np.max(data) - np.min(data)

    if _range <= 0:
        return data
    else:
        return (data - np.min(data)) / _range


def imshow_cv2_uint8(file_name, img, time=1):
    img_rgb = img
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    cv2.imshow(file_name, img_rgb)
    cv2.waitKey(time)


def imshow_cv2_not_uint8(file_name, img, time=1):
    img_rgb = (normalization(img) * 255).astype(np.uint8)
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    cv2.imshow(file_name, img_rgb)
    cv2.waitKey(time)


def imshow_pil_uint8(img):
    img_rgb = img
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    Image.fromarray(img_rgb).show()


def imshow_pil_not_uint8(img):
    img_rgb = (normalization(img) * 255).astype(np.uint8)
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    Image.fromarray(img_rgb).show()