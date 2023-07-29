import cv2
import numpy as np
from ..utils.hamphel_filter import hampel


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def scale_image(image: np.array) -> np.array:
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    contours = list(filter(lambda c: len(c) > 2, contours))
    boxes = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if h > 100 or w > 100:
            box = cv2.minAreaRect(i)
            boxes.append(box)
    angles = np.array([*map(lambda b: b[2], boxes)], dtype="float32")
    angle = np.mean(hampel(angles))
    rotated = rotate(image, angle)
    return rotated
