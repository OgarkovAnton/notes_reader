import cv2
import numpy as np


def to_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_val = np.max(image)
    return cv2.adaptiveThreshold(
        gray,
        max_val,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def invert(image):
    return np.abs(image * -1 + 255).astype("uint8")
