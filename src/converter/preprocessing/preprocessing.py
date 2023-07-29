import cv2
import numpy as np

from binarize import to_binary
from rotate import scale_image


def process_image(path):
    img = cv2.imread(path)
    binarized = to_binary(np.real(img))
    return scale_image(binarized)
