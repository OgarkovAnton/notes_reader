import numpy as np


def hampel(vals_orig: np.array) -> np.array:
    vals = vals_orig.copy()
    difference = np.abs(np.median(vals.astype(int)) - vals)
    median_abs_deviation = np.median(difference)
    threshold = 3 * median_abs_deviation
    in_idx = difference < threshold
    return vals[in_idx]
