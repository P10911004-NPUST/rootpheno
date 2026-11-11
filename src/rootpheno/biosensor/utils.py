

import numpy as np

from skimage.morphology import disk as kernel_disk
from skimage.filters import threshold_otsu
from skimage.filters.rank import median as median_filter


def scale_min_max(arr: np.ndarray, value_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    vmin = np.min(arr)
    vmax = np.max(arr)
    if vmin == vmax:
        return np.full(arr.shape, value_range[0], dtype=np.float64)
    arr = (arr - vmin) / (vmax - vmin)
    arr = arr * (value_range[1] - value_range[0]) + value_range[0]
    return arr


def convert_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = scale_min_max(arr, (0.0, 255.0))
    return np.round(arr).astype(np.uint8)


def extract_root_region(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    # Background must be black, target must be white
    kernel = kernel_disk(kernel_size)
    mask = convert_to_uint8(arr)
    mask = median_filter(mask, kernel)
    mask = mask > threshold_otsu(mask)
    root_arr = np.multiply(arr, mask)
    return root_arr, mask


def percentile_filter(arr: np.ndarray, perc: tuple = (0, 99)) -> np.ndarray:
    # Discard saturated pixels, especially the root cap
    raw_arr = arr
    arr = np.double(arr)
    bg_val = arr.min()
    lower_threshold = np.percentile(arr, perc[0])
    upper_threshold = np.percentile(arr, perc[1])
    ratio_of_unique_val = len(np.unique(arr)) / len(arr)
    if ratio_of_unique_val > 0.3:
        arr[raw_arr < lower_threshold] = bg_val
        arr[raw_arr > upper_threshold] = bg_val
    return arr


def gray_to_lut(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != "np.uint8":
        arr = scale_min_max(arr) * 255
    lut = np.zeros((256, 3), dtype=np.uint8)
    # The lower part (blue -> white)
    lut[:128, 0] = np.linspace(0, 255, 128) # R
    lut[:128, 1] = np.linspace(0, 255, 128) # G
    lut[:128, 2] = 255                      # B
    # The upper part (white -> red)
    lut[128:, 0] = 255                      # R
    lut[128:, 1] = np.linspace(255, 0, 128) # G
    lut[128:, 2] = np.linspace(255, 0, 128) # B
    arr = convert_to_uint8(arr)
    return lut[arr]