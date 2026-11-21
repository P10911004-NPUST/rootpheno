


import numpy as np


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


def estimate_kernel_size(arr: np.ndarray) -> int:
    h = arr.shape[0]
    w = arr.shape[1]
    kernel_size = (np.sum(arr > 0) / (h * w)) * 100 * 0.66
    kernel_size = np.ceil(np.sqrt(kernel_size / 3.14159))
    # coerce to odd value
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    kernel_size = 3 if kernel_size < 3 else kernel_size
    return int(kernel_size)


def output_dict():
    return {
        "img_dirname": None,
        "img_basename": None,
        "nbt_area": -999,
        "nbt_mean": -999,
        "nbt_total": -999,
        "note": "ok"
    }


    