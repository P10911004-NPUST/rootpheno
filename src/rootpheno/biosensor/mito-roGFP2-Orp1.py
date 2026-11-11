
import numpy as np

from utils import *

def mito_roGFP2_Orp1(e405: np.ndarray, e488: np.ndarray):
    if e405.ndim != 2 or e488.ndim != 2:
        raise ValueError("Accept only 2D array.")
    unsaturated_mask = np.subtract(e405, e488) != 0
    e405 = e405 * unsaturated_mask
    e488 = e488 * unsaturated_mask
    e405, e405_mask = extract_root_region(e405)
    e488, e488_mask = extract_root_region(e488)
    protein_exist_mask = np.bitwise_or(e405_mask, e488_mask)
    background_mask = np.bitwise_invert(protein_exist_mask)
    e405 = np.float64(e405)
    e488 = np.float64(e488)
    e488 = e488 * (e405.max() / e488.max())
    e405 = e405 ** 2
    e488 = e488 ** 2
    ro = np.divide(e405 - e488, e405 + e488, out = np.zeros_like(e488), where = e405 != e488)
    lut = gray_to_lut(ro)
    return ro, lut

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Testing
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if "__name__" == "__main__":
    import czifile
    from PIL import Image
    with czifile.CziFile("Z:/JKLAI_QNAP/Gigi/mitoOrp1-roGFP2/orp-roGFP ritf1-4-1.czi") as czi:
        scene = czi.scenes()
        arr = scene.asarray()
        e405 = np.take(arr, 0, 0)
        e488 = np.take(arr, 1, 0)
        ro, lut = mito_roGFP2_Orp1(e405, e488)
        Image.fromarray(lut).show()
