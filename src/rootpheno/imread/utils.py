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


def czi_standard_dims(collapse: bool = True) -> list[str] | str:
    ''' CZI file dimensions

        Canonical image dimensions: X, Y, C, Z, T, R, S, I, B, M, H

        X: Pixel index / offset in the X direction. Used for tiled images. Image width.
        Y: Pixel index / offset in the Y direction. Used for tiled images. Image height.
        C: Channel in a Multi-Channel data set
        Z: Slice index (Z-direction)
        T: Time point in a sequentially acquired series of data
        R: Rotation -- used in acquisiton modes where the data is recorded from various angles.
        S: Scene -- for clustering items in X / Y direction.
        I: Illumination -- illumination direction index (e.g. from left=0, from right=1)
        B: Block index in segmented experiments.
        M: Mosaic tile index
        H: Phase index -- for specific acquisition methods.

        Reference: https://www.physics.hmc.edu/~gerbode/wppriv/wp-content/uploads/2012/06/DS_ZISRAW-FileFormat_Rel_ZEN2011.pdf
    '''
    if collapse:
        return "XYCZTRSIBMH"
    else:
        return ['X', 'Y', 'C', 'Z', 'T', 'R', 'S', 'I', 'B', 'M', 'H']


def tiff_standard_dims(collapse: bool = True) -> list[str] | str:
    ''' TIFF file dimensions

        Canonical image dimensions: X, Y, C, Z

        X: Pixel index / offset in the X direction. Used for tiled images. Image width.
        Y: Pixel index / offset in the Y direction. Used for tiled images. Image height.
        C: Channel in a Multi-Channel data set
        Z: Slice index (Z-direction). For multi-frames TIFF file.
    '''
    dims = ['X', 'Y', 'C', 'Z']
    return "".join(dims) if collapse else dims


def standardize_dims(
    arr: np.ndarray, 
    from_axes: str, 
    to_axes: str = czi_standard_dims()
) -> np.ndarray:
    from_axes = from_axes.upper()
    to_axes = to_axes.upper()
    if len(from_axes) != arr.ndim:
         raise ValueError(f"Number of axes ({len(from_axes)}) does not match array dimensions ({arr.ndim}).")
    if len(set(from_axes)) != len(from_axes):
        raise ValueError("Duplicate axis labels are not allowed.")
    if not set(from_axes).issubset(set(to_axes)):
        raise ValueError(f"Unexpected axes: {set(from_axes) - set(to_axes)}")
    existing_axes = [x for x in to_axes if x in from_axes]
    permutation = [from_axes.index(x) for x in existing_axes]
    arr_out = np.transpose(arr, permutation)
    for i, x in enumerate(to_axes):
        if x not in existing_axes:
            arr_out = np.expand_dims(arr_out, axis=i)
    return arr_out


def meter_convert_to_unit(value: float, to_unit: str | int | None = None) -> np.float64: 
    if to_unit is None:
        return value
    to_unit = str(to_unit)
    convert = {
        "m": 1, "meter": 1, "metre": 1,
        "3": 100, "cm": 100, "centi": 100, "centimeter": 100, "centimetre": 100,
        "mm": 1000, "milli": 1000, "millimeter": 1000, "millimetre": 1000,
        "2": 39.3701, "i": 39.3701, "inch": 39.3701, "inches": 39.3701,
        "1": 1e6, "µ": 1e6,"µm": 1e6, "u": 1e6,
        "um": 1e6, "micro": 1e6, "micron": 1e6, "micrometer": 1e6, "micrometre": 1e6,
        "nm": 1e9, "nano": 1e9, "nanometer": 1e9
    }
    ret = value * convert.get(to_unit) if to_unit in convert.keys() else None
    return np.float64(ret)


def unit_convert_to_micron(value: float, from_unit: str | int | None = None) -> np.float64: 
    if from_unit is None:
        return value
    from_unit = str(from_unit)
    convert = {
        "m": 1e6, "meter": 1e6, "metre": 1e6,
        "3": 1e4, "cm": 1e4, "centi": 1e4, "centimeter": 1e4, "centimetre": 1e4,
        "mm": 1000, "milli": 1000, "millimeter": 1000, "millimetre": 1000,
        "2": 25400 , "i": 25400, "inch": 25400, "inches": 25400,
        "1": 1, "µ": 1,"µm": 1, "u": 1, "um": 1, "micro": 1, "micron": 1, "micrometer": 1, "micrometre": 1,
        "nm": 1e-3, "nano": 1e-3, "nanometer": 1e-3
    }
    ret = value * convert.get(from_unit) if from_unit in convert.keys() else None
    return np.float64(ret)