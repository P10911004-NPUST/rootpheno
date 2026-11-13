
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

from utils import *


class TiffReader:
    ''' Read TIFF file and tidy up arbitrary dimensions canonically to "XYCZ" 
    
        Canonical image dimensions: X, Y, C, Z

        X: Pixel index / offset in the X direction. Used for tiled images. Image width.
        Y: Pixel index / offset in the Y direction. Used for tiled images. Image height.
        C: Channel in a Multi-Channel data set
        Z: Slice index (Z-direction). For multi-frames TIFF file.
    '''
    def __init__(self, img_path: str | Path):
        self.img_path = Path(img_path).expanduser().resolve()
        self.dirname = self.img_path.parent.as_posix()
        self.basename = self.img_path.name
        self.filename = self.img_path.stem
        self.suffix = self.img_path.suffix

        if not self.img_path.is_file():
            raise FileNotFoundError(f"{self.img_path} is not a file.")

        if not self.img_path.suffix.lower() in (".tif", ".tiff"):
            raise ValueError(f"{self.img_path} is not a TIFF file.")

        with Image.open(self.img_path.as_posix()) as tif:
            self.img = tif
            self.arr = np.asarray(self.img)        # (Y, X, ...)
            self.arr = np.moveaxis(self.arr, 0 ,1) # (X, Y, ...)
            is_GRAY = len(self.arr.shape) == 2
            is_RGB = len(self.arr.shape) == 3
            if self.img.n_frames == 1 and is_GRAY:
                orig_dims = "XY"
            elif self.img.n_frames == 1 and is_RGB:
                orig_dims = "XYC"
            else:
                orig_dims = "XYCZ"
            self.arr = standardize_dims(self.arr, orig_dims, tiff_standard_dims())
            self.dims = tiff_standard_dims(collapse=False)
            self.sizes = dict(zip(self.dims, self.arr.shape))

    @property
    def mpp(self) -> dict | None:
        tag = self.img.tag_v2
        if tag is not None:
            default_vx = 1 / tag.get(282)   # unit per pixel
            default_vy = 1 / tag.get(283)   # unit per pixel
            default_unit = tag.get(296)  # ResolutionUnit
            vx = unit_convert_to_micron(default_vx, default_unit)
            vy = unit_convert_to_micron(default_vy, default_unit)
            return {"X": vx, "Y": vy}
        else:
            return None

    @property
    def tiff_info(self) -> ImageFileDirectory_v2:
        return self.img.tag_v2

    def try_extract_GRAY(self) -> np.ndarray | None:
        XY_valid = (self.sizes.get("X") > 1) and (self.sizes.get("Y") > 1)
        XY_only =  all(self.sizes.get(i) == 1 for i in "CZ")
        return self.arr[:, :, 0, 0] if XY_valid and XY_only else None

    def try_extract_RGB(self) -> np.ndarray | None:
        XY_valid = (self.sizes.get("X") > 1) and (self.sizes.get("Y") > 1)
        C_valid = self.sizes.get("C") == 3
        C_only =  self.sizes.get("Z") == 1
        S_valid = self.sizes.get("Z") == 3
        S_only =  self.sizes.get("C") == 1
        if XY_valid and C_valid and C_only:
            return self.arr[:, :, :, 0]
        elif XY_valid and S_valid and S_only:
            return self.arr[:, :, 0, :]
        else:
            return None

    def show(self) -> None:
        GRAY = self.try_extract_GRAY()
        RGB = self.try_extract_RGB()
        if GRAY is not None:
            GRAY = convert_to_uint8(GRAY)
            GRAY = np.moveaxis(GRAY, 0, 1)
            Image.fromarray(np.squeeze(GRAY)).show()
        if RGB is not None:
            RGB = convert_to_uint8(RGB)
            RGB = np.moveaxis(RGB, 0, 1)
            Image.fromarray(np.squeeze(RGB)).show()


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Testing
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if "__name__" == "__main__":
    from pathlib import Path
    from PIL import Image
    ath = Path("../../../tests/img/NBT/ath_nbt_01.tif").resolve()
    rice  = Path("../../../tests/img/NBT/rice_nbt_01.tif").resolve()

    img = TiffReader(rice)
    print(img.sizes)
    img.try_extract_RGB()

