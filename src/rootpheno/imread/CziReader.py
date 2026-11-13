import czifile
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

from utils import *


class CziReader:
    ''' Read CZI file and tidy up arbitrary dimensions canonically to "XYCZTRSIBMH" 
    
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
    def __init__(self, img_path: str | Path):
        self.img_path = Path(img_path).expanduser().resolve()
        self.dirname = self.img_path.parent.as_posix()
        self.basename = self.img_path.name
        self.filename = self.img_path.stem
        self.suffix = self.img_path.suffix

        if not self.img_path.is_file():
            raise FileNotFoundError(f"{self.img_path} is not a file.")

        if not self.img_path.suffix.lower() == ".czi":
            raise ValueError(f"{self.img_path} is not a CZI file.")

        with czifile.CziFile(self.img_path.as_posix(), squeeze=False) as czi:
            scene = czi.scenes()
            self.metadata = czi.metadata(asdict=True)["ImageDocument"]["Metadata"]
            self.distance = self.metadata["Scaling"]["Items"]["Distance"]
            self.arr = standardize_dims(scene.asarray(), "".join(scene.dims))
            self.dims = czi_standard_dims(collapse=False)
            self.sizes = dict(zip(self.dims, self.arr.shape))
            # self.mpp = scene.mpp  # micrometer per pixel

    @property
    def mpp(self) -> dict | None:
        ''' Micrometer per pixel (um / pixel)

            The default unit of self.distance.get("Value") is meter
        '''
        unit_convert = meter_convert_to_unit()

        ret = {}
        for d in self.distance:
            default_val = d.get("Value")
            default_unit = d.get("DefaultUnitFormat")
            ret[d.get("Id")] = meter_convert_to_unit(default_val, default_unit)
            # ret[d.get("Id")] = None
            # default_unit = d.get("DefaultUnitFormat")
            # if default_unit is not None:
            #     conversion = unit_convert.get(default_unit)
            # if conversion is not None:
            #     converted_value = d.get("Value") * conversion
            # if converted_value is not None:
            #     ret[d.get("Id")] = converted_value
        return ret

    @property
    def tiff_info(self) -> ImageFileDirectory_v2:
        ''' TIFF info

            | Code |     Description
            |======|===================================================================
            | 256  | Image width in pixels
            | 257  | Image height in pixels
            | 259  | Compression level [1 (uncompressed) - 32773 (high)]
            | 262  | Color space (1=[grayscale, black is zero], 2=[RGB])
            | 282  | Resolution of X-axis (pixels/unit), default unit is 3 (centimeter)
            | 283  | Resolution of Y-axis (pixels/unit), default unit is 3 (centimeter)
            | 296  | Resolution unit: 1=none, 2=inch, 3=centimeter
        '''
        is_RGB = self.try_extract_RGB() is not None
        tiff_info = ImageFileDirectory_v2()
        tiff_info[256] = self.sizes.get("X")         # ImageWidth (pixels)
        tiff_info[257] = self.sizes.get("Y")         # ImageHeight (pixels)
        tiff_info[259] = 1                           # Compression (1 - 32773)
        tiff_info[262] = 2 if is_RGB else 1          # PhotometricInterpretation
        tiff_info[282] = 10_000 / self.mpp.get("X")  # XResolution (pixels/cm)
        tiff_info[283] = 10_000 / self.mpp.get("Y")  # YResolution (pixels/cm)
        tiff_info[296] = 3                           # ResolutionUnit (cm)
        return tiff_info

    def try_extract_GRAY(self) -> np.ndarray | None:
        XY_valid = (self.sizes.get("X") > 1) and (self.sizes.get("Y") > 1)
        XY_only =  all(self.sizes.get(i) == 1 for i in "CZTRSIBMH")
        if XY_valid and XY_only:
            return self.arr[:, :, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            return None

    def try_extract_RGB(self) -> np.ndarray | None:
        XY_valid = (self.sizes.get("X") > 1) and (self.sizes.get("Y") > 1)
        C_valid = self.sizes.get("C") == 3
        S_valid = self.sizes.get("S") == 3
        C_only =  all(self.sizes.get(i) == 1 for i in "ZTRSIBMH")
        S_only = all(self.sizes.get(i) == 1 for i in "CZTRIBMH")
        if XY_valid and C_valid and C_only:
            return self.arr[:, :, :, 0, 0, 0, 0, 0, 0, 0, 0]
        elif XY_valid and S_valid and S_only:
            return self.arr[:, :, 0, 0, 0, 0, :, 0, 0, 0, 0]
        else:
            return None

    def try_extract_Zstack(self) -> np.ndarray | None:
        Z_valid = self.sizes.get("Z") > 1
        if Z_valid:
            return self.arr[:, :, 0, :, 0, 0, 0, 0, 0, 0, 0]
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
    axio     = Path("../../../tests/img/NBT/ath_nbt-reddish.czi").resolve()
    confocal = Path("../../../tests/img/BES-H2O2-Ac.czi").resolve()
    img_1 = CziReader(axio)
    img_2 = CziReader(confocal)
    print(img_1.sizes)
    print(img_2.sizes)
    img_1.try_extract_RGB()
    img_2.try_extract_Zstack()

