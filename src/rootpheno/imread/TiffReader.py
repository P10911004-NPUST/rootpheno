from __future__ import annotations

import copy
import numpy as np

from PIL import Image
from pathlib import Path


class TiffReader:
    """Read TIFF images with support for multiple frames."""

    def __init__(self, img_path: str | Path):
        self.img_path = Path(img_path).expanduser().resolve()

        if self.img_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Not a TIFF file: {self.img_path}")

        if not self.img_path.is_file():
            raise FileNotFoundError(self.img_path)

        self._load_img()


    def _load_img(self) -> None:
        """Load all TIFF frames and metadata.

        Image dimensions: X, Y, C, F
            X: Image width
            Y: Image height
            C: Image channels
            F: Image frames
        """
        with Image.open(self.img_path) as img:
            if img.format != "TIFF":
                raise ValueError(f"Invalid TIFF file: {self.img_path}")

            self.img_type = img.format
            self.frames = img.frames
            self.tiff_info = copy.deepcopy(img.tag_v2)

            frames = []

            for i in range(self.frames):
                img.seek(i)
                arr = np.asarray(img)

                # Convert to X × Y × C.
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                elif arr.ndim != 3:
                    raise ValueError(f"Unsupported frame dimensions: {arr.shape}")

                frames.append(arr)

            try:
                # Stack frames along the last axis.
                self.arr = np.stack(frames, axis=-1)
            except ValueError as e:
                raise ValueError("TIFF frames have inconsistent dimensions or channels.") from e

        # X × Y × C × F
        self.dims = ("X", "Y", "C", "F")
        self.sizes = dict(zip(self.dims, self.arr.shape))
        self.width, self.height, self.channel, self.frames = self.arr.shape


    @property
    def suffix(self) -> str:
        return self.img_path.suffix
    
    @property
    def dirname(self) -> str:
        return self.img_path.parent.as_posix()

    @property
    def basename(self) -> str:
        return self.img_path.name

    @property
    def filename(self) -> str:
        return self.img_path.stem

    def get_frame(self, index: int) -> np.ndarray:
        """Return a single frame in C, Y, X format."""
        return self.arr[index]

    def __repr__(self) -> str:
        return f"TiffReader(path='{self.img_path.as_posix()}', sizes={self.sizes}, dtype={self.arr.dtype})"
