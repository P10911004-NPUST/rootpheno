
from __future__ import annotations

import copy
import czifile

import numpy as np

from pathlib import Path


class CziReader:
    """Read scientific microscopy images from CZI files.

    Canonical image dimensions: X, Y, C, F, S, Z, T, H

    X: Image width
    Y: Image height
    C: Channel
    F: Frame
    S: Scene
    Z: Z-stack
    T: Time
    H: Additional H dimension (as defined by CZI)
    """

    _DIMS = ("X", "Y", "C", "F", "S", "Z", "T", "H")
    _SCENE_DIMS = ("X", "Y", "C", "F", "Z", "T", "H")

    def __init__(self, img_path: str | Path):
        self.img_path = Path(img_path).expanduser().resolve()

        if self.img_path.suffix.lower() != ".czi":
            raise ValueError(f"Not a CZI file: {self.img_path}")

        if not self.img_path.is_file():
            raise FileNotFoundError(self.img_path)

        self._load_img()

    @staticmethod
    def _normalize(
        arr: np.ndarray,
        axes: tuple[str, ...],
    ) -> np.ndarray:
        """Normalize an image to X, Y, C, F, Z, T, H."""

        axes = tuple(axes)

        if len(axes) != arr.ndim:
            raise ValueError(
                f"Axis information {axes} does not match "
                f"array dimensions {arr.shape}."
            )

        if len(set(axes)) != len(axes):
            raise ValueError(f"Duplicate axis labels: {axes}")

        if not {"X", "Y"}.issubset(axes):
            raise ValueError(f"Missing X or Y dimensions: {axes}")

        unsupported = set(axes) - set(self._SCENE_DIMS)

        if unsupported:
            raise NotImplementedError(
                f"Unsupported CZI dimensions: {unsupported}. "
                f"Original axes: {axes}"
            )

        # Add missing dimensions.
        for dim in self._SCENE_DIMS:
            if dim not in axes:
                arr = np.expand_dims(arr, axis=-1)
                axes += (dim,)

        # Reorder dimensions.
        order = [axes.index(dim) for dim in self._SCENE_DIMS]

        return np.transpose(arr, order)

    def _load_img(self) -> None:
        """Load all CZI scenes and metadata."""

        with czifile.CziFile(self.img_path, squeeze=False) as czi:

            self.img_type = "CZI"

            # Preserve original CZI metadata.
            self.metadata = czi.metadata()

            scenes = czi.scenes()

            if not scenes:
                raise ValueError("No image scenes found in CZI file.")

            scene_arrays = []
            self.original_axes = []
            self.original_shapes = []
            self.channel_names = []

            for scene in scenes:
                axes = tuple(scene.dims)
                arr = scene.asarray()

                self.original_axes.append(axes)
                self.original_shapes.append(arr.shape)

                # Preserve channel information when available.
                self.channel_names.append(
                    copy.deepcopy(getattr(scene, "channels", None))
                )

                arr = self._normalize(arr, axes)
                scene_arrays.append(arr)

        # Stack scenes along S.
        try:
            self.arr = np.stack(scene_arrays, axis=4)
        except ValueError as e:
            raise ValueError(
                "CZI scenes have inconsistent dimensions. "
                "They cannot be stacked into a single array."
            ) from e

        # X × Y × C × F × S × Z × T × H
        self.dims = self._DIMS
        self.sizes = dict(zip(self.dims, self.arr.shape))

        (
            self.width,
            self.height,
            self.channel,
            self.frames,
            self.scenes,
            self.n_z,
            self.n_time,
            self.n_h,
        ) = self.arr.shape

        # TIFF export information.
        self.tiff_info = {
            "axes": self.dims,
            "shape": self.arr.shape,
            "dtype": str(self.arr.dtype),
            "channel_names": self.channel_names,
            "metadata": self.metadata,
        }

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

    def get_frame(
        self,
        index: int = 0,
        scene: int = 0,
        z: int = 0,
        t: int = 0,
        h: int = 0,
    ) -> np.ndarray:
        """Return one frame in X, Y, C format."""

        return self.arr[:, :, :, index, scene, z, t, h]

    def __repr__(self) -> str:
        return (
            f"CziReader("
            f"path='{self.img_path.as_posix()}', "
            f"sizes={self.sizes}, "
            f"dtype={self.arr.dtype})"
        )


img = CziReader("../../../tests/img/BES-H2O2-Ac.czi")
# img = czifile.CziFile("./tests/img/BES-H2O2-Ac.czi", squeeze=False)


