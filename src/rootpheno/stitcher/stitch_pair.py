
import cv2
import numpy as np

from rootpheno.imread.utils import convert_to_uint8


def stitch_pair(img1, img2, transform_mat: np.ndarray | None = None):
    """
    Stitch two grayscale images using SIFT, FLANN and RANSAC.

    The estimated transformation maps img1 coordinates to img2 coordinates.

    Inputs
    ------
    img1: the reference array to be stitch on
    img2: the array stitch onto the img1
    transform_mat: np.ndarray
        2x3 affine transformation matrix, used to transform the img1


    Returns
    -------
    stitched_img : np.ndarray
        2D-array of the stitched image
    transform_mat : np.ndarray
        2x3 affine transformation.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if transform_mat is None:
        # Feature detection.
        detector = cv2.SIFT_create(nfeatures=5000)

        kp1, des1 = detector.detectAndCompute(convert_to_uint8(img1), None)
        kp2, des2 = detector.detectAndCompute(convert_to_uint8(img2), None)

        if des1 is None or des2 is None:
            raise RuntimeError("Not enough SIFT features.")

        # FLANN matcher. algorithm = 1 -> FLANN_INDEX_KDTREE
        # index_params = dict(algorithm=1, trees=5)
        # search_params = dict(checks=200)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test.
        good_matches = [
            m for pair in matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.75 * n.distance
        ]

        if len(good_matches) < 4:
            raise RuntimeError(f"Insufficient matches: {len(good_matches)}")

        # Corresponding coordinates.
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Estimate transformation: img1 -> img2.
        transform_mat, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.995,
            refineIters=20
        )

        if transform_mat is None:
            raise RuntimeError("Transformation estimation failed.")

        inlier_count = int(inliers.sum())

        if inlier_count < 4:
            raise RuntimeError("Too few RANSAC inliers.")

    corners1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    corners2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ])

    transformed_corners = cv2.transform(corners1, transform_mat).reshape(-1, 2)
    all_corners = np.vstack([transformed_corners, corners2])

    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)

    canvas_width, canvas_height = (max_xy - min_xy)

    # Translate both images into the positive canvas.
    offset = np.array([
        [1, 0, -min_xy[0]],
        [0, 1, -min_xy[1]]
    ], dtype=np.float64)

    M_canvas = offset @ np.vstack([transform_mat, [0, 0, 1]])
    M_canvas = M_canvas[:2]

    # Warp first image.
    warped1 = cv2.warpAffine(
        img1,
        M_canvas,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR
    )

    # Place second image on the canvas.
    warped2 = cv2.warpAffine(
        img2,
        offset,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR
    )

    # Generate valid-pixel masks.
    mask1 = cv2.warpAffine(
        np.ones((h1, w1), dtype=np.uint8),
        M_canvas,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST
    )

    mask2 = cv2.warpAffine(
        np.ones((h2, w2), dtype=np.uint8),
        offset,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST
    )

    # Maximum intensity blending.
    stitched_img = np.zeros(
        (canvas_height, canvas_width),
        dtype=img1.dtype
    )

    stitched_img[mask1 > 0] = warped1[mask1 > 0]
    stitched_img[mask2 > 0] = warped2[mask2 > 0]

    # # In the overlapping region, pixels were substituted with the 
    # # maximum value between the img1 and img2.
    # overlap = (mask1 > 0) & (mask2 > 0)
    # only2 = (mask1 == 0) & (mask2 > 0)
    # stitched[overlap] = np.maximum(warped1[overlap], warped2[overlap])
    # stitched[only2] = warped2[only2]

    return stitched_img, transform_mat




#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Testing
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# if __name__ == "__main__":
#     from PIL import Image
#     from pathlib import Path
#     from rootpheno.imread.CziReader import CziReader
#     from rootpheno.imread.TiffReader import TiffReader

#     def imshow(arr):
#         Image.fromarray(convert_to_uint8(arr)).show()

#     input_folder = Path("./tests/img/RO-index/czi")
#     czi_list = [p for p in input_folder.iterdir()if p.suffix.lower() in (".czi", ".czi")]

#     input_folder = Path("./tests/img/RO-index/RO_LUT")
#     tiff_list = [p for p in input_folder.iterdir()if p.suffix.lower() in (".tif", ".tiff")]

#     images = []
#     mpp_x = []
#     mpp_y = []
#     stage_x_um = []
#     stage_y_um = []
#     for path in czi_list:
#         czi = CziReader(path)
#         img = np.moveaxis(np.take(czi.arr, 2, 2).squeeze(), 0, -1)
#         images.append(img)
#         mpp_x.append(czi.mpp.get("X"))
#         mpp_y.append(czi.mpp.get("Y"))
#         sp = czi.get_stage_position()
#         stage_x_um.append(sp.get("X"))
#         stage_y_um.append(sp.get("Y"))

#     warped_img, transform_mat = stitch_pair(images[0], images[1])
#     imshow(warped_img)
    
#     images = []
#     for path in tiff_list:
#         tiff = TiffReader(path)
#         img = np.moveaxis(np.mean(tiff.arr, 2).squeeze(), 0, -1)
#         images.append(img)

#     warped_img, transform_mat = stitch_pair(images[0], images[1], transform_mat)

#     imshow(warped_img)
        
        
