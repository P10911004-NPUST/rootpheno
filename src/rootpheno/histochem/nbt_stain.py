
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.filters.rank import median as median_filter
from skimage.filters.rank import mean as mean_filter
from skimage.morphology import disk, erosion, dilation, remove_small_objects
from skimage.measure import label, regionprops, find_contours

from utils import *


def nbt_stain(
    R:np.ndarray, 
    G: np.ndarray, 
    B:np.ndarray, 
    k: int = 3,      # how many groups for Otsu thresholding
    q: float = 50,   # Percentile (??%) for thresholding the multi-Otsu thresholds
    n: int = 1,      # Keep how many blobs
    iter: int = 3    # Iterations to perform for image filtering and smoothing
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.uint64, np.uint64, np.float64]:
    k = max(2, k)
    q = max(0, q)

    if not R.ndim == G.ndim == B.ndim == 2:
        raise ValueError("Accept only 2D array.")

    if not R.shape == G.shape == B.shape:
        raise ValueError("Shape of all input must be identical.")

    RGB = np.stack([R, G, B], axis=-1)
    if not R.dtype == G.dtype == B.dtype == np.uint8:
        RGB = convert_to_uint8(RGB)

    R = np.float64(R)
    G = np.float64(G)
    B = np.float64(B)
    GRAY = 0.2125 * R + 0.7154 * G + 0.0721 * B  # skimage.rgb2gray formula
    GRAY = np.invert(convert_to_uint8(GRAY))
    h, w = GRAY.shape

    # Excess blue for NBT measure, not simply converting to gray
    ExB = (R + G + B + B + B) / 5
    ExB = np.invert(convert_to_uint8(ExB))
    
    # Reduce image size to accelerate computation
    ExB_small = np.asarray(Image.fromarray(ExB).resize(size=(512, 512)))

    # Extract root region, this is the first step to roughly extract the NBT stained area
    root_region = ExB_small * (ExB_small >= threshold_otsu(ExB_small))

    # Eliminate noises such as the cell wall or other tissues that were mis-stained
    kernel_size = estimate_kernel_size(root_region)
    kernel = disk(kernel_size)
    root_region = median_filter(root_region, kernel)
    
    # Extract ROI (the NBT stained area) from the root region
    threshold = threshold_multiotsu(root_region, classes=k)
    ROI = root_region >= np.percentile(threshold, q)
    ROI = np.uint8(ROI * 255)
    for _ in range(iter):
        ROI = erosion(ROI, kernel)
        ROI = median_filter(ROI, kernel)
        ROI = dilation(ROI, kernel)

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Remove off-target blobs
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ROI = mean_filter(ROI, kernel) > 0
    blobs = regionprops(label(ROI))
    if blobs is not None:
        area = np.sort([i.area for i in blobs])
        keep = int(len(area) - n - 1) 
        # area = max(area) - 1  # keep only the largest
        ROI = remove_small_objects(ROI, max_size=area[keep])
    
    # Slightly enlarge the ROI to cover the NBT stained area more completely
    kernel2 = disk(np.ceil(np.sqrt(kernel_size)))
    ROI = dilation(ROI, kernel2)  # This is boolean 2D-array
    ROI = Image.fromarray(np.uint8(ROI)).resize(size=(w, h))
    ROI = np.asarray(ROI)  # This is uint8 2D-array, range from 0 to 1
    
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Output NBT measures
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ExB = ExB * ROI  # This is uint8 2D-array, used for NBT measures
    nbt_total = np.sum(ExB)
    nbt_area = np.sum(ROI)
    nbt_mean = 0 if nbt_area == 0 else nbt_total / nbt_area

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Draw contour
    # Visualize the NBT stained area by drawing contour on the original image
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    NBT = np.array(RGB)
    contours = find_contours(ROI, level=0)
    if len(contours) > 0:
        NBT = Image.fromarray(RGB)
        draw = ImageDraw.Draw(NBT)
        font_color = (255, 0, 0)  # RGB, red color
        font_size = np.ceil(np.sqrt((h * 0.035) * (w * 0.035)))
        font_settings = ImageFont.load_default(size=font_size)
        text = f"Avg: {np.round(nbt_mean, 2)} = {np.round(nbt_total/1_000_000, 2)} M / {nbt_area} pixels"
        for contour in contours:
            points_xy = [(c[1], c[0]) for c in contour]
            draw.line(points_xy, fill=font_color, width=int(np.ceil(font_size * 0.15)))
            draw.text(xy=(30, 10), text=text, fill=font_color, font=font_settings)
        NBT = np.asarray(NBT)
  
    return RGB, GRAY, NBT, nbt_total, nbt_area, nbt_mean


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Testing    
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
if "__name__" == "__main__":
    import numpy as np
    from PIL import Image
    img = Image.open("../../../tests/img/NBT/ath_nbt_01.tif")
    arr = np.asarray(img)
    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]
    RGB, GRAY, NBT, nbt_total, nbt_area, nbt_mean = nbt_stain(R, G, B)
    Image.fromarray(NBT).show()

