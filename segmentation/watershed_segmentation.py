import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

from utils import data_utils

import numpy as np
from PIL import Image
from multiprocessing import Pool

import matplotlib.pyplot as plt
import imageio
import cv2
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, erosion, dilation, area_closing
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from pathlib import Path
from skimage import exposure, measure
from scipy.ndimage import binary_fill_holes

def segment(image, resize=None, old_intensity_range=None):
    image = image.astype(np.float64)

    if resize:
        original_size = image.shape
        image = data_utils.resize(image, (resize[1], resize[0]))

    if old_intensity_range:
        image = data_utils.normalize(image, old_intensity_range, (0.0, 1.0)) # normalize image

    image = cv2.bilateralFilter(image.astype(np.float32),28,0.2,0.2)

    for i in range(1):
        for j in range(5):
            image = dilation(image)
        image = area_closing(image, area_threshold=500)
        for j in range(5):
            image = erosion(image)

    # apply threshold
    thresh = max(threshold_otsu(image), 0.05)
    bw = closing(image > thresh, square(5))

    # label image regions
    label_image = label(bw)

    total_labs = 0
    blob_i = 1
    final_blobs = np.zeros_like(image)
    for i, region in enumerate(regionprops(label_image)):
        if 10000 > region.area > 50: # take regions with large enough areas
            blob_mask = (label_image == i+1)

            # Separate objects in image by generating markers as local maxima of the distance to the background and watershed
            distance = ndi.distance_transform_edt(blob_mask)
            coords = peak_local_max(distance, footprint=np.ones((9, 9)), labels=blob_mask, min_distance=10)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            sub_blob_labels = watershed(-distance, markers, mask=blob_mask)
            total_labs += len([x for x in regionprops(sub_blob_labels)])
            for sub_blob in regionprops(sub_blob_labels):
                if sub_blob.area >= 50: # take regions with large enough areas
                    for l in np.unique(sub_blob_labels)[1:]:
                        if np.unique(final_blobs[sub_blob_labels==l]) == [0]:
                            final_blobs[sub_blob_labels==l] = blob_i
                            blob_i += 1

    if resize:
        final_blobs = data_utils.resize_mask(final_blobs, (original_size[1], original_size[0]))

    return final_blobs

def segment_and_save(args):
    tif_path, seg_path, channel, resize_shape, intensity_range = args
    image = imageio.mimread(tif_path)[channel] # read image
    seg = segment(image, resize=resize_shape, old_intensity_range=intensity_range)
    seg_img = Image.fromarray(seg.astype(np.uint8), 'L')
    seg_img.save(seg_path)

def segment_dir(tif_dir, seg_dir, threads, channel=1, resize_shape=None, normalize=True):
    os.makedirs(seg_dir, exist_ok=True)

    tif_paths = data_utils.get_paths(tif_dir, extension='.tif', recursive=True)
    seg_paths = [os.path.join(seg_dir, Path(tif_path).stem + '.png') for tif_path in tif_paths]

    if normalize:
        ranges = data_utils.find_folder_range(tif_paths, channels=[channel])
        file_ranges = [ranges[os.path.dirname(tif_path)][channel] if normalize else None for tif_path in tif_paths]

    filewise_arguments = [(tif_path, seg_path, channel, resize_shape, file_range) for tif_path, seg_path, file_range in zip(tif_paths, seg_paths, file_ranges)]
    filewise_arguments = [args for args in filewise_arguments if not os.path.exists(args[1])]

    with Pool(threads) as p:
        p.map(segment_and_save, filewise_arguments)


if __name__ == "__main__":
    tif_dir = "/mnt/DATA1/anton/data/high_res_subset_from_Felix/B04/"
    seg_dir = "/mnt/DATA1/anton/data/high_res_subset_from_Felix/B04_watershed"
    
    # Channel starts counting the channels from 0
    segment_dir(tif_dir, seg_dir, threads=40, channel=1, resize_shape=(1040, 1392))

# strain
# -- day
# -----experiment1
# -----experiment2
# --day2