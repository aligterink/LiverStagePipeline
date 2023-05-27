import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# np.set_printoptions(threshold=sys.maxsize)

def get_segmentation(image, old_intensity_range=None, resize=None):
    image = image.astype(np.float64)

    if resize:
        original_size = image.shape
        image = data_utils.resize(image, (resize[1], resize[0]))

    if old_intensity_range:
        print(np.unique(image), image.dtype)
        image = data_utils.normalize(image, old_intensity_range, (0.0, 1.0)) # normalize image

    image = cv2.bilateralFilter(image.astype(np.float32),28,250,500)

    for i in range(1):
        for j in range(5):
            image = dilation(image)
        image = area_closing(image, area_threshold=500)
        for j in range(5):
            image = erosion(image)

    # apply threshold
    thresh = max(threshold_otsu(image), 0.1)
    bw = closing(image > thresh, square(5))

    # bw = ndimage.binary_fill_holes(bw)

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

def plot(image, mask):
    fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax1.imshow(image, cmap='hot')
    ax2.imshow(label2rgb(mask, bg_label=0), alpha=.9)
    ax1.set_title('Original')
    ax2.set_title('Detected parasite cells')

    # for u in np.unique(mask)[1:]:
    #     contours = measure.find_contours((mask==u))
    #     for contour in contours:
    #         ax2.plot(contour[:, 1], contour[:, 0], linewidth=3)

    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()

def segment_and_save(args):
    tif_path, seg_path = args[0], args[1]
    image = imageio.mimread(tif_path)[args[2]] # read image
    seg = get_segmentation(image, resize=args[3])
    seg_img = Image.fromarray(seg.astype(np.uint8), 'L')
    seg_img.save(seg_path)

def segment_dir(tif_dir, seg_dir, threads, channel=1, resize_shape=None):
    tif_paths = data_utils.get_paths(tif_dir, extension='.tif', recursive=True)
    seg_paths = [os.path.join(seg_dir, Path(tif_path).stem + '.png') for tif_path in tif_paths]

    combined_arguments = [(tif_path, seg_path, channel, resize_shape) for tif_path, seg_path in zip(tif_paths, seg_paths)]
    combined_arguments = [args for args in combined_arguments if not os.path.exists(args[1])]

    with Pool(threads) as p:
        p.map(segment_and_save, combined_arguments)

def segment_dataset(dataset, seg_dir, threads=1, channel=1, resize_shape=None):
    ids = np.arange(0, dataset.__len__())
    combined_arguments = []
    for idx in ids:
        tif_path, max_intensities = dataset.get_path_max(idx)
        seg_path = os.path.join(seg_dir, Path(tif_path).stem + '.png')
        combined_arguments.append((tif_path, seg_path, channel, resize_shape, max_intensities[0]))

    combined_arguments = [(tif_path, seg_path, channel, resize_shape) for tif_path, seg_path in zip(tif_paths, seg_paths)]

    combined_arguments = [(tif_path, seg_path, channel, resize_shape) for tif_path, seg_path in zip(tif_paths, seg_paths)]
    combined_arguments = [args for args in combined_arguments if not os.path.exists(args[1])]

    with Pool(threads) as p:
        p.map(segment_and_save, combined_arguments)

if __name__ == "__main__":
    # This is for testing
    # path = R"C:\Users\anton\Documents\microscopy_data\high_res_subset_from_Felix\B04\Exp2021025A-01-Scene-03-B4-B04_series_1_Exp2021025A-01-Scene-03-B4-B04.tif"
    # image = imageio.mimread(path)[2] # read image
    # mask = get_segmentation(image, (0,16000), resize=(1040, 1392))

    # path = R"C:\Users\anton\Documents\microscopy_data\dataset\images\NF135\D5\2019003_D5_135_hsp_20x_2_series_3_TileScan_001.tif"
    # image = imageio.mimread(path)[1] # read image
    # mask = get_segmentation(image, (0.0, 4095.0), resize=(1040, 1392))

    # plot(image, mask)

    # tif_dir = R"C:\Users\anton\Documents\microscopy_data\dataset\images\NF135\D5"
    # seg_dir = R"C:\Users\anton\Documents\microscopy_data\dataset\watershed"
    # segment_dir(tif_dir, seg_dir, threads=8, channel=1, resize=(1040, 1392))

    tif_dir = "/mnt/DATA1/anton/data/lowres_dataset/images/"
    seg_dir = "/mnt/DATA1/anton/data/lowres_dataset/watershed"
    segment_dir(tif_dir, seg_dir, threads=40, channel=1, resize_shape=(1040, 1392))
