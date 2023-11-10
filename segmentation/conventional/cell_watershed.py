import sys
import os
sys.path.append(os.path.sep + os.path.join(*(__file__.split(os.path.sep)[:next((i for i in range(len(__file__.split(os.path.sep)) -1, -1, -1) if 'LiverStagePipeline' in __file__.split(os.path.sep)[i]), None)+1])))

from utils import data_utils, mask_utils

import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

import imageio.v2 as imageio
import cv2
from scipy import ndimage as ndi
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, erosion, dilation, area_closing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from pathlib import Path
from skimage import filters, morphology, measure, segmentation
from scipy.ndimage import binary_fill_holes

import numpy as np
from PIL import Image
from multiprocessing import Pool
import imageio.v2 as imageio
from scipy import ndimage as ndi

from pathlib import Path
from skimage import filters
from skimage.morphology import extrema
from skimage.morphology import remove_small_holes
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage import feature
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

### Methods for segmenting entire images, either schizonts or hepatocytes
# equalize_adapthist: adjusts contrast to make cells spring out more, even if the lighting is dim. However, this parameter may also highlight background noise.
#   Therefore, only fill this argument with value N when you are sure that in +95% of cases when you divide an image into an NxN grid, at least one cell will be present.
#   I.e. this parameter should be kept None if images may be void of cells.
def segment_cells_in_image(image, eraser=None, save_path=None, channel=None, resize=(1392, 1040), old_intensity_range=None, equalize_adapthist=None, otsu_threshold=None):
    if isinstance(image, str):
        image_path = image
        image = data_utils.parse_image(image_path, channels=channel, numpy_dtype=np.int64)

    if eraser:
        if isinstance(eraser, str):
            eraser_path = eraser
            eraser_mask = data_utils.parse_image(eraser_path, numpy_dtype=np.int16)
        image = mask_utils.subtract_mask(image, eraser_mask)
        image.shape

    if resize:
        original_size = image.shape
        image = data_utils.resize(image, (resize[1], resize[0]))

    if old_intensity_range:
        image = data_utils.rescale(image, old_intensity_range, (0.0, 1.0)) # normalize image
    # else: 
    #     image = data_utils.rescale(image, (np.min(image), np.max(image)), (0.0, 1.0)) # normalize image

    min_pixels_cell, max_pixels_cell = np.product(image.shape) / 10000, np.product(image.shape) / 100
    image_for_mask = image.copy()

    # get the mask that indicates which pixels belong to cells
    if equalize_adapthist:
        old_min, old_max = np.min(image), np.max(image)
        image = data_utils.rescale(image, (old_min, old_max), (-1.0, 1.0))
        image = exposure.equalize_adapthist(image, kernel_size=(image.shape[0]/equalize_adapthist, image.shape[1]/equalize_adapthist), nbins=1000, clip_limit=0.005)
        image = data_utils.rescale(image, (-1.0, 1.0), (old_min, old_max))


    mask_threshold = otsu_threshold if otsu_threshold else threshold_otsu(image_for_mask)
    mask = binary_fill_holes(image_for_mask > mask_threshold)
    mask = morphology.remove_small_objects(mask, min_size=50)

    for i in range(3):
        mask = morphology.binary_erosion(mask)
    for i in range(3):
        mask = morphology.binary_dilation(mask)

    # label image regions
    label_image = label(mask)

    total_labs = 0
    blob_i = 1
    final_blobs = np.zeros_like(image, dtype=np.uint16)
    for i, region in enumerate(regionprops(label_image)):
        if max_pixels_cell > region.area > min_pixels_cell: # take regions with large enough areas
            blob_mask = (label_image == i+1)
            
            blob_bbox = mask_utils.get_bbox_from_mask(blob_mask)
            blob_dims = [blob_bbox[3]-blob_bbox[1], blob_bbox[2]-blob_bbox[0]]
            if any(dim < 2 for dim in blob_dims):
                continue
            blob_mask_crop = blob_mask[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]]

            # Separate objects in image by generating markers as local maxima of the distance to the background and watershed
            distance = ndi.distance_transform_edt(blob_mask_crop)
            # print(image.shape, np.sum(image.shape), np.sum(image.shape)/2, int(np.sum(image.shape)/2*0.01))
            coords = peak_local_max(distance, footprint=np.ones((9, 9)), labels=blob_mask_crop, min_distance=int(np.sum(image.shape)/2*0.01))
            # import matplotlib.pyplot as plt
            # plt.imshow(distance)
            # plt.show()
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            sub_blob_labels = watershed(-distance, markers=markers, mask=blob_mask_crop, compactness=10)
            total_labs += len([x for x in regionprops(sub_blob_labels)])
            for sub_blob in regionprops(sub_blob_labels):
                if sub_blob.area > min_pixels_cell: # take regions with large enough areas
                    for l in np.unique(sub_blob_labels)[1:]:
                        if np.unique(final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l]) == [0]:
                            final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l] = blob_i
                            blob_i += 1
            if len(np.unique(sub_blob_labels)) < 2:
                final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][blob_mask_crop] = blob_i
                blob_i += 1

    if resize:
        final_blobs = data_utils.resize_mask(final_blobs, (original_size[1], original_size[0]))

    if save_path:
        data_utils.save_image(final_blobs, save_path)
    return final_blobs


def call_segment_cells_in_image(args):
    img_path, cells_mask_path, channel, resize, old_intensity_range, parasite_mask_path, equalize_adapthist, threshold = args
    segment_cells_in_image(img_path, save_path=cells_mask_path, channel=channel, resize=resize, old_intensity_range=old_intensity_range, eraser=parasite_mask_path, equalize_adapthist=equalize_adapthist, otsu_threshold=threshold)

def segment_cells_in_folder(image_paths, segmentation_folder, threads, channels, resize_shape=None, normalize=True, parasite_mask_paths=None, equalize_adapthist=None):
# def segment_cells_in_folder(image_folder, segmentation_folder, threads, channels, resize_shape=None, normalize=True, parasite_mask_folder=None, equalize_adapthist=None):
    # if parasite_mask_folder:
    #     image_paths, parasite_mask_paths = data_utils.get_two_sets(image_folder, parasite_mask_folder, common_subset=True, extension_dir1='.tif', extension_dir2='', return_paths=True)
    #     segmentation_paths = [os.path.join(segmentation_folder, data_utils.get_common_suffix(img_path, parasite_mask_path), Path(img_path).stem + '.tif') for img_path, parasite_mask_path in zip(image_paths, parasite_mask_paths)]
    # else:
    #     image_paths = data_utils.get_paths(image_folder, extension='.tif', recursive=True)
    #     parasite_mask_paths = [None] * len(image_paths)
    #     segmentation_paths = [os.path.join(segmentation_folder, image_path[len(image_folder)+1:]) for image_path in image_paths]

    parasite_mask_paths = parasite_mask_paths if parasite_mask_paths else [None] * len(image_paths)
    # segmentation_paths = [os.path.join(segmentation_folder, image_path[len(image_folder)+1:]) for image_path in image_paths]
    # segmentation_paths = [os.path.join(segmentation_folder, ''.join(c1 for c1, c2 in zip(image_path, segmentation_folder) if c1 == c2) for image_path in image_paths]
    # image_paths_prefix = data_utils.find_common_substring(image_paths)
    segmentation_paths = [os.path.join(segmentation_folder, os.path.basename(image_path)) for image_path in image_paths]

    if normalize:
        # file_ranges = [data_utils.find_folder_range(image_paths, channels=[[channel]]*len(image_paths))[os.path.dirname(img_path)][str(channel)] if normalize else None for img_path in image_paths]
        # file_ranges, thresholds = data_utils.find_folder_range(image_paths, channels=[[channel]]*len(image_paths), otsu=True)
        file_ranges, thresholds = data_utils.find_folder_range(image_paths, channels=[[c] for c in channels], otsu=True)
    else:
        thresholds = [None]*len(image_paths)
    file_ranges = [None]*len(image_paths)

    filewise_arguments = [(img_path, cells_mask_path, channel, resize_shape, file_range, parasite_mask_path, equalize_adapthist, threshold) for img_path, cells_mask_path, channel, file_range, parasite_mask_path, threshold in zip(image_paths, segmentation_paths, channels, file_ranges, parasite_mask_paths, thresholds)]
    filewise_arguments = [args for args in filewise_arguments if not os.path.exists(args[1])]

    with Pool(threads) as p:
        list(tqdm(p.imap(call_segment_cells_in_image, filewise_arguments), leave=False, desc='Segmenting cells', total=len(filewise_arguments)))
    # for args in filewise_arguments:
    #     call_segment_cells_in_image(args)

if __name__ == "__main__":

    # image_folder = "/mnt/DATA1/anton/data/lowres_dataset/images/NF135/D5"
    # parasite_mask_folder = "/mnt/DATA1/anton/data/lowres_dataset/annotation/NF135/D5"
    # merozoite_mask_folder = '/mnt/DATA1/anton/data/lowres_dataset/merozoite_watershed'
    # hepatocyte_folder = '/mnt/DATA1/anton/data/lowres_dataset/hepatocyte_watershed'
    # segment_cells_in_folder(image_folder=image_folder, cells_folder=hepatocyte_folder, threads=40, channel=0, parasite_mask_folder=parasite_mask_folder)

    # ### Lowres parasite segmentation testing
    # img_path = '/mnt/DATA1/anton/data/lowres_dataset/images/NF135/D5/2019003_D5_135_hsp_20x_2_series_2_TileScan_001.tif'
    # # segment_cells_in_image(image=img_path, eraser=parasite_mask_path, channel=0)
    # segment_cells_in_image(image=img_path, eraser=None, channel=1)

    ### High res parasite and hepatocyte segmentation

    # img_folder = '/mnt/DATA1/anton/data/dataset/hepatocyte_data/images/lowres'
    # parasite_mask_folder = '/mnt/DATA1/anton/data/dataset/annotation'
    # hepatocyte_mask_folder = '/mnt/DATA1/anton/data/dataset/hepatocyte_data/annotation2/lowres'

    # # img_folder = '/mnt/DATA1/anton/highres_watershed_test/tifs'
    # # parasite_mask_folder = '/mnt/DATA1/anton/highres_watershed_test/parasite_segmentations'
    # # hepatocyte_mask_folder = '/mnt/DATA1/anton/highres_watershed_test/hepatocyte_segmentations'

    # # segment_cells_in_folder(image_folder=img_folder, segmentation_folder=parasite_mask_folder, threads=40, channel=0, resize_shape=None, normalize=True, equalize_adapthist=None)

    # ### High res hepatocyte segmentation
    # segment_cells_in_folder(image_folder=img_folder, parasite_mask_folder=parasite_mask_folder, segmentation_folder=hepatocyte_mask_folder, threads=40, channel=0, normalize=False, equalize_adapthist=None)

    img_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/images/lowres/NF54/D7'
    parasite_mask_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/annotation/lowres/NF54/D7'
    hepatocyte_mask_folder = '/mnt/DATA1/anton/data/basically_trash'

    segment_cells_in_folder(image_folder=img_folder, parasite_mask_folder=parasite_mask_folder, segmentation_folder=hepatocyte_mask_folder, threads=40, channel=0, normalize=False, equalize_adapthist=None)



