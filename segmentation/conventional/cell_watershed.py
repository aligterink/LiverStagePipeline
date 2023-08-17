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

### Methods for segmenting entire images, either schizonts or hepatocytes
# equalize_adapthist: adjusts contrast to make cells spring out more, even if the lighting is dim. However, this parameter may also highlight background noise.
#   Therefore, only fill this argument with value N when you are sure that in +95% of cases when you divide an image into an NxN grid, at least one cell will be present.
#   I.e. this parameter should be kept None if images may be void of cells.
def segment_cells_in_image(image, eraser=None, save_path=None, channel=None, resize=None, old_intensity_range=None, equalize_adapthist=None):
    if isinstance(image, str):
        image_path = image
        image = data_utils.parse_image(image_path, channels=channel, numpy_dtype=np.int64)

    if eraser:
        if isinstance(eraser, str):
            eraser_path = eraser
            eraser_mask = data_utils.parse_image(eraser_path, numpy_dtype=np.int8)
        image = mask_utils.subtract_mask(image, eraser_mask)
        image.shape

    if resize:
        original_size = image.shape
        image = data_utils.resize(image, (resize[1], resize[0]))

    if old_intensity_range:
        image = data_utils.normalize(image, old_intensity_range, (0.0, 1.0)) # normalize image
    else: 
        image = data_utils.normalize(image, (np.min(image), np.max(image)), (0.0, 1.0)) # normalize image

    min_pixels_cell, max_pixels_cell = np.product(image.shape) / 10000, np.product(image.shape) / 100
    image_for_mask = image.copy()

    # get the mask that indicates which pixels belong to cells
    if equalize_adapthist:
        image_for_mask = exposure.equalize_adapthist(image, kernel_size=(image.shape[0]/equalize_adapthist, image.shape[1]/equalize_adapthist), nbins=1000, clip_limit=0.005)

    mask_threshold = threshold_otsu(image_for_mask)
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
    final_blobs = np.zeros_like(image)
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
            coords = peak_local_max(distance, footprint=np.ones((9, 9)), labels=blob_mask_crop, min_distance=10)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            sub_blob_labels = watershed(-distance, markers, mask=blob_mask_crop)
            total_labs += len([x for x in regionprops(sub_blob_labels)])
            for sub_blob in regionprops(sub_blob_labels):
                if sub_blob.area > min_pixels_cell: # take regions with large enough areas
                    for l in np.unique(sub_blob_labels)[1:]:
                        if np.unique(final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l]) == [0]:
                            final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l] = blob_i
                            blob_i += 1

    ## For plotting
    # import matplotlib.pyplot as plt
    # imgs = {'Original': image, 'image for mask': image_for_mask, 'label image': label_image}
    # fig, axes = plt.subplots(ncols=len(imgs.items())+1, figsize=(50,20), sharex=True, sharey=True)
    # ax = axes.ravel()
    # for i,(k,v) in enumerate(imgs.items()):
    #     ax[i].imshow(v, cmap=plt.cm.gray)
    #     ax[i].set_title(k)
    # ax[-1].imshow(final_blobs, cmap=plt.cm.nipy_spectral)
    # ax[-1].set_title('Labels')
    # fig.tight_layout()
    # plt.show()
    # plt.savefig('/mnt/DATA1/anton/example7.png')

    if resize:
        final_blobs = data_utils.resize_mask(final_blobs, (original_size[1], original_size[0]))

    if save_path:
        data_utils.save_image(final_blobs.astype(np.uint16), save_path)

    return final_blobs


def call_segment_cells_in_image(args):
    img_path, cells_mask_path, channel, resize, old_intensity_range, parasite_mask_path, equalize_adapthist = args
    segment_cells_in_image(img_path, save_path=cells_mask_path, channel=channel, resize=resize, old_intensity_range=old_intensity_range, eraser=parasite_mask_path, equalize_adapthist=equalize_adapthist)

def segment_cells_in_folder(image_folder, segmentation_folder, threads, channel, resize_shape=None, normalize=True, parasite_mask_folder=None, equalize_adapthist=None):
    if parasite_mask_folder:
        image_paths, parasite_mask_paths = data_utils.get_two_sets(image_folder, parasite_mask_folder, common_subset=True, extension_dir1='.tif', extension_dir2='', return_paths=True)
        segmentation_paths = [os.path.join(segmentation_folder, data_utils.get_common_suffix(img_path, parasite_mask_path), Path(img_path).stem + '.tif') for img_path, parasite_mask_path in zip(image_paths, parasite_mask_paths)]
    else:
        image_paths = data_utils.get_paths(image_folder, extension='.tif', recursive=True)
        parasite_mask_paths = [None] * len(image_paths)
        segmentation_paths = [os.path.join(segmentation_folder, image_path[len(image_folder)+1:]) for image_path in image_paths]

    if normalize:
        ranges = data_utils.find_folder_range(image_paths, channels=[channel])
        file_ranges = [ranges[os.path.dirname(img_path)][channel] if normalize else None for img_path in image_paths]

    filewise_arguments = [(img_path, cells_mask_path, channel, resize_shape, file_range, parasite_mask_path, equalize_adapthist) for img_path, cells_mask_path, file_range, parasite_mask_path in zip(image_paths, segmentation_paths, file_ranges, parasite_mask_paths)]
    filewise_arguments = [args for args in filewise_arguments if not os.path.exists(args[1])]

    with Pool(threads) as p:
        list(tqdm(p.imap(call_segment_cells_in_image, filewise_arguments), leave=False, desc='Segmenting cells', total=len(filewise_arguments)))

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

    # ### Parasite segmentation
    # img_folder = '/mnt/DATA1/anton/data/unformatted/high_res_subset_from_Felix/F10_GS-HSP'
    # mask_folder = '/mnt/DATA1/anton/data/unformatted/high_res_subset_from_Felix/F10_GS-HSP_watershed_test'
    # segment_cells_in_folder(image_folder=img_folder, segmentation_folder=mask_folder, threads=1, channel=0, resize_shape=None, normalize=True, equalize_adapthist=None)

    img_folder = '/mnt/DATA1/anton/data/unformatted/high_res_subset_from_Felix/F10_GS-HSP'
    parasite_mask_folder = '/mnt/DATA1/anton/data/unformatted/high_res_subset_from_Felix/F10_GS-HSP_watershed_test'
    hepatocyte_mask_folder = '/mnt/DATA1/anton/data/unformatted/high_res_subset_from_Felix/F10_GS-HSP_watershed_test_hepatocytes'
    hepatocyte_channel = 1 # channel the hepatocytes occur in, when counting from 0

    segment_cells_in_folder(image_folder=img_folder, segmentation_folder=hepatocyte_mask_folder, parasite_mask_folder=parasite_mask_folder, threads=40, channel=hepatocyte_channel, resize_shape=None, normalize=True, equalize_adapthist=24)