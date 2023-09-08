import sys
import os
sys.path.append(os.path.sep + os.path.join(*(os.path.abspath('').split(os.path.sep)[:next((i for i in range(len(os.path.abspath('').split(os.path.sep)) -1, -1, -1) if 'LiverStagePipeline' in os.path.abspath('').split(os.path.sep)[i]), None)+1])))

from utils import data_utils, mask_utils

import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import imageio.v2 as imageio
from scipy import ndimage as ndi

from pathlib import Path
from skimage import filters
from skimage.morphology import extrema
from skimage.morphology import remove_small_holes
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

#### Methods for segmenting the merozoites within schizonts
def _segment_merozoites_in_crop(image):
    image = data_utils.normalize(image, (np.min(image), np.max(image)), (0, 1)) # normalize

    image_for_peaks_blurred = filters.gaussian(image, sigma=1) # blur image
    image_for_peaks_sharp = filters.unsharp_mask(image_for_peaks_blurred, radius=1.0, amount=50) # make sharper
    peaks_image_threshold = threshold_otsu(image_for_peaks_sharp) # compute otsu treshold
    peak_mask = image_for_peaks_sharp > peaks_image_threshold #binary_fill_holes(sharp_image > peak_threshold)
    peak_mask = remove_small_holes(image_for_peaks_sharp > peaks_image_threshold, area_threshold=np.product(image.shape)/50)
    distance_peak_mask = ndi.distance_transform_edt(peak_mask)

    # Get peak positions by finding local maxima of the peaks mask
    markers = extrema.local_maxima(distance_peak_mask)
    labeled_markers, _ = ndi.label(markers, structure=np.ones(shape=(3, 3)))
    labels = watershed(-distance_peak_mask, labeled_markers, mask=peak_mask, compactness=2) # do watershed


    # Remove small labels
    unique_elements, element_counts = np.unique(labels, return_counts=True)
    elements_to_keep = unique_elements[element_counts > 6]
    mask = np.isin(labels, elements_to_keep)
    labels[~mask] = 0

    ## For plotting
    import matplotlib.pyplot as plt
    imgs = {'Original': image, 'sharp': image_for_peaks_sharp, 
            'Sharp binary': peak_mask, 'distance sharp': distance_peak_mask, 'markers': markers, 'eroded': labeled_markers}
    fig, axes = plt.subplots(ncols=len(imgs.items())+1, figsize=(20,5), sharex=True, sharey=True)
    ax = axes.ravel()
    for i,(k,v) in enumerate(imgs.items()):
        ax[i].imshow(v, cmap=plt.cm.gray)
        ax[i].set_title(k)
    ax[-1].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[-1].set_title('Labels')
    fig.tight_layout()
    plt.show()
    # plt.savefig('/mnt/DATA1/anton/example5.png')
    # input('waiting for input ...')
    return labels

# returns: mask where each individual parasite is segmented. A mask of the original size is returned, 
# where each cell contains multiple labeled regions. Labels are not unique.
def segment_merozoites_in_image(image, parasite_mask, channel, merozoite_mask_path=None):
    if isinstance(image, str):
        image_path = image
        image = data_utils.parse_image(image_path, channels=channel, numpy_dtype=np.float64)

    if isinstance(parasite_mask, str):
        parasite_mask_path = parasite_mask
        parasite_mask = data_utils.parse_image(parasite_mask_path, numpy_dtype=np.int8)

    merozoite_mask = np.zeros_like(parasite_mask)
    
    for parasite_id in np.unique(parasite_mask)[1:]:
        cell_mask = (parasite_mask == parasite_id)
        cell_bbox = mask_utils.get_bbox_from_mask(cell_mask)
        cell_dims = [cell_bbox[3]-cell_bbox[1], cell_bbox[2]-cell_bbox[0]]
        if any(dim < 2 for dim in cell_dims):
            continue

        intensity_crop = image[cell_bbox[1]:cell_bbox[3], cell_bbox[0]:cell_bbox[2]]
        cell_mask_crop = cell_mask[cell_bbox[1]:cell_bbox[3], cell_bbox[0]:cell_bbox[2]]

        pad_width = round(np.sqrt(np.product(intensity_crop.shape))/10)
        pad_constant = np.min(intensity_crop) # np.percentile(intensity_crop, 0.4)

        intensity_crop[~cell_mask_crop] = pad_constant # make outside of mask minimum of image
        padded_intensity_crop = np.pad(intensity_crop, mode='constant', constant_values=pad_constant, pad_width=pad_width)

        cell_segmentation_mask_crop = _segment_merozoites_in_crop(padded_intensity_crop)
        cell_segmentation_mask_crop = cell_segmentation_mask_crop[pad_width:-pad_width, pad_width:-pad_width] if pad_width > 0 else cell_segmentation_mask_crop
        merozoite_mask[cell_bbox[1]:cell_bbox[3], cell_bbox[0]:cell_bbox[2]] = cell_segmentation_mask_crop

    if merozoite_mask_path:
        data_utils.save_image(merozoite_mask.astype(np.uint16), merozoite_mask_path)

    # ### Plotting
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(figsize=(20, 20), sharex=True, sharey=True)
    # plt.imshow(cell_segmentation_mask)
    # plt.show()
    return merozoite_mask

def call_segment_merozoites_in_image(args):
    img_path, parasite_mask_path, channel, merozoite_mask_path = args
    segment_merozoites_in_image(img_path, parasite_mask_path, channel, merozoite_mask_path)

def segment_merozoites_in_folder(image_folder, parasite_mask_folder, merozoite_mask_folder, threads, channel=1):
    image_paths, parasite_mask_paths = data_utils.get_two_sets(image_folder, parasite_mask_folder, common_subset=True, extension_dir1='.tif', extension_dir2='', return_paths=True)
    merozoite_mask_paths = [os.path.join(merozoite_mask_folder, data_utils.get_common_suffix(img_path, parasite_mask_path), Path(img_path).stem + '.tif') for img_path, parasite_mask_path in zip(image_paths, parasite_mask_paths)]
    filewise_arguments = [(tif_path, parasite_mask_path, channel, merozoite_mask_path) for tif_path, parasite_mask_path, merozoite_mask_path in zip(image_paths, parasite_mask_paths, merozoite_mask_paths)]

    filewise_arguments = [args for args in filewise_arguments if not os.path.exists(args[3])]
    with Pool(threads) as p:
        list(tqdm(p.imap(call_segment_merozoites_in_image, filewise_arguments), leave=False, desc='Segmenting merozoites'))
    # for args in filewise_arguments:
    #     call_segment_merozoites_in_image(args)

if __name__ == "__main__":
    # image_folder = "/mnt/DATA1/anton/data/lowres_dataset/images"
    # parasite_mask_folder = "/mnt/DATA1/anton/data/lowres_dataset/annotation"
    # merozoite_mask_folder = '/mnt/DATA1/anton/data/lowres_dataset/merozoite_watershed'
    # hepatocyte_folder = '/mnt/DATA1/anton/data/lowres_dataset/hepatocyte_watershed'

    # segment_merozoites_in_folder(image_folder, parasite_mask_folder, merozoite_mask_folder, threads=40, channel=0)

    img_path = '/mnt/DATA1/anton/data/lowres_dataset/images/NF135/D7/2019003_D7_135_hsp_gs_a_series_2_TileScan_001.tif'
    parasite_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/annotation/NF135/D7/2019003_D7_135_hsp_gs_a_series_2_TileScan_001.png'
    segment_merozoites_in_image(img_path, parasite_mask_path, 0, merozoite_mask_path=None)
