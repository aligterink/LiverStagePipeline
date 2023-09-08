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
from tqdm import tqdm
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
# def segment_cells_in_image(image, eraser=None, save_path=None, channel=None, resize=(1392, 1040), old_intensity_range=None, equalize_adapthist=None):
#     if isinstance(image, str):
#         image_path = image
#         image = data_utils.parse_image(image_path, channels=channel, numpy_dtype=np.int64)

#     if eraser:
#         if isinstance(eraser, str):
#             eraser_path = eraser
#             eraser_mask = data_utils.parse_image(eraser_path, numpy_dtype=np.int8)
#         image = mask_utils.subtract_mask(image, eraser_mask)
#         image.shape

#     if resize:
#         original_size = image.shape
#         image = data_utils.resize(image, (resize[1], resize[0]))

#     if old_intensity_range:
#         image = data_utils.rescale(image, old_intensity_range, (0.0, 1.0)) # normalize image
#     else: 
#         image = data_utils.rescale(image, (np.min(image), np.max(image)), (0.0, 1.0)) # normalize image

#     min_pixels_cell, max_pixels_cell = np.product(image.shape) / 10000, np.product(image.shape) / 100
#     image_for_mask = image.copy()

#     # get the mask that indicates which pixels belong to cells
#     if equalize_adapthist:
#         image_for_mask = exposure.equalize_adapthist(image, kernel_size=(image.shape[0]/equalize_adapthist, image.shape[1]/equalize_adapthist), nbins=1000, clip_limit=0.005)
        

#     mask_threshold = threshold_otsu(image_for_mask)
#     mask = binary_fill_holes(image_for_mask > mask_threshold)
#     mask = morphology.remove_small_objects(mask, min_size=50)

#     for i in range(3):
#         mask = morphology.binary_erosion(mask)
#     for i in range(3):
#         mask = morphology.binary_dilation(mask)

#     # label image regions
#     label_image = label(mask)

#     total_labs = 0
#     blob_i = 1
#     final_blobs = np.zeros_like(image)
#     for i, region in enumerate(regionprops(label_image)):
#         if max_pixels_cell > region.area > min_pixels_cell: # take regions with large enough areas
#             blob_mask = (label_image == i+1)
            
#             blob_bbox = mask_utils.get_bbox_from_mask(blob_mask)
#             blob_dims = [blob_bbox[3]-blob_bbox[1], blob_bbox[2]-blob_bbox[0]]
#             if any(dim < 2 for dim in blob_dims):
#                 continue
#             blob_mask_crop = blob_mask[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]]

#             # Separate objects in image by generating markers as local maxima of the distance to the background and watershed
#             distance = ndi.distance_transform_edt(blob_mask_crop)
#             # print(image.shape, np.sum(image.shape), np.sum(image.shape)/2, int(np.sum(image.shape)/2*0.01))
#             coords = peak_local_max(distance, footprint=np.ones((9, 9)), labels=blob_mask_crop, min_distance=int(np.sum(image.shape)/2*0.01))
#             # import matplotlib.pyplot as plt
#             # plt.imshow(distance)
#             # plt.show()
#             mask = np.zeros(distance.shape, dtype=bool)
#             mask[tuple(coords.T)] = True
#             markers, _ = ndi.label(mask)
#             sub_blob_labels = watershed(-distance, markers=markers, mask=blob_mask_crop, compactness=10)
#             total_labs += len([x for x in regionprops(sub_blob_labels)])
#             for sub_blob in regionprops(sub_blob_labels):
#                 if sub_blob.area > min_pixels_cell: # take regions with large enough areas
#                     for l in np.unique(sub_blob_labels)[1:]:
#                         if np.unique(final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l]) == [0]:
#                             final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l] = blob_i
#                             blob_i += 1

#     # # For plotting
#     # import matplotlib.pyplot as plt
#     # # import matplotlib as mpl
#     # # from matplotlib.colors import LinearSegmentedColormap
#     # # cmap = mpl.cm.get_cmap('Paired', len(np.unique(segmentation)))
#     # # colors = cmap(np.linspace(0, 1, cmap.N))
#     # # np.random.shuffle(colors)
#     # # cmap = LinearSegmentedColormap.from_list('ShuffledCmap', colors, N=cmap.N)
#     # # cmap.set_under(color='black')
    
#     # # Show mask
#     # # axs[axs_coord].imshow(segmentation, cmap=cmap, vmin=0.9, interpolation='nearest')

#     # # imgs = {'Original': image, 'label image': label_image}
#     # fig, axes = plt.subplots(ncols=3, figsize=(50,20), sharex=True, sharey=True)
#     # ax = axes.ravel()
#     # # for i,(k,v) in enumerate(imgs.items()):
#     # #     ax[i].imshow(v, cmap=plt.cm.gray)
#     # #     ax[i].set_title(k)
#     # ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
#     # ax[1].imshow(image_for_mask, vmin=0.9, cmap=plt.cm.gray, interpolation='nearest')
#     # ax[2].imshow(final_blobs, vmin=0.9, cmap=plt.cm.gray, interpolation='nearest')
#     # fig.tight_layout()
#     # plt.show()
#     # input('hi')
#     # # # plt.savefig('/mnt/DATA1/anton/example7.png')

#     if resize:
#         final_blobs = data_utils.resize_mask(final_blobs, (original_size[1], original_size[0]))

#     if save_path:
#         data_utils.save_image(final_blobs.astype(np.uint16), save_path)

#     return final_blobs

def segment_cells_in_image(image, eraser=None, save_path=None, channel=None, resize=None, old_intensity_range=None, equalize_adapthist=None):
    if isinstance(image, str):
        image_path = image
        image = data_utils.parse_image(image_path, channels=channel, numpy_dtype=np.int64)

    if eraser:
        if isinstance(eraser, str):
            eraser_path = eraser
            eraser_mask = data_utils.parse_image(eraser_path, numpy_dtype=np.int8)
        image = mask_utils.subtract_mask(image, eraser_mask)

    if old_intensity_range:
        image = data_utils.rescale(image, old_intensity_range, (0.0, 1.0)) # normalize image
    else: 
        image = data_utils.rescale(image, (np.min(image), np.max(image)), (0.0, 1.0)) # normalize image

    min_pixels_cell, max_pixels_cell = np.product(image.shape) / 10000, np.product(image.shape) / 100
    image_for_mask = image.copy()

    # get the mask that indicates which pixels belong to cells
    if equalize_adapthist:
        image_for_mask = exposure.equalize_adapthist(image, kernel_size=(image.shape[0]/equalize_adapthist, image.shape[1]/equalize_adapthist), nbins=1000, clip_limit=0.005)
    
    image_for_mask_blurred = filters.gaussian(image_for_mask, sigma=10) # blur image

    mask = binary_fill_holes(image_for_mask_blurred > threshold_otsu(image_for_mask_blurred))
    mask = morphology.remove_small_objects(mask, min_size=50)
    for i in range(3):
        mask = erosion(mask)
    labeled_mask, _ = ndi.label(mask, structure=np.ones(shape=(3, 3)))

    unique_elements, counts = np.unique(labeled_mask, return_counts=True)
    mask = np.isin(labeled_mask, unique_elements[(counts >= min_pixels_cell) & (counts <= max_pixels_cell)], invert=True) # create a mask to identify elements to be set to 0
    labeled_mask[mask] = 0 # set elements to 0 based on the mask


    # blob_i = 1
    # final_blobs = np.zeros_like(image)
    # for i in np.unique(labeled_mask)[1:]:

    #     # For each blob extract a crop of the image and of the mask
    #     blob_mask = (labeled_mask == i)
    #     blob_bbox = mask_utils.get_bbox_from_mask(blob_mask, padding=0)
    #     blob_bbox = [max(0, blob_bbox[0]), max(0, blob_bbox[1]), min(mask.shape[1], blob_bbox[2]), min(mask.shape[0], blob_bbox[3])]
    #     if blob_bbox[3]-blob_bbox[1] < 2 or blob_bbox[2]-blob_bbox[0] < 2:
    #         continue
    #     blob_mask_crop = blob_mask[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]]
    #     blob_crop = image_for_mask_blurred[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]]
    #     blob_mask_crop_distance = ndi.distance_transform_edt(blob_mask_crop)



    #     # oi = blob_mask_crop_distance > (threshold_otsu(blob_mask_crop_distance) * 2)
    #     # marker_array, _ = ndi.label(oi)

    #     # x = filters.gaussian(blob_mask_crop.astype(np.uint8), sigma=10)

    #     # x = blob_crop.copy()

    #     # for i in range(2):
    #     #     x = x > threshold_otsu(ndi.distance_transform_edt(x))

    #     # blob_mask_crop_distance_sharp = filters.unsharp_mask(blob_mask_crop_distance, radius=1000.0, amount=0.00001) # make sharper

    #     # coords = extrema.local_maxima(blob_mask_crop_distance, indices=True)
    #     coords = peak_local_max(blob_mask_crop_distance, threshold_rel=0.7)
    #     print(len(coords))

    #     if len(coords) > 1:

    #         similarities = np.zeros((len(coords), len(coords)), dtype=np.float64)

    #         for i, coord1 in enumerate(coords):
    #             for j, coord2 in enumerate(coords):
    #                 values_inbetween = mask_utils.find_minimum_value_between_two_points(coord1, coord2, blob_mask_crop_distance)
    #                 similarities[i, j] = min(values_inbetween) / min(values_inbetween[0], values_inbetween[-1])
    #                 similarities[j, i] = min(values_inbetween) / min(values_inbetween[0], values_inbetween[-1])

    #         print(similarities)


    #         # Define the threshold value (N) for clustering
    #         threshold = 0.9  # Adjust as needed

    #         # Convert the similarity matrix to a condensed distance matrix
    #         distance_matrix = 1 - similarities  # Convert similarity to distance
    #         condensed_distance_matrix = squareform(distance_matrix)

    #         linkage_matrix = linkage(condensed_distance_matrix, method='single')  # Single-linkage clustering
    #         clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    #         # Initialize a dictionary to store cluster indices and centroids
    #         cluster_dict = {}

    #         # Calculate the centroid for each cluster
    #         for cluster_id in np.unique(clusters):
    #             cluster_samples = np.where(clusters == cluster_id)[0]  # Indices of samples in this cluster
    #             cluster_similarity_scores = similarities[cluster_samples, :]  # Similarity scores for this cluster
    #             cluster_centroid = np.mean(cluster_similarity_scores, axis=0)  # Calculate the centroid
    #             cluster_dict[cluster_id] = {
    #                 'samples': cluster_samples,
    #                 'centroid': cluster_centroid
    #             }
            
    #         # If you need the indices of the centroids only:
    #         centroid_indices = [info['samples'][np.argmax(info['centroid'])] for info in cluster_dict.values()]
    #         print("Indices of Centroids:", centroid_indices)

    #     marker_array = np.zeros(blob_crop.shape, dtype=bool)

    #     for coord in coords:
    #         marker_array[coord[0], coord[1]] = True

    #     # coords = feature.blob_dog(x, min_sigma=1, max_sigma=200, sigma_ratio=1.01, threshold=.003)#, min_sigma=4, max_sigma=100, sigma_ratio=1.15, overlap=0.1)
    #     # print(coords.shape)
    #     # marker_array = np.zeros(shape=blob_mask_crop.shape, dtype=np.uint8)
    #     # for i in range(coords.shape[0]):
    #     #     marker_array[np.array(coords[i,:][0]).astype(np.uint8), np.array(coords[i,:][1]).astype(np.uint8)] = True

    #     # marker_array, _ = ndi.label(marker_array)


    #     sub_blob_labels = watershed(-blob_mask_crop_distance, markers=marker_array, mask=blob_mask_crop, compactness=1).astype(np.uint16)
    #     # print('marker array', len(np.unique(marker_array))-1, len(np.unique(sub_blob_labels))-1)


    #     # For plotting
    #     import matplotlib.pyplot as plt
    #     oi = np.zeros(shape=mask.shape)

    #     # for i in range(coords.shape[0]):
    #     #     oi[int(coords[i,:][0]), int(coords[i,:][1])] = True

    #     imgs = {'blob_crop': blob_crop, 'blob_mask_crop': blob_mask_crop, 'marker_array': marker_array, 
    #             'blob_mask_crop_distance': blob_mask_crop_distance,
    #             'sub_blob_labels': sub_blob_labels}
    #     fig, axes = plt.subplots(ncols=len(imgs.items())+1, figsize=(20,5), sharex=True, sharey=True)
    #     ax = axes.ravel()
    #     for i,(k,v) in enumerate(imgs.items()):
    #         ax[i].imshow(v, cmap=plt.cm.gray)
    #         ax[i].set_title(k)
    #     fig.tight_layout()
    #     plt.show()


    #     for sub_blob in regionprops(sub_blob_labels):
    #         if sub_blob.area > min_pixels_cell: # take regions with large enough areas
    #             for l in np.unique(sub_blob_labels)[1:]:
    #                 # if np.unique(final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l]) == [0]:
    #                 final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l] = blob_i
    #                 blob_i += 1
        
    if save_path:
        data_utils.save_image(labeled_mask.astype(np.uint16), save_path)

    return labeled_mask

# def segment_cells_in_image(image, eraser=None, save_path=None, channel=None, resize=None, old_intensity_range=None, equalize_adapthist=None):
#     if isinstance(image, str):
#         image_path = image
#         image = data_utils.parse_image(image_path, channels=channel, numpy_dtype=np.int64)

#     if eraser:
#         if isinstance(eraser, str):
#             eraser_path = eraser
#             eraser_mask = data_utils.parse_image(eraser_path, numpy_dtype=np.int8)
#         image = mask_utils.subtract_mask(image, eraser_mask)

#     # if resize:
#     #     original_size = image.shape
#     #     image = data_utils.resize(image, (resize[1], resize[0]))

#     if old_intensity_range:
#         image = data_utils.rescale(image, old_intensity_range, (0.0, 1.0)) # normalize image
#     else: 
#         image = data_utils.rescale(image, (np.min(image), np.max(image)), (0.0, 1.0)) # normalize image

#     min_pixels_cell, max_pixels_cell = np.product(image.shape) / 10000, np.product(image.shape) / 100
#     image_for_mask = image.copy()

#     # get the mask that indicates which pixels belong to cells
#     if equalize_adapthist:
#         image_for_mask = exposure.equalize_adapthist(image, kernel_size=(image.shape[0]/equalize_adapthist, image.shape[1]/equalize_adapthist), nbins=1000, clip_limit=0.005)
        
#     image_for_mask = filters.gaussian(image_for_mask, sigma=10) # blur image

#     mask_threshold = threshold_otsu(image_for_mask)
#     mask = binary_fill_holes(image_for_mask > mask_threshold)
#     mask = morphology.remove_small_objects(mask, min_size=80)
#     for i in range(3):
#         mask = erosion(mask)

#     for i in range(3):
#         mask = morphology.binary_erosion(mask)
#     for i in range(3):
#         mask = morphology.binary_dilation(mask)

#     # label image regions
#     label_image = label(mask)

#     # ## For plotting
#     # import matplotlib.pyplot as plt
#     # imgs = {'image': image, 'image_for_mask': image_for_mask, 'mask': mask, 'label_image': label_image}
#     # fig, axes = plt.subplots(ncols=len(imgs.items())+1, figsize=(20,5), sharex=True, sharey=True)
#     # ax = axes.ravel()
#     # for i,(k,v) in enumerate(imgs.items()):
#     #     ax[i].imshow(v, cmap=plt.cm.gray)
#     #     ax[i].set_title(k)

#     # fig.tight_layout()
#     # plt.show()
#     total_labs = 0
#     blob_i = 1
#     final_blobs = np.zeros_like(image)
#     for i, region in enumerate(regionprops(label_image)):
#         if max_pixels_cell > region.area > min_pixels_cell: # take regions with large enough areas
#             blob_mask = (label_image == i+1)
            
#             blob_bbox = mask_utils.get_bbox_from_mask(blob_mask)
#             blob_dims = [blob_bbox[3]-blob_bbox[1], blob_bbox[2]-blob_bbox[0]]
#             if any(dim < 2 for dim in blob_dims):
#                 continue
#             blob_mask_crop = blob_mask[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]]

#             # Separate objects in image by generating markers as local maxima of the distance to the background and watershed
#             distance = ndi.distance_transform_edt(blob_mask_crop)
#             coords = peak_local_max(distance, footprint=np.ones((9, 9)), labels=blob_mask_crop, min_distance=int(np.sum(image.shape)/2*0.01))

#             mask = np.zeros(distance.shape, dtype=bool)
#             mask[tuple(coords.T)] = True
#             markers, _ = ndi.label(mask)
#             sub_blob_labels = watershed(-distance, markers=markers, mask=blob_mask_crop, compactness=10)
#             total_labs += len([x for x in regionprops(sub_blob_labels)])
#             for sub_blob in regionprops(sub_blob_labels):
#                 if sub_blob.area > min_pixels_cell: # take regions with large enough areas
#                     for l in np.unique(sub_blob_labels)[1:]:
#                         if np.unique(final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l]) == [0]:
#                             final_blobs[blob_bbox[1]:blob_bbox[3], blob_bbox[0]:blob_bbox[2]][sub_blob_labels==l] = blob_i
#                             blob_i += 1

#     if save_path:
#         data_utils.save_image(final_blobs.astype(np.uint16), save_path)

#     return final_blobs


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

    img_folder = '/home/anton/Documents/high_res_testing/tifs'
    parasite_mask_folder = '/home/anton/Documents/high_res_testing/parasite_segmentations'
    hepatocyte_mask_folder = '/home/anton/Documents/high_res_testing/hepatocyte_segmentations'

    segment_cells_in_folder(image_folder=img_folder, segmentation_folder=parasite_mask_folder, threads=40, channel=0, resize_shape=None, normalize=True, equalize_adapthist=8)

    ### High res hepatocyte segmentation
    segment_cells_in_folder(image_folder=img_folder, parasite_mask_folder=parasite_mask_folder, segmentation_folder=hepatocyte_mask_folder, threads=40, channel=2, normalize=True, equalize_adapthist=24)