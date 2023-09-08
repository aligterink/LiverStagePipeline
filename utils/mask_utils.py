from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

import torchvision.transforms as transforms
import numpy as np
import torch
import os
from pathlib import Path
import imageio.v2 as imageio
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt
from skimage.filters import gaussian
from scipy.interpolate import interp1d

def get_bbox_from_mask(mask, padding=0):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return torch.tensor([xmin-padding, ymin-padding, xmax+1+padding, ymax+1+padding])

def get_crop(image, bbox):
    if len(image.shape) > 2:
        return image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def multiple_bboxes_to_single_mask(bboxes, shape):
    mask = np.zeros(shape, dtype=bool)
    for bbox in bboxes:
        mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = True
    return mask

def get_crops(image, bboxes, masks, channels=None):
    # channels: a list of integers indicating which channels should be extracted. When None all are extracted.
    
    if channels is None:
        channels = [c for c in range(image.shape[0])]

    crops = []
    for bbox, mask in zip(bboxes, masks):

        inverse_mask_3d = (~mask).repeat(len(channels), 1, 1)
        zerod_image = image[channels, :, :]
        zerod_image[inverse_mask_3d] = 0

        crops.append(zerod_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]])

    return crops

def get_better_crops(image, bboxes, masks, channels=None):
    # channels: a list of integers indicating which channels should be extracted. When None all are extracted.
    
    if channels is None:
        channels = [c for c in range(image.shape[0])]

    crops = []
    for bbox, mask in zip(bboxes, masks):

        inverse_mask_3d = (~mask).repeat(len(channels), 1, 1)
        zerod_image = image[channels, :, :]
        zerod_image[inverse_mask_3d] = 0

        crops.append(zerod_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]])

    return crops

# From a pytorch bbox object formatted as XYXY returns central x,y coordinates
def bbox_to_position(bbox):
    return round(((bbox[2]-bbox[0]) / 2) + bbox[0]), round(((bbox[3]-bbox[1]) / 2) + bbox[1])

def store_crops(crops, bboxes, path_prefix, exclude_edgecases=True, image=None):
    coords = [bbox_to_position(bbox) for bbox in bboxes]

    for crop, coord, bbox in zip(crops, coords, bboxes):
        if exclude_edgecases:
            # Check if the bounding box lies within 2 pixels of the image border
            if np.any(np.array([bbox[0:2]]) < 3) or np.any((np.array([image.shape[2], image.shape[1]]) - np.array(bbox[2:])) < 3):
                continue
            
        path = path_prefix + '_{}x{}.tif'.format(coord[0], coord[1])
        crop = [crop[c, :, :].numpy() for c in range(crop.shape[0])] # format from 3d array to list of 2d arrays
        imageio.mimwrite(path, crop)

def find_minimum_value_between_two_points(coord1, coord2, array):
    y1, x1 = coord1
    y2, x2 = coord2
    # Calculate the horizontal and vertical components of movement
    dx = 1 if x2 > x1 else -1 if x2 < x1 else 0
    dy = 1 if y2 > y1 else -1 if y2 < y1 else 0

    # Calculate the number of steps in each direction
    steps_x = abs(x2 - x1)
    steps_y = abs(y2 - y1)

    # Initialize variables to store the values and current coordinates
    values = []
    x, y = x1, y1

    # Iterate over the maximum number of steps in either direction
    for step in range(max(steps_x, steps_y)):
        values.append(array[y, x])  # Collect the value at the current coordinate
        x += dx if step < steps_x else 0  # Move horizontally if there are more steps in that direction
        y += dy if step < steps_y else 0  # Move vertically if there are more steps in that direction

    # Collect the value at the ending coordinate
    values.append(array[y, x])

    return values


# def extract_crops_from_loader(loader, folder=None, channels=None, exclude_edgecases=True):
#     # exclude_edgecases: exclude crops that are on the edge of the image

#     for batch in tqdm(loader, leave=False):
#         for i in range(len(batch['images'])):
#             image = batch['images'][i]
#             bboxes = batch['boxes'][i].tolist()
#             mask_3d = batch['masks_3d'][i]
#             bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]
#             filepath = batch['file_paths'][i]

#             # modifications to cropping
#             better = True
#             if better:
#                 padding = 10
#                 bboxes = [[box[0] - padding, box[1] - padding, box[2]+padding, box[3]+padding] for box in bboxes]
#                 mask_3d =

#                 inverse_mask_3d = ~mask_3d
#                 dist = distance_transform_edt(inverse_mask_3d)
#                 g = 100**(1/padding)
#                 dist = g**-dist
#                 # print(dist)

#                 dist = dist**-0.5
                
#                 if channels is None:
#                     channels = [c for c in range(image.shape[0])]

#                 crops = []
#                 for bbox, mask in zip(bboxes, masks):

#                     inverse_mask_3d = (~mask).repeat(len(channels), 1, 1)
#                     zerod_image = image[channels, :, :]
#                     zerod_image[inverse_mask_3d] = 0

#                     crops.append(zerod_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]])

#                 return crops

#             else:
#                 crops = get_crops(image, bboxes, mask_3d, channels=channels)

#             if folder:
#                 path_prefix = os.path.join(folder, Path(filepath).stem)
#                 store_crops(crops=crops, bboxes=bboxes, path_prefix=path_prefix, exclude_edgecases=exclude_edgecases, image=image)

def extract_crops_from_loader(loader, folder=None, channels=None, exclude_edgecases=True):
    # exclude_edgecases: exclude crops that are on the edge of the image

    padding = 10
    g = 100**(1/padding)

    for batch in tqdm(loader, leave=False, desc='Extracting crops'): # loop over batches
        for i in range(len(batch['images'])):
            image = batch['images'][i]
            bboxes = batch['boxes'][i].tolist()
            mask_3d = batch['masks_3d'][i]
            bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]
            filepath = batch['file_paths'][i]

            if channels is None:
                channels = [c for c in range(image.shape[0])]

            for j in range(len(bboxes)): # loop over individual cells in one image
                bbox = bboxes[j]
                bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding] # add padding to bbox

                if exclude_edgecases:
                    # Check if the bounding box lies within 2 pixels of the image border
                    if np.any(np.array([bbox[0:2]]) < 3) or np.any((np.array([image.shape[2], image.shape[1]]) - np.array(bbox[2:])) < 3):
                        continue

                coords = bbox_to_position(bbox)
                mask = mask_3d[j, bbox[1]:bbox[3], bbox[0]:bbox[2]]

                inverse_mask = ~mask
                dist = distance_transform_edt(inverse_mask) # mask of distance to the cell for each pixel
                continuous_mask = g**-dist # 0 = 1, all other values are between 0 and 1. PADDING pixels away = 0.01
                continuous_mask = gaussian(continuous_mask, sigma=1)

                cropped_image = image[channels, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                crop = cropped_image * continuous_mask

                if folder: 
                    path = '{}_{}x{}.tif'.format(os.path.join(folder, Path(filepath).stem), coords[0], coords[1])
                    crop = [crop[c, :, :].numpy() for c in range(crop.shape[0])] # format from 3d array to list of 2d arrays
                    imageio.mimwrite(path, crop)

def extract_non_overlapping_crops_from_loader(loader, val_loader, folder=None, channels=None, exclude_edgecases=True):
    # exclude_edgecases: exclude crops that are on the edge of the image

    val_masks = {}

    for images, targets, _, filenames in val_loader:
        for image, target, filename in zip(images, targets, filenames):
            masks = target['masks'].cpu().numpy()
            val_masks[filename] = np.any(masks, axis=0)

    for images, targets, _, filenames in loader:
        for image, target, filename in zip(images, targets, filenames):

            bboxes = target['boxes'].tolist()
            masks = target['masks']
            bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]

            if len(bboxes) < 1:
                continue

            overlapping_indices = []

            for i in range(masks.shape[0]):
                overlapping_indices.append(np.any(np.logical_and(val_masks[filename], masks[i,:,:].cpu().numpy())))

            non_overlapping_indices = np.invert(overlapping_indices)
            
            bboxes = [bbox for bbox,non_overlap in zip(bboxes, non_overlapping_indices) if non_overlap]
            masks = masks[non_overlapping_indices]

            crops = get_crops(image, bboxes, masks, channels=channels)

            if folder:
                path_prefix = os.path.join(folder, Path(filename).stem)
                store_crops(crops=crops, bboxes=bboxes, path_prefix=path_prefix, exclude_edgecases=exclude_edgecases, image=image)

# From https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def create_circular_mask(h, w, centre=None, radius=None):
    Y, X = np.ogrid[:h, :w]
    dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)
    mask = dist_from_centre <= radius
    return mask

# for a grayscale image, set all pixels of a binary mask to zero.
def subtract_mask(img, mask):
    return np.where(mask > 0, np.median(img), img)

# Given a single pixel-wise mask of [H, W], generate a binary mask [n, H, W] where each n represents a different object
def mask_2d_to_3d(mask_2d):
    ids = np.unique(mask_2d)[1:]
    mask_3d = np.zeros((len(ids), mask_2d.shape[0], mask_2d.shape[1]), dtype=bool)
    for i, id in enumerate(ids):
        mask_3d[i,:,:][mask_2d == id] = True
    return mask_3d

# Given a boolean mask of [n, H, W] (numpy), generate an integer mask [H, W] where background = 0 and each integer represents a different object
def mask_3d_to_2d(mask_3d, max_overlap=0.3):
    # initialize 2d array with zeros
    mask_2d = np.zeros(mask_3d.shape[1:], dtype=int)
    
    # sort 3d array by descending size # TODO check if this works
    layer_sizes = np.sum(mask_3d, axis=(1, 2))
    sorted_indices = np.argsort(layer_sizes)[::-1]
    sorted_array = mask_3d[sorted_indices]

    for i in range(sorted_array.shape[0]):
        # create a mask for the current layer
        mask = sorted_array[i, :, :]
        
        # compare overlap of current layer with 2d array
        intersection = np.sum(np.logical_and(mask, mask_2d))
        mask_sum = np.sum(mask)
        overlap = intersection / mask_sum if mask_sum > 0 else 0
        
        # if the IoU is less than the threshold, add the layer to the output array
        if overlap <= max_overlap:
            mask_2d[mask] = i + 1
    
    return mask_2d
