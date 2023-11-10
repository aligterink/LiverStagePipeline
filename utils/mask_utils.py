from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

import numpy as np
import torch
import os
from pathlib import Path
import imageio.v2 as imageio
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt
import glob

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

def extract_crops_from_set(dataset, folder=None, channels=None, exclude_edgecases=True):
    # exclude_edgecases: exclude crops that are on the edge of the image

    crop_files = glob.glob(folder + '/*.tif')
    for f in crop_files:
        os.remove(f)

    transform, individual_transform = dataset.transform, dataset.individual_transform
    dataset.transform, dataset.individual_transform = None, None

    padding = 30
    g = 100**(1/padding)

    for sample in tqdm(dataset, leave=False, desc='Extracting crops'): # loop over batches
        # image = batch['images'][i]
        # bboxes = batch['boxes'][i].tolist()
        # mask_3d = batch['masks_3d'][i]
        # bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]
        # filepath = batch['file_paths'][i]

        image = sample['image']
        bboxes = [[int(coord) for coord in bbox] for bbox in sample['boxes']]
        mask_3d = sample['mask_3d']
        filepath = sample['file_path']

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
            # dist = gaussian(dist, sigma=10)
            continuous_mask = g**-dist # 0 = 1, all other values are between 0 and 1. PADDING pixels away = 0.01
            # continuous_mask = gaussian(continuous_mask, sigma=1)

            cropped_image = image[channels, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            crop = cropped_image * continuous_mask

            if folder: 
                path = '{}_{}x{}.tif'.format(os.path.join(folder, Path(filepath).stem), coords[0], coords[1])
                crop = [crop[c, :, :].numpy() for c in range(crop.shape[0])] + [mask.numpy()] # format from 3d array to list of 2d arrays
                imageio.mimwrite(path, crop)

    dataset.transform, dataset.individual_transform = transform, individual_transform

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
def mask_3d_to_2d(mask_3d, max_overlap=0.5):
    # initialize 2d array with zeros
    mask_2d = np.zeros(mask_3d.shape[1:], dtype=np.int32)
    
    # sort 3d array by descending size # TODO check if this works
    # layer_sizes = np.sum(mask_3d, axis=(1, 2))
    # sorted_indices = np.argsort(layer_sizes)[::-1]
    # sorted_array = mask_3d[sorted_indices]

    for i in range(mask_3d.shape[0]):
        # create a mask for the current layer
        # mask = mask_3d[i, :, :]
        
        # compare overlap of current layer with 2d array
        intersection = np.sum(np.logical_and(mask_3d[i, :, :], mask_2d))
        mask_sum = np.sum(mask_3d[i, :, :])
        overlap = intersection / mask_sum if mask_sum > 0 else 0
        
        # if the IoU is less than the threshold, add the layer to the output array
        if overlap <= max_overlap:
            mask_2d[mask_3d[i, :, :]] = i + 1
    return mask_2d


def surround_true_values(original_mask, N):
    # Ensure N is a positive integer
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")

    # Create a copy of the original mask
    new_mask = original_mask.copy()

    # Get the shape of the mask
    rows, cols = original_mask.shape

    # Iterate through the original mask
    for i in range(rows):
        for j in range(cols):
            if original_mask[i, j]:
                # Set the surrounding values to True within distance N
                for dx in range(-N, N + 1):
                    for dy in range(-N, N + 1):
                        new_x = i + dx
                        new_y = j + dy

                        # Check if the new coordinates are within the mask boundaries
                        if 0 <= new_x < rows and 0 <= new_y < cols:
                            new_mask[new_x, new_y] = True

    return new_mask