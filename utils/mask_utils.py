from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
import exploration.get_dataloaders as get_dataloaders
import torchvision.transforms as transforms
import numpy as np
import torch
import os
from pathlib import Path
import imageio.v2 as imageio

def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return torch.tensor([xmin, ymin, xmax, ymax])

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
        crop = [crop[c, :, :] for c in range(crop.shape[0])] # format from 3d array to list of 2d arrays
        imageio.mimwrite(path, crop)


def extract_crops_from_loader(loader, folder=None, channels=None, exclude_edgecases=True):
    # exclude_edgecases: exclude crops that are on the edge of the image

    for images, targets, _, filenames in loader:
        for image, target, filename in zip(images, targets, filenames):
            bboxes = target['boxes'].tolist()
            masks = target['masks']
            bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]
            crops = get_crops(image, bboxes, masks, channels=channels)

            if folder:
                path_prefix = os.path.join(folder, Path(filename).stem)
                store_crops(crops=crops, bboxes=bboxes, path_prefix=path_prefix, exclude_edgecases=exclude_edgecases, image=image)

def extract_non_overlapping_crops_from_loader(loader, val_loader, folder=None, channels=None, exclude_edgecases=True):
    # exclude_edgecases: exclude crops that are on the edge of the image

    nono_bboxes = {}

    for images, targets, _, filenames in val_loader:
        for image, target, filename in zip(images, targets, filenames):
            nono_bboxes[filename] = target['boxes'].tolist()


    for images, targets, _, filenames in loader:
        for image, target, filename in zip(images, targets, filenames):
            overlap_mask = multiple_bboxes_to_single_mask(bboxes, (image.shape[2], image.shape[1]))

            bboxes = target['boxes'].tolist()
            masks = target['masks']
            bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]
            crops = get_crops(image, bboxes, masks, channels=channels)

            if folder:
                path_prefix = os.path.join(folder, Path(filename).stem)
                store_crops(crops=crops, bboxes=bboxes, path_prefix=path_prefix, exclude_edgecases=exclude_edgecases, image=image)

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders.v2(train_transform=None, train_individual_transform=None, test_transform=None, batch_size=4, num_workers=4)
    extract_crops_from_loader(train_loader, folder="/home/anton/Documents/microscopy_data/crops")

    # p = "/home/anton/Documents/microscopy_data/3564_annotated_subset/Annie_subset_tiffs/NF135_D5/2019006c_D5_135_exp2_hsp_series_2_TileScan_001.tif"
    # im = imageio.imread(p)
    # imageio.mimwrite("/home/anton/Documents/results/figures/testingtif.tif", im)











