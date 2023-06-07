import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[0] + 'LiverStagePipeline')

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

from utils import data_utils

import torch
import numpy as np
from PIL import Image
import imageio
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datapoints
from torchvision.ops import masks_to_boxes

class MicroscopyDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, channels=[0, 1], transform=None, individual_transform=None, filter_empty=False, folder_normalize=True):
        # set_folder_max: we are using data with a multitude of intensity ranges caused by different microscopes and settings. When set to True, this parameter finds the highest pixel intensity in the folder of each respective image, which can be used for normalization purposes. Note that this normalization relies on the assumption that folders do not contain mixtures of different microscopes or microscope settings.

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channels = channels
        self.transform = transform
        self.individual_transform = individual_transform
        self.filter_empty = filter_empty # if True, return None for images with empty masks

        self.folder_normalize = folder_normalize
        self.folder_ranges = data_utils.find_folder_range(image_paths, channels) if folder_normalize else None
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = {}

        image_filepath = self.image_paths[idx]
        image = np.array(list(map(imageio.mimread(image_filepath, memtest=False).__getitem__, self.channels))).astype(np.float64)
        image = torch.Tensor(image)

        # if masks files are supplied, then compute corresponding labels, bboxes and masks.
        if self.mask_paths:
            mask_filepath = self.mask_paths[idx]
            labeled_mask = np.array(Image.open(mask_filepath).convert('L')).astype(np.int32)
            labeled_mask = torch.Tensor(labeled_mask)

            labels = torch.unique(labeled_mask)[1:]
            masks = datapoints.Mask(labeled_mask == labels.view(-1, 1, 1))
            labels = torch.ones_like(labels, dtype=torch.int64)
            bounding_boxes = datapoints.BoundingBox(masks_to_boxes(masks), format=datapoints.BoundingBoxFormat.XYXY, spatial_size=image.shape[-2:])

        # Apply transformations
        if self.individual_transform:
            for i, channel in enumerate(self.channels):
                if self.individual_transform[channel]:
                    channel_img = torch.unsqueeze(image[i,:,:], 0)
                    augmented_channel_img = self.individual_transform[channel](channel_img)
                    augmented_channel_img = torch.squeeze(augmented_channel_img)
                    image[i,:,:] = augmented_channel_img

        if self.transform:
            if self.mask_paths:
                image, bounding_boxes, masks, labels = self.transform(image, bounding_boxes, masks, labels)
            else:
                image = self.transform(image)

        if self.folder_normalize:
            folder = os.path.dirname(image_filepath)
            for i, channel in enumerate(self.channels):
                old_range = self.folder_ranges[folder][channel]
                image[i,:,:] = data_utils.normalize(image[i,:,:], old_range=old_range, new_range=(0, 1))
        
        if self.mask_paths:
            # For each bounding boxes, check if 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H. 
            # If not, then remove that mask in the list of bboxes, masks and labels.
            verify_bbox = lambda bbox: ((0 <= bbox[0] < bbox[2] <= labeled_mask.shape[1] and 0 <= bbox[1] < bbox[3] <= labeled_mask.shape[0]).item())
            bbox_validity = torch.Tensor([verify_bbox(bounding_boxes[i,:].type(torch.int32)) for i in range(bounding_boxes.shape[0])])==True
            
            masks = datapoints.Mask(masks[bbox_validity])
            bounding_boxes = datapoints.BoundingBox(bounding_boxes[bbox_validity], format=datapoints.BoundingBoxFormat.XYXY, spatial_size=image.shape[-2:])
            labels = labels[bbox_validity]
            sample.update({'mask_2d': labeled_mask, 'boxes': bounding_boxes, 'labels': labels, 'masks': masks})

        sample.update({'image': image, 'file_path': image_filepath})

        return sample
    
# class MicroscopyDatasetNoTarget(Dataset):
#     def __init__(self, image_paths, channels=[0, 1], transform=None, individual_transform=None, set_folder_ranges=False):
#         # set_folder_ranges: we are using data with a multitude of intensity ranges caused by different microscopes and settings. When set to True, this parameter finds the highest pixel intensity in the folder of each respective image, which can be used for normalization purposes. Note that this normalization relies on the assumption that folders do not contain mixtures of different microscopes or microscope settings.

#         self.image_paths = image_paths
#         self.channels = channels
#         self.transform = transform
#         self.individual_transform = individual_transform
#         self.folder_ranges = find_folder_range(image_paths, channels) if set_folder_ranges else None
        
#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_filepath = self.image_paths[idx]
#         image = np.array(list(map(imageio.mimread(image_filepath, memtest=False).__getitem__, self.channels))).astype(np.float32)

#         image = torch.Tensor(image)

#         # Apply transformations
#         if self.individual_transform:
#             for c in range(image.shape[0]):
#                 channel = torch.unsqueeze(image[c,:,:], 0)
#                 channel = self.individual_transform(channel)
#                 channel = torch.squeeze(channel)
#                 image[c,:,:] = channel

#         # if self.transform:
#         #     image = self.transform(image)[0]
#         image = torch.Tensor(np.interp(image, (0, 4095), (-1, +1)))
            
#         name = Path(image_filepath).stem
        

#         return image, name
