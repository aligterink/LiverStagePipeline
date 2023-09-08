import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

from utils import data_utils, mask_utils

import torch
import numpy as np
from PIL import Image
import imageio
from torch.utils.data import Dataset
from torchvision import datapoints
from torchvision.ops import masks_to_boxes

class MicroscopyDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, channels=[0, 1], groups=None, transform=None, individual_transform=None, filter_empty=False, folder_normalize=True):
        # set_folder_max: we are using data with a multitude of intensity ranges caused by different microscopes and settings. When set to True, this parameter finds the highest pixel intensity in the folder of each respective image, which can be used for normalization purposes. Note that this normalization relies on the assumption that folders do not contain mixtures of different microscopes or microscope settings.

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channels = channels
        self.groups = groups
        self.transform = transform
        self.individual_transform = individual_transform
        self.filter_empty = filter_empty # if True, return None for images with empty masks

        self.folder_normalize = folder_normalize
        # self.folder_ranges = data_utils.find_folder_range(image_paths, channels) if folder_normalize else None
        self.folder_means, self.folder_SDs = data_utils.find_folder_mean_and_SD(image_paths, channels) if folder_normalize else (None, None)
        
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
            mask_2d = np.array(Image.open(mask_filepath).convert('L')).astype(np.int32)
            mask_2d = torch.Tensor(mask_2d)

            object_ids = torch.unique(mask_2d)[1:]
            mask_3d = datapoints.Mask(mask_2d == object_ids.view(-1, 1, 1))
            labels = torch.ones_like(object_ids, dtype=torch.int64)
            bounding_boxes = datapoints.BoundingBox(masks_to_boxes(mask_3d), format=datapoints.BoundingBoxFormat.XYXY, spatial_size=image.shape[-2:])

        # Normalize folders, scale min-max to 0-1. Min and max are found per channel and for each folder, not for each image.
        if self.folder_normalize:
            folder = os.path.dirname(image_filepath)
            for i, channel in enumerate(self.channels):
                # old_range = self.folder_ranges[folder][channel]
                # image[i,:,:] = data_utils.normalize(image[i,:,:], old_range=old_range, new_range=(0, 1))
                folder_mean = self.folder_means[folder][channel]
                folder_SD = self.folder_SDs[folder][channel]
                image[i,:,:] = data_utils.normalize(image[i,:,:], folder_mean, folder_SD)

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
                image, bounding_boxes, mask_3d, labels = self.transform(image, bounding_boxes, mask_3d, labels)
            else:
                image = self.transform(image)
        
        if self.mask_paths:
            # For each bounding box, check if 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H. 
            # If not, then remove that mask in the list of bboxes, masks and labels.
            verify_bbox_func = lambda bbox: ((0 <= bbox[0] < bbox[2] <= mask_2d.shape[1] and 0 <= bbox[1] < bbox[3] <= mask_2d.shape[0]).item())
            bbox_validity = torch.Tensor([verify_bbox_func(bounding_boxes[i,:].type(torch.int32)) for i in range(bounding_boxes.shape[0])])==True
            
            # Remove invalid masks, bboxes and labels
            mask_3d = mask_3d[bbox_validity]
            bounding_boxes = bounding_boxes[bbox_validity]
            labels = labels[bbox_validity]

            mask_3d = datapoints.Mask(mask_3d)
            bounding_boxes = datapoints.BoundingBox(bounding_boxes, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=image.shape[-2:])
            sample.update({'original_mask_2d': mask_2d})
            mask_2d = torch.from_numpy(mask_utils.mask_3d_to_2d(mask_3d.data.cpu().numpy()))

            sample.update({'mask_2d': mask_2d, 'boxes': bounding_boxes, 'labels': labels, 'mask_3d': mask_3d})

        sample.update({'image': image, 'file_path': image_filepath})
        sample.update({'group': self.groups[idx]}) if self.groups else None

        return sample
    