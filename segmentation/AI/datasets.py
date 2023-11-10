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
    def __init__(self, image_paths, channels, mask_paths=None, hepatocyte_mask_paths=None, groups=None, transform=None, individual_transform=None, filter_empty=False, folder_normalize=True, rescale_img=None, rescale_mask=None):
        # set_folder_max: we are using data with a multitude of intensity ranges caused by different microscopes and settings. When set to True, this parameter finds the highest pixel intensity in the folder of each respective image, which can be used for normalization purposes. Note that this normalization relies on the assumption that folders do not contain mixtures of different microscopes or microscope settings.

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.hepatocyte_mask_paths = hepatocyte_mask_paths
        self.channels = channels
        self.groups = groups
        self.transform = transform
        self.individual_transform = individual_transform
        self.filter_empty = filter_empty # if True, return None for images with empty masks

        self.folder_normalize = folder_normalize
        # self.folder_means, self.folder_SDs = data_utils.find_folder_mean_and_SD(image_paths, channels) if folder_normalize else (None, None)
        # print(self.folder_means, self.folder_SDs)

        self.folder_ranges = data_utils.find_folder_range(image_paths, channels) if folder_normalize else None
        # print(self.folder_ranges)

        self.rescale_img = rescale_img
        self.rescale_mask = rescale_mask
        self.original_sizes = [None]*len(image_paths)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = {}
        sample_channels = self.channels[idx]
        image_filepath = self.image_paths[idx]

        if sample_channels:
            image = np.array(list(map(imageio.mimread(image_filepath, memtest=False).__getitem__, sample_channels))).astype(np.float64)
            sample.update({'original_image': image, 'original_size': image.shape[1:]})
            image = torch.Tensor(image)

        # Check for rescaling and verify image is not already the correct shape
        if self.rescale_img and image.shape[1:] != self.rescale_img:
            image = torch.tensor([data_utils.resize(image[i,:,:].numpy(), shape=self.rescale_img) for i in range(image.shape[0])])

        # if masks files are supplied, then compute corresponding labels, bboxes and masks.
        if self.mask_paths:
            mask_filepath = self.mask_paths[idx]
            mask_2d = np.array(Image.open(mask_filepath)).astype(np.int32)

            if self.rescale_mask and mask_2d.shape != self.rescale_mask:
                mask_2d = data_utils.resize_mask(mask_2d, shape=self.rescale_mask)

            mask_2d = torch.Tensor(mask_2d)

            object_ids = torch.unique(mask_2d)[1:]
            mask_3d = datapoints.Mask(mask_2d == object_ids.view(-1, 1, 1))
            labels = torch.ones_like(object_ids, dtype=torch.int64)
            bounding_boxes = datapoints.BoundingBox(masks_to_boxes(mask_3d), format=datapoints.BoundingBoxFormat.XYXY, spatial_size=mask_2d.shape)

        # Normalize folders
        if self.folder_normalize and sample_channels:
            folder = os.path.dirname(image_filepath)
            for i, channel in enumerate(sample_channels):
                old_range = self.folder_ranges[folder][str(channel)]
                image[i,:,:] = data_utils.rescale(image[i,:,:], old_range=old_range, new_range=(0, 1))
                # folder_mean = self.folder_means[folder][i]
                # folder_SD = self.folder_SDs[folder][i]
                # image[i,:,:] = data_utils.normalize(image[i,:,:], folder_mean, folder_SD)

        # Apply transformations
        if self.individual_transform:
            for i, channel in enumerate(sample_channels):
                if self.individual_transform[i]:
                    channel_img = torch.unsqueeze(image[i,:,:], 0)
                    augmented_channel_img = self.individual_transform[i](channel_img)
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
            bounding_boxes = datapoints.BoundingBox(bounding_boxes, format=datapoints.BoundingBoxFormat.XYXY, spatial_size=mask_2d.shape)
            sample.update({'original_mask_2d': mask_2d})
            mask_2d = torch.from_numpy(mask_utils.mask_3d_to_2d(mask_3d.data.cpu().numpy()))

            sample.update({'mask_2d': mask_2d, 'boxes': bounding_boxes, 'labels': labels, 'mask_3d': mask_3d})

        sample.update({'file_path': image_filepath})
        sample.update({'image': image}) if sample_channels else None
        sample.update({'group': self.groups[idx]}) if self.groups else None

        if self.hepatocyte_mask_paths:
            hepatocyte_mask = np.array(Image.open(self.hepatocyte_mask_paths[idx])).astype(np.int32)
            sample.update({'hepatocyte_mask': hepatocyte_mask})
        return sample
    