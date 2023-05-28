from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import imageio
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision import datapoints
from torchvision.ops import masks_to_boxes
import os

# 
def find_folder_range(image_paths, channels):
    folder_paths = set([os.path.dirname(path) for path in image_paths])
    range_dict = {fp: [(999999999, 0)]*len(channels) for fp in folder_paths}

    for path in image_paths:
        folder = os.path.dirname(path)
        for i, channel in enumerate(channels):
            range_dict[folder][i] = (min(range_dict[folder][i][0], np.min(imageio.mimread(path)[channel])), max(range_dict[folder][i][1], np.max(imageio.mimread(path)[channel])))
    print(range_dict)
    return range_dict

class MicroscopyDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, channels=[0, 1], transform=None, individual_transform=None, filter_empty=False, set_folder_ranges=False):
        # set_folder_max: we are using data with a multitude of intensity ranges caused by different microscopes and settings. When set to True, this parameter finds the highest pixel intensity in the folder of each respective image, which can be used for normalization purposes. Note that this normalization relies on the assumption that folders do not contain mixtures of different microscopes or microscope settings.

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.channels = channels
        self.transform = transform
        self.individual_transform = individual_transform
        self.filter_empty = filter_empty # if True, return None for images with empty masks

        self.folder_ranges = find_folder_range(image_paths, channels) if set_folder_ranges else None
        
    def __len__(self):
        return len(self.image_paths)
    
    def plot(self, original_image, labeled_mask, augmented_image, target, name, savepath):
        augmented_mask = torch.zeros_like(torch.Tensor(3, 1040, 1392)).type(torch.uint8)
        masks = target['masks'].type(torch.bool)
        augmented_mask = draw_segmentation_masks(augmented_mask, masks)
        augmented_mask = draw_bounding_boxes(augmented_mask, target['boxes'], width=3, colors='red')
        augmented_mask = torch.permute(augmented_mask, (1, 2, 0))

        # print('Original: blue: {} ... {}, red: {}, {}'.format(original_image[0].min(), original_image[0].max(), 
        #                                                     original_image[1].min(), original_image[1].max()))
        # print('Normalized: blue: {} ... {}, red: {}, {}'.format(augmented_image[0].min(), augmented_image[0].max(), 
        #                                                     augmented_image[1].min(), augmented_image[1].max()))

        fig, axs = plt.subplots(2, 3, figsize=(60,30), sharex=True, sharey=True)
        aspect = 'auto'
        axs[0, 0].imshow(original_image[0], aspect=aspect, cmap='Blues')
        axs[0, 1].imshow(original_image[1], aspect=aspect, cmap='hot')
        axs[0, 2].imshow(labeled_mask, aspect=aspect)
        axs[1, 0].imshow(augmented_image[0], aspect=aspect, cmap='Blues')
        axs[1, 1].imshow(augmented_image[1], aspect=aspect, cmap='hot')
        axs[1, 2].imshow(augmented_mask, aspect=aspect)
        fig.tight_layout()
        if savepath:
            plt.savefig(savepath)
        else:
            plt.show()

    def __getitem__(self, idx, savepath=None):
        image_filepath = self.image_paths[idx]
        image = np.array(list(map(imageio.mimread(image_filepath).__getitem__, self.channels))).astype(np.float64)
        original_image = image

        mask_filepath = self.mask_paths[idx]
        labeled_mask = np.array(Image.open(mask_filepath).convert('L')).astype(np.int32)

        image = torch.Tensor(image)
        labeled_mask = torch.Tensor(labeled_mask)

        labels = torch.unique(labeled_mask)[1:]
        masks = datapoints.Mask(labeled_mask == labels.view(-1, 1, 1))
        labels = torch.ones_like(labels, dtype=torch.int64)
        bounding_boxes = datapoints.BoundingBox(masks_to_boxes(masks), format=datapoints.BoundingBoxFormat.XYXY, spatial_size=image.shape[-2:])

        # Apply transformations, lastly the normalization augmentation.
        if self.individual_transform:
            for c in range(image.shape[0]):
                channel = torch.unsqueeze(image[c,:,:], 0)
                channel = self.individual_transform(channel)
                channel = torch.squeeze(channel)
                image[c,:,:] = channel

        if self.transform:
            image, bounding_boxes, masks, labels = self.transform(image, bounding_boxes, masks, labels)
            
        # For each bounding boxes, check if 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H. 
        # If not, then remove that mask in the list of bboxes, masks and labels.
        verify_bbox = lambda bbox: ((0 <= bbox[0] < bbox[2] <= labeled_mask.shape[1] and 0 <= bbox[1] < bbox[3] <= labeled_mask.shape[0]).item())
        bbox_validity = torch.Tensor([verify_bbox(bounding_boxes[i,:].type(torch.int32)) for i in range(bounding_boxes.shape[0])])==True
        
        masks = datapoints.Mask(masks[bbox_validity])
        bounding_boxes = datapoints.BoundingBox(bounding_boxes[bbox_validity], format=datapoints.BoundingBoxFormat.XYXY, spatial_size=image.shape[-2:])
        labels = labels[bbox_validity]

        target = {'boxes': bounding_boxes, 'labels': labels, 'masks': masks}
        name = Path(image_filepath).stem

        if savepath:

            self.plot(original_image, labeled_mask, image, target, name, savepath=None)

        return image, target, labeled_mask, name

    def get_path_max(self, idx):
        image_filepath = self.image_paths[idx]
        image = np.array(list(map(imageio.mimread(image_filepath).__getitem__, self.channels))).astype(np.float)
        img_max = self.max_intensity_dict[os.path.dirname(image_filepath)]
        return image, img_max

