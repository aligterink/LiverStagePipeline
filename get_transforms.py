from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms
from torchvision.transforms import transforms as transforms_v1

# import albumentations as A
import torch
import torch.nn.functional as F
import numpy as np
import os
import imageio.v2 as imageio
from torchvision import datapoints
from torchvision.ops import masks_to_boxes

import utils.mask_utils as mask_utils
import math
import glob

class GaussianNoise:
    def __call__(self, sample):
        img = sample[0]
        img = img * (1 + (0.1**0.5)*torch.randn_like(img))
        return (img, *sample[1:])

class CustomNormalize:
    def __call__(self, sample):
        img = sample[0]
        img = torch.Tensor(np.interp(img, (0, 4095), (-1, +1)))
        # assert img.max() <= 4095, "Max pixel value in image is larger than 4095. Normalization will not work."
        return (img, *sample[1:])
    
class CustomNormalizeSingleChannel:
    def __call__(self, img):
        return img / 10000
    
class RevertNormalizationSingleChannel:
    def __call__(self, img):
        return img * 10000

# NOTE: some transforms on the crops will not work given that they are applied on all channels of the crop
class RandomCopyPaste:
    def __init__(self, folder, channel_paste=[(1, 1)], min_objs=5, max_objs=10, transform=None):
        # channel_paste: a list of tuples where the 0th element is the channel of the crop and the 1st element is the image channel the crop is pasted onto.

        self.folder = folder
        self.channel_paste = channel_paste
        self.min_objs = min_objs
        self.max_objs = max_objs
        self.crop_paths = [os.path.join(folder, p) for p in os.listdir(folder)]
        self.total_crops = len(os.listdir(folder))
        self.transform = transform

    def __call__(self, sample):
        img = sample[0]
        img_height, img_width = img.shape[1:]

        num_crops = np.random.randint(self.min_objs, self.max_objs)
        crop_paths = np.random.choice(self.crop_paths, size=num_crops)
        crops = [imageio.mimread(p) for p in crop_paths]

        for crop in crops:
            # do transformation stuff
            if self.transform:
                crop = self.transform(crop)

            # Get dimensions of crop
            crop_height, crop_width = crop[0].shape

            # Determine random pasting area
            min_x = np.random.randint(0, img_width - crop_width)
            min_y = np.random.randint(0, img_height - crop_height)
            max_x, max_y = min_x + crop_width, min_y + crop_height

            # Paste
            for paste in self.channel_paste:
                img[paste[1], min_y:max_y, min_x:max_x] = torch.maximum(img[paste[1], min_y:max_y, min_x:max_x], torch.Tensor(crop[paste[0]]))

        return (img, *sample[1:])

class ClusterCopyPaste:
    def __init__(self, folder, channel_paste=[(0, 0), (1, 1)], min_objs_per_obj=1, max_objs_per_obj=4, max_objs=25, max_overlap=0.3, transform=None, individual_transform=None, min_crop_size=6, max_crop_size=70):
        # channel_paste: a list of tuples where the 0th element is the channel of the crop and the 1st element is the image channel the crop is pasted onto.

        self.folder = folder
        self.channel_paste = channel_paste
        self.min_objs_per_obj = min_objs_per_obj
        self.max_objs_per_obj = max_objs_per_obj
        self.max_objs = max_objs
        self.max_overlap = max_overlap
        self.transform = transform
        self.individual_transform = individual_transform
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size

        self.crop_paths = [os.path.join(folder, p) for p in os.listdir(folder)]
        self.total_crops = len(os.listdir(folder))

    def __call__(self, sample):
        img = sample[0]
        img_bboxes = sample[1]
        img_masks = sample[2]
        num_objs, dont_continue = 0, False

        img_height, img_width = img.shape[1:]

        for cell_index in range(img_masks.shape[0]):
            if dont_continue:
                break

            num_pastes = np.random.randint(self.min_objs_per_obj, self.max_objs_per_obj)
            paste_paths = np.random.choice(self.crop_paths, size=num_pastes)
            crops = [imageio.mimread(p) for p in paste_paths]

            for paste_cell_crop in crops:
                paste_cell_mask = datapoints.Mask(paste_cell_crop[0] != 0)
                paste_cell_crop = torch.tensor(paste_cell_crop)

                if not self.min_crop_size <= paste_cell_crop[0].shape[0] <= self.max_crop_size or not self.min_crop_size <= paste_cell_crop[0].shape[1] <= self.max_crop_size:
                    continue
                
                # do transformation stuff
                if self.individual_transform:
                    for c in [cp[0] for cp in self.channel_paste]:
                        channel = self.individual_transform(torch.unsqueeze(paste_cell_crop[c,:,:], 0))
                        channel = torch.squeeze(channel)
                        paste_cell_crop[c,:,:] = channel


                if self.transform:
                    paste_cell_crop, paste_cell_mask = self.transform(paste_cell_crop, paste_cell_mask)

                assert paste_cell_crop.shape[1:] == paste_cell_mask.shape, "ehuh"

                # Get dimensions of crop
                paste_cell_height, paste_cell_width = paste_cell_crop[0].shape

                # Select a random cell in the image and compute its width, height and central x,y coordinates
                lonely_cell_bbox = img_bboxes[cell_index, :].numpy()

                lonely_cell_width = lonely_cell_bbox[2] - lonely_cell_bbox[0]
                lonely_cell_height = lonely_cell_bbox[3] - lonely_cell_bbox[1]
                lonely_cell_x, lonely_cell_y = mask_utils.bbox_to_position(lonely_cell_bbox)

                # Select the mean, min and max of the x and y coordinates of the to-be-pasted cell
                paste_cell_x = lonely_cell_x + np.random.randint(-round(lonely_cell_width/2 + paste_cell_width/2), round(lonely_cell_width/2 + paste_cell_width/2))
                paste_cell_y = lonely_cell_y + np.random.randint(-round(lonely_cell_height/2 + paste_cell_height/2), round(lonely_cell_height/2 + paste_cell_height/2))
                
                # Make sure the paste cell does not fall outside of the image
                paste_cell_x = min(max(math.ceil(paste_cell_width/2) + 1, paste_cell_x), math.floor(img_width - paste_cell_width/2) - 1)
                paste_cell_y = min(max(math.ceil(paste_cell_height/2) + 1, paste_cell_y), math.floor(img_height - paste_cell_height/2) - 1)

                paste_cell_min_x = math.floor(paste_cell_x - (paste_cell_width/2))
                paste_cell_min_y = math.floor(paste_cell_y - (paste_cell_height/2))

                paste_cell_max_x = paste_cell_min_x + paste_cell_width
                paste_cell_max_y = paste_cell_min_y + paste_cell_height

                # Scale paste_cell_mask to the shape of the image
                zeros_img = torch.zeros(size=(img_height, img_width), dtype=torch.bool)
                zeros_img[paste_cell_min_y:paste_cell_max_y, paste_cell_min_x:paste_cell_max_x] = paste_cell_mask
                paste_cell_mask = zeros_img
                paste_cell_mask_area = torch.sum(paste_cell_mask)

                overlapping_cell_indices = [i for i in range(img_masks.shape[0]) if torch.any(torch.logical_and(paste_cell_mask, img_masks[i,:,:]))]
                overlapping_cell_intersections = [np.logical_and(paste_cell_mask, img_masks[i,:,:]) for i in overlapping_cell_indices]

                # Checks for every overlapping cell whether overlap/overlapping_cell_area > max_overlap or overlap/paste_cell_area > max_overlap. If one is true, then the cell is not pasted.
                if any([(torch.sum(intersection) / torch.sum(img_masks[i,:,:]) > self.max_overlap or torch.sum(intersection) / paste_cell_mask_area > self.max_overlap).item() for i,intersection in zip(overlapping_cell_indices, overlapping_cell_intersections)]):
                    continue

                # For all cells that overlap with the to-be-pasted cell, update the masks
                for i, overlapping_cell_index in enumerate(overlapping_cell_indices):
                    intersection = overlapping_cell_intersections[i]
                    y_indices, x_indices = np.where(intersection) # get the indices of the intersection

                    overlapping_cell_x, overlapping_cell_y = mask_utils.bbox_to_position(img_bboxes[overlapping_cell_index, :].numpy())

                    overlapping_cell_distances = np.sqrt((x_indices - overlapping_cell_x) ** 2 + (y_indices - overlapping_cell_y) ** 2)
                    paste_cell_distances = np.sqrt((x_indices - paste_cell_x) ** 2 + (y_indices - paste_cell_y) ** 2)

                    area_for_overlapping_cell = np.zeros_like(intersection, dtype=bool) # create a matrix of zeros
                    np.place(area_for_overlapping_cell, intersection, overlapping_cell_distances < paste_cell_distances)

                    area_for_paste_cell = np.zeros_like(intersection, dtype=bool) # create a matrix of zeros
                    np.place(area_for_paste_cell, intersection, paste_cell_distances <= overlapping_cell_distances)

                    # Update paste_cell- and overlapping_cell masks
                    paste_cell_mask[area_for_paste_cell] = True
                    paste_cell_mask[area_for_overlapping_cell] = False
                    img_masks[overlapping_cell_index,:,:][area_for_overlapping_cell] = True
                    img_masks[overlapping_cell_index,:,:][area_for_paste_cell] = False

                    # Update bounding box for both
                    img_bboxes[overlapping_cell_index,:] = masks_to_boxes(img_masks[overlapping_cell_index,:,:].unsqueeze(dim=0))

                # Update the image and all lists
                for paste in self.channel_paste:
                    img[paste[1], paste_cell_min_y:paste_cell_max_y, paste_cell_min_x:paste_cell_max_x] = torch.maximum(img[paste[1], paste_cell_min_y:paste_cell_max_y, paste_cell_min_x:paste_cell_max_x], torch.Tensor(paste_cell_crop[paste[0],:,:]))
                
                img_bboxes = torch.cat((img_bboxes, masks_to_boxes(paste_cell_mask.unsqueeze(dim=0))), dim=0)
                img_masks = torch.cat((img_masks, paste_cell_mask.unsqueeze(dim=0)), dim=0)
                num_objs += 1
                if num_objs >= self.max_objs:
                    dont_continue = True
                    break

        return (img, img_bboxes, img_masks, torch.ones(size=(img_bboxes.shape[0],), dtype=torch.int64))


def v1(crops_folder):
    train_individual_transform = transforms.Compose([
        CustomNormalizeSingleChannel(),

        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.4),
        transforms.RandomAdjustSharpness(5, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1,8.0))], p=0.4),

        RevertNormalizationSingleChannel()
    ])

    train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.9, 1.3), shear=15)], p=0.3),
        transforms.RandomApply([transforms.RandomResizedCrop(size=(1040, 1392), antialias=True)], p=0.3),

        transforms.RandomApply([RandomCopyPaste(folder=crops_folder,
                        channel_paste=[(1, 1)], min_objs=5, max_objs=30,
                        transform=transforms.Compose([ # transformations for the pasted crops
                            transforms.RandomRotation(degrees=(0, 360), expand=True),
                            transforms.RandomApply([transforms.ColorJitter(brightness=(0.4, 1.2), contrast=0.2, saturation=0.2, hue=0.2)], p=0.6),
                            # transforms.RandomAdjustSharpness(3, p=0.3),
                            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1,3.0))], p=0.3),
                            transforms.RandomApply([transforms.RandomResize(min_size=8, max_size=40)], p=0.5)
                        ]))], p=0.8),

        transforms.RandomApply([ClusterCopyPaste(folder=crops_folder, min_objs_per_obj=1, max_objs_per_obj=4,
                        transform=transforms.Compose([ # transformations for the pasted crops
                            transforms.RandomRotation(degrees=(0, 360), expand=True),
                            transforms.RandomApply([transforms.RandomResize(min_size=20, max_size=60)], p=1)
                        ]),
                        individual_transform=transforms.Compose([ # individual transformations for the pasted crops
                            CustomNormalizeSingleChannel(),

                            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2)], p=1),
                            # transforms.RandomAdjustSharpness(2, p=0.3),
                            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1,3.0))], p=0.4),

                            RevertNormalizationSingleChannel()
                        ])
                    )], p=0.5),
        
        transforms.RandomApply([GaussianNoise()], p=0.3),

        CustomNormalize()
    ])

    test_transform = transforms.Compose([
        CustomNormalize()
    ])

    return train_transform, train_individual_transform, test_transform