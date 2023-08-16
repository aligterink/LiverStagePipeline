from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms
from torchvision.transforms import transforms as transforms_v1

import torch
import torch.nn.functional as F
import numpy as np
import os
import imageio.v2 as imageio
from torchvision import datapoints
from torchvision.ops import masks_to_boxes
from skimage.exposure import match_histograms

import os, random
import torchvision.transforms.functional as TF
import utils.mask_utils as mask_utils
import math
import glob
from skimage import filters
from skimage import io, filters


# class GaussianNoise:
#     def __call__(self, sample):
#         img = sample[0]
#         img = img * (1 + (0.1**0.5)*torch.randn_like(img))
#         return (img, *sample[1:])
    
class GaussianNoise:
    def __init__(self, variance=0.1):
        self.variance = variance

    def __call__(self, sample):
        return sample * (1 + (self.variance**0.5)*torch.randn_like(sample))
        
class RewrittenRandomResize:
    def __init__(self, min_multiplier, max_multiplier):
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
    
    def __call__(self, sample):
        min_size = (sample[0].shape[1] + sample[0].shape[2]) / 2 * self.min_multiplier
        max_size = (sample[0].shape[1] + sample[0].shape[2]) / 2 * self.max_multiplier
        t = transforms.RandomResize(min_size=min_size, max_size=max_size)

class Aug060823:
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample
    
# TODO implement padding here instead of doing it by hard coding
# class Pad:
#     def __init__(self, pixels):
#         self.pixels = pixels
    
#     def __call__(self, img: np.array, mask: np.array):
#         pass

# class CustomNormalize:
#     def __call__(self, sample):
#         img = sample[0]
#         img = torch.Tensor(np.interp(img, (0, 4095), (-1, +1)))
#         # assert img.max() <= 4095, "Max pixel value in image is larger than 4095. Normalization will not work."
#         return (img, *sample[1:])
    
# class CustomNormalize:
#     def __init__(self, range_dict):
#         self.range_dict = range_dict

#     def __call__(self, sample):
#         img = sample[0]
#         img = torch.Tensor(np.interp(img, (0, 4095), (-1, +1)))
#         # assert img.max() <= 4095, "Max pixel value in image is larger than 4095. Normalization will not work."
#         return (img, *sample[1:])
    
class CustomNormalizeSingleChannel:
    def __call__(self, img):
        return img / 10000
    
class RevertNormalizationSingleChannel:
    def __call__(self, img):
        return img * 10000
    
class GaussianFilter:
    def __init__(self, sigma=1.5):
        self.sigma = sigma
    
    def __call__(self, sample):
        return torch.from_numpy(filters.gaussian(sample, sigma=self.sigma)) # blur with 2D Gaussian

class SizeAdjustedPytorchGaussianBlur:
    def __call__(self, sample):
        kernel_size = max(3, round(sum(sample.shape)/2 / 5))
        sigma = min(0.1, sum(sample.shape)/2/20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(sample)

class MakeMedian:
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier

    def __call__(self, sample):
        median = torch.median(sample[sample != 0]) 
        sample[sample != 0] = median * self.multiplier
        return sample
    
class CustomBlur: 
    def __init__(self, blur=0.2):
        self.blur = blur

    def __call__(self, sample):
        image = sample


        # Determine the image dimensions
        height, width = image.shape[-2:]

        # Define the center and radius of the region of interest
        center_x = width // 2
        center_y = height // 2
        radius = min(center_x, center_y)

        # Create a mask with a gradient that fades from white to black from the center to the edge
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        mask = mask.to(torch.float32)

        # Define the sigma (standard deviation) for blurring
        sigma = radius * self.blur  # Adjust the factor (0.2) as desired for the level of blur

        # Apply Gaussian blur to the image using the mask and sigma
        kernel_size = int(sigma * 6 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred_image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
        result = mask.unsqueeze(0).unsqueeze(0) * blurred_image + (1 - mask).unsqueeze(0).unsqueeze(0) * image
        result = result.squeeze(0)

        return result


    
# def normalizeCrop(reference_intensities, crop, channel_paste, mask):
#     if len(reference_intensities[0]) > 0:
#         # Perform histogram matching between every pasted cell and the combined intensities of all cells already there
#         for i,cp in enumerate(channel_paste):
#             this_channel = crop[cp[0]]
#             values_to_match = np.array(this_channel[mask])
#             matched_values = match_histograms(values_to_match, reference_intensities[i].numpy())
#             crop[cp[0]][mask] = matched_values
#     return crop

def normalizeCrop(reference_intensities, crop, channel_paste, mask):
    if len(reference_intensities[0]) > 0:
        # Perform histogram matching between every pasted cell and the combined intensities of all cells already there
        for i,cp in enumerate(channel_paste):
            this_channel = crop[cp[0],:,:]
            values_to_match = this_channel[mask].numpy()
            matched_values = match_histograms(values_to_match, reference_intensities[i].numpy())
            crop[cp[0],:,:][mask] = torch.tensor(matched_values)
    return crop

class RandomCopyPaste:
    def __init__(self, folder, channel_paste=[(1, 1)], min_objs=50, max_objs=100, transform=None, individual_transform=None, min_crop_size=6, max_crop_size=70):
        # channel_paste: a list of tuples where the 0th element is the channel of the crop and the 1st element is the image channel the crop is pasted onto.

        self.folder = folder
        self.channel_paste = channel_paste
        self.min_objs = min_objs
        self.max_objs = max_objs
        self.transform = transform
        self.individual_transform = individual_transform

        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size

    def __call__(self, sample):
        print(type(sample[0]), )
        img = sample[0]
        img_height, img_width = img.shape[1:]
        reference_intensities = [img[cp[0],:,:][torch.any(sample[2], axis=0)] for cp in self.channel_paste]

        num_crops = np.random.randint(self.min_objs, self.max_objs)
        crop_paths = [os.path.join(self.folder, random.choice(os.listdir(self.folder))) for i in range(num_crops)]

        for crop_path in crop_paths:
            crop = torch.Tensor(imageio.mimread(crop_path))

            # Verify that crops falls in the specified size range
            if not self.min_crop_size <= crop.shape[1] <= self.max_crop_size or not self.min_crop_size <= crop.shape[2] <= self.max_crop_size:
                continue

            # crop = np.pad(crop, ((0,0),(2,2),(2,2)))
            mask = torch.any(crop, axis=0).numpy()

            # crop = torch.tensor(crop)

            # transform crop
            if self.individual_transform:
                for c in [cp[0] for cp in self.channel_paste]:
                    channel = self.individual_transform[c](crop[c,:,:])
                    crop[c,:,:] = channel
            
            if self.transform:
                crop, mask = self.transform(crop, datapoints.Mask(mask))

            # crop = normalizeCrop(reference_intensities, crop, self.channel_paste, mask)

            # Get dimensions of crop
            crop_height, crop_width = crop.shape[1], crop.shape[2]

            # Determine random pasting area
            min_x = np.random.randint(0, img_width - crop_width)
            min_y = np.random.randint(0, img_height - crop_height)
            max_x, max_y = min_x + crop_width, min_y + crop_height

            # Paste
            for paste in self.channel_paste:
                img[paste[1], min_y:max_y, min_x:max_x] = torch.maximum(img[paste[1], min_y:max_y, min_x:max_x], torch.Tensor(crop[paste[0],:,:]))
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

        # self.crop_paths = [os.path.join(folder, p) for p in os.listdir(folder)]
        # self.total_crops = len(os.listdir(folder))

    def __call__(self, sample):
        img = sample[0]
        img_bboxes = sample[1]
        img_masks = sample[2]
        num_objs, dont_continue = 0, False

        img_height, img_width = img.shape[1:]
        reference_intensities = [img[cp[0],:,:][torch.any(sample[2], axis=0)] for cp in self.channel_paste]

        for cell_index in range(img_masks.shape[0]):
            if dont_continue:
                break

            # num_pastes = np.random.randint(self.min_objs_per_obj, self.max_objs_per_obj)
            # paste_paths = np.random.choice(self.crop_paths, size=num_pastes)

            num_crops = np.random.randint(self.min_objs_per_obj, self.max_objs_per_obj)
            paste_paths = [os.path.join(self.folder, random.choice(os.listdir(self.folder))) for i in range(num_crops)]
            
            for path in paste_paths:
                paste_cell_crop = np.array(imageio.mimread(path))
                # paste_cell_crop = np.pad(paste_cell_crop, ((0,0),(5,5),(5,5)))
                paste_cell_mask = datapoints.Mask(np.any(paste_cell_crop, axis=0))
                paste_cell_crop = torch.tensor(paste_cell_crop)

                if not self.min_crop_size <= paste_cell_crop[0].shape[0] <= self.max_crop_size or not self.min_crop_size <= paste_cell_crop[0].shape[1] <= self.max_crop_size:
                    continue

                if self.transform:
                    paste_cell_crop, paste_cell_mask = self.transform(paste_cell_crop, paste_cell_mask)

                if self.individual_transform:
                    for c in [cp[0] for cp in self.channel_paste]:
                        channel = self.individual_transform[c](torch.unsqueeze(paste_cell_crop[c,:,:], 0))
                        channel = torch.squeeze(channel)
                        paste_cell_crop[c,:,:] = channel

                paste_cell_crop = normalizeCrop(reference_intensities, paste_cell_crop, self.channel_paste, paste_cell_mask)

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

                # Update the image
                for paste in self.channel_paste:
                    img[paste[1], paste_cell_min_y:paste_cell_max_y, paste_cell_min_x:paste_cell_max_x] = torch.maximum(img[paste[1], paste_cell_min_y:paste_cell_max_y, paste_cell_min_x:paste_cell_max_x], torch.Tensor(paste_cell_crop[paste[0],:,:]))
                
                # Add the bbox and mask of the crop
                img_bboxes = torch.cat((img_bboxes, masks_to_boxes(paste_cell_mask.unsqueeze(dim=0))), dim=0)
                img_masks = torch.cat((img_masks, paste_cell_mask.unsqueeze(dim=0)), dim=0)

                # Check if the maximum amount of cells have been pasted
                num_objs += 1
                if num_objs >= self.max_objs:
                    dont_continue = True
                    break

        return (img, img_bboxes, img_masks, torch.ones(size=(img_bboxes.shape[0],), dtype=torch.int64))
