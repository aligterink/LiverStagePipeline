import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[0] + 'LiverStagePipeline')

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms

import torch
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
from segmentation.AI.augmentations import *

def v1(crops_folder, watershed_crop_folder):
    train_individual_transform = transforms.Compose([
        CustomNormalizeSingleChannel(),

        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.4),
        # transforms.RandomAdjustSharpness(5, p=0.3),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1,8.0))], p=0.4),

        RevertNormalizationSingleChannel()
    ])

    train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.9, 1.3), shear=15)], p=0.3),
        transforms.RandomApply([transforms.RandomResizedCrop(size=(1040, 1392), antialias=True)], p=0.3),

        # # HSP copy paste
        # transforms.RandomApply([RandomCopyPaste(folder=crops_folder,
        #                 channel_paste=[(1, 1)], min_objs=40, max_objs=80,
        #                 transform=transforms.Compose([ # transformations for the pasted crops
        #                     transforms.RandomRotation(degrees=(0, 360), expand=True),
        #                     transforms.RandomApply([transforms.ColorJitter(brightness=(0.4, 1.2), contrast=0.2, saturation=0.2, hue=0.2)], p=0.6),
        #                     # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(9), sigma=(12))], p=1),
        #                     # transforms.RandomApply([transforms.RandomResize(min_size=8, max_size=40)], p=0.5)
        #                 ]), individual_transform=
        #                     {1:transforms.Compose([
        #                         GaussianNoise(variance=0.2),
        #                         GaussianFilter()
        #                         # transforms.GaussianBlur(kernel_size=13, sigma=4)
        #                         # SizeAdjustedPytorchGaussianBlur()
        #                         # CustomBlur(0.3)
        #                     ])}
        #                 )], p=0.7),

        # # HSP copy paste, blur DAPI
        # transforms.RandomApply([RandomCopyPaste(folder=crops_folder,
        #                 channel_paste=[(0, 0), (1, 1)], min_objs=10, max_objs=20, min_crop_size=20,
        #                 transform=transforms.Compose([ # transformations for the pasted crops
        #                     transforms.RandomRotation(degrees=(0, 360), expand=True),
        #                     # transforms.RandomApply([transforms.ColorJitter(brightness=(0.4, 1.2), contrast=0.2, saturation=0.2, hue=0.2)], p=0.6),
        #                 ]), individual_transform=
        #                     {0: transforms.Compose([
        #                         MakeMedian(multiplier=1.5),
        #                         GaussianNoise(variance=0.3),
        #                         GaussianFilter(sigma=1.1),
        #                         CustomBlur(0.05)
        #                     ]),
        #                     1: transforms.Compose([
        #                         GaussianNoise(0.35),
        #                         GaussianFilter(sigma=1.1),
        #                         CustomBlur(0.05)
        #                     ])}
        #                 )], p=0.5),

        # transforms.RandomApply([ClusterCopyPaste(folder=crops_folder, min_objs_per_obj=1, max_objs_per_obj=4,
        #                 transform=transforms.Compose([ # transformations for the pasted crops
        #                     transforms.RandomRotation(degrees=(0, 360), expand=True),
        #                     # transforms.RandomApply([transforms.RandomResize(min_size=20, max_size=60)], p=1)
        #                 ]),
        #                 individual_transform=
        #                 {0: transforms.Compose([ # individual transformations for the pasted crops
        #                     # CustomNormalizeSingleChannel(),
        #                     # transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2)], p=1),
        #                     # RevertNormalizationSingleChannel()
        #                     CustomBlur(0.05)

        #                 ]),
        #                 1: transforms.Compose([ # individual transformations for the pasted crops
        #                     # CustomNormalizeSingleChannel(),
        #                     # transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2)], p=1),
        #                     # transforms.RandomApply([CustomGaussianBlur()], p=1),
        #                     # RevertNormalizationSingleChannel()
        #                     GaussianNoise(0.35),
        #                     GaussianFilter(sigma=1.1),
        #                     CustomBlur(0.1)
        #                 ])
        #                 }
        #             )], p=0.5),
        
        # transforms.RandomApply([GaussianNoise()], p=0.3),

        CustomNormalize()
    ])

    test_transform = transforms.Compose([
        CustomNormalize()
    ])

    return train_transform, train_individual_transform, test_transform