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

def v1(crops_folder):
    train_individual_transform = {
                    0: transforms.Compose([
                        # CustomNormalizeSingleChannel(),

                        # transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.4),
                        # transforms.RandomAdjustSharpness(4, p=0.3),
                        # # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1,6.0))], p=0.3),

                        # RevertNormalizationSingleChannel()
                        ]),
                    1: transforms.Compose([
                        # CustomNormalizeSingleChannel(),

                        # transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.4),
                        # transforms.RandomAdjustSharpness(4, p=0.3),
                        # # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1,6.0))], p=0.3),

                        # RevertNormalizationSingleChannel()
                        ]),
                    }

    train_transform = transforms.Compose([
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomHorizontalFlip(p=0.5),
        # # transforms.RandomApply([transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.9, 1.3), shear=15)], p=0.3),
        # transforms.RandomApply([transforms.RandomResizedCrop(size=(1040, 1392), antialias=True)], p=0.3),

        # # HSP copy paste
        # transforms.RandomApply([RandomCopyPaste(folder=crops_folder, channel_paste=[(1, 1)], min_objs=60, max_objs=100, min_crop_size=0, max_crop_size=99999999999999,
        #                 transform=transforms.Compose([ # transformations for the pasted crops
        #                     # transforms.RandomRotation(degrees=(0, 360), expand=True),
        #                     # transforms.RandomApply([transforms.ColorJitter(brightness=(0.4, 1.2), contrast=0.2, saturation=0.2, hue=0.2)], p=0.6),
        #                     # transforms.RandomApply([transforms.RandomResize(min_size=0.8, max_size=1.2)], p=1)
        #                 ]) #, individual_transform={1:transforms.Compose([])}
        #                 )], p=1),

        transforms.RandomApply([ClusterCopyPaste(folder=crops_folder,
                        transform=transforms.Compose([ # transformations for the pasted crops
                            # transforms.RandomRotation(degrees=(0, 360), expand=True),                            
                        ]),
                        individual_transform=
                        {0: transforms.Compose([ # individual transformations for the pasted crops
                            # CustomNormalizeSingleChannel(),
                            # transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2)], p=1),
                            # RevertNormalizationSingleChannel()
                        ]),
                        1: transforms.Compose([ # individual transformations for the pasted crops
                            # CustomNormalizeSingleChannel(),
                            # transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2)], p=1),
                            # # transforms.RandomApply([CustomGaussianBlur()], p=1),
                            # RevertNormalizationSingleChannel()
                        ])}
                    )], p=1)
    ])

    return train_transform, train_individual_transform