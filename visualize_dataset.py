import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[0] + 'LiverStagePipeline')

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
from utils import data_utils


def plot(self, original_image, labeled_mask, augmented_image, target, name, savepath):
    augmented_mask = torch.zeros_like(torch.Tensor(3, 1040, 1392)).type(torch.uint8)
    masks = target['masks'].type(torch.bool)
    augmented_mask = draw_segmentation_masks(augmented_mask, masks)
    # augmented_mask = draw_bounding_boxes(augmented_mask, target['boxes'], width=3, colors='red')
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