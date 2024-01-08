import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

import imageio.v3
import matplotlib.pyplot as plt
import matplotlib as mpl

from skimage.color import label2rgb
import math
import segmentation.evaluate as evaluate
import numpy as np
from random import shuffle

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
import imageio
from matplotlib.colors import LinearSegmentedColormap

image_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/images/NF135/D5/2019003_D5_135_hsp_20x_2_series_5_TileScan_001.tif"
parasite_mask_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.png"
merozoite_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/merozoite_watershed/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif'
hepatocyte_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/hepatocyte_watershed/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif'


tif = imageio.mimread(image_path)#[0:2]
segmentation = imageio.v3.imread(parasite_mask_path)
print(len(tif))

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(30, 6))

axs[0].imshow(tif[0], cmap='Blues')
axs[1].imshow(tif[1], cmap='hot')
axs[2].imshow(tif[2], cmap='BuGn')
axs[3].imshow(tif[3], cmap='Greens')


# Get cmap
cmap = mpl.cm.get_cmap('Paired', len(np.unique(segmentation)))
colors = cmap(np.linspace(0, 1, cmap.N))
np.random.shuffle(colors)
cmap = LinearSegmentedColormap.from_list('ShuffledCmap', colors, N=cmap.N)
cmap.set_under(color='black')

# Show mask
# axs[2].imshow(segmentation, cmap=cmap, vmin=0.9, interpolation='nearest')

# axs[0].axis('off')
# axs[1].axis('off')
for ax in axs:
    ax.axis('off')
# axs[2].axis('off')

plt.tight_layout()
plt.show()