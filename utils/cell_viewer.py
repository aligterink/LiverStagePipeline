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


def plot_augmentations(sample, draw_boxes=False, show=True, save_path=None):
    original_image = imageio.mimread(sample['file_path']) # read in the original image

    # from the masks and bounding boxes, generate an RGB version of the mask
    augmented_mask = torch.zeros(size=(3, *original_image[0].shape)).type(torch.uint8)
    augmented_mask = draw_segmentation_masks(augmented_mask, sample['mask_3d'].type(torch.bool))
    if draw_boxes:
        augmented_mask = draw_bounding_boxes(augmented_mask, sample['boxes'], width=3, colors='red')
    augmented_mask = torch.permute(augmented_mask, (1, 2, 0))

    # Plot the original image, original mask, augmented image and augmented mask
    fig, axs = plt.subplots(2, 3, figsize=(60,30), sharex=True, sharey=True)
    aspect = 'auto'
    axs[0, 0].imshow(original_image[0], aspect=aspect, cmap='Blues')
    axs[0, 1].imshow(original_image[1], aspect=aspect, cmap='hot')
    axs[0, 2].imshow(sample['mask_2d'], aspect=aspect)
    axs[1, 0].imshow(sample['image'][0], aspect=aspect, cmap='Blues')
    axs[1, 1].imshow(sample['image'][1], aspect=aspect, cmap='hot')
    axs[1, 2].imshow(augmented_mask, aspect=aspect)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

def show_dataset(dataset, show=True, save_path=None):
    ids = np.arange(0, dataset.__len__())
    np.random.shuffle(ids)
    for idx in ids:
        sample = dataset.__getitem__(idx)
        plot_augmentations(sample, show=show, save_path=save_path)
        if not show:
            input('Showing image {}. Pres enter to continue ...'.format(idx))

# Show a single tiff with accompanying segmentations.
def plot(tif, segmentations, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None, save_path=None):
    num_channels = tif.shape[0]
    num_plots = num_channels + len(segmentations)

    fig, axs = plt.subplots(nrows=2, ncols=math.ceil(num_plots/2), sharex=True, sharey=True, figsize=(32, 16))
    if num_plots % 2:
        fig.delaxes(axs[1, math.ceil(num_plots/2)-1])
        
    for i in range(num_plots):
        axs_coord = ((i+1) // ((math.ceil(num_plots/2)+1)), i % (math.ceil(num_plots/2)))

        if i >= num_channels: # segmentations go here
            segmentation = segmentations[i-num_channels]

            # Perform optional evaluation
            if eval and i-num_channels+1 > 0 and i != num_channels+ground_truth_index:
                evaluate.eval(segmentations[ground_truth_index], segmentation, print_metrics=True)
                
            # Get cmap
            cmap = mpl.cm.get_cmap('Paired', len(np.unique(segmentation)))
            colors = cmap(np.linspace(0, 1, cmap.N))
            np.random.shuffle(colors)
            cmap = LinearSegmentedColormap.from_list('ShuffledCmap', colors, N=cmap.N)
            cmap.set_under(color='black')
            
            # Show mask
            axs[axs_coord].imshow(segmentation, cmap=cmap, vmin=0.9, interpolation='nearest')
            if titles: axs[axs_coord].set_title(title)

        else: # regular channels go here
            axs[axs_coord].imshow(tif[i], cmap=colormaps[i] if colormaps else None)
            axs[axs_coord].set_title(titles[i] if titles else None)
    
    if title: fig.suptitle(os.path.basename(title))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.tight_layout(h_pad=0)#3) 
        plt.show()

def show_file(tif_path, segmentation_paths, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None, save_path=None):
    tif = np.array(imageio.mimread(tif_path)[0:2])
    segmentations = [imageio.v3.imread(segmentation_path) for segmentation_path in segmentation_paths]
    segmentations = [np.squeeze(s, 0) if len(s.shape) > 2 else s for s in segmentations]
    plot(tif, segmentations, eval=eval, colormaps=colormaps, titles=titles, ground_truth_index=ground_truth_index, title=title, save_path=save_path)

def show_files(tif_paths, segmentation_paths_2d, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None, shuffle_indices=True):
    tif_indices = list(range(len(tif_paths)))
    if shuffle_indices: shuffle(tif_indices)

    for i in tif_indices:
        tif_path = tif_paths[i]
        seg_paths = [segmentation_paths[i] for segmentation_paths in segmentation_paths_2d]
        show_file(tif_path, seg_paths, eval=eval, colormaps=colormaps, titles=titles, ground_truth_index=ground_truth_index, title=title)

# Show a tiff file in a directory. When closed keep opening the next tiff file.
def show_folder(tif_dir, seg_dirs=[], colormaps=None, titles=None, eval=True, ground_truth_index=0, shuffle_indices=True):
    tif_paths = data_utils.get_paths(tif_dir, extension='.tif')
    if seg_dirs:
        seg_paths = [data_utils.get_paths(folder) for folder in seg_dirs]
        path_intersections = data_utils.intersection_of_lists([tif_paths] + seg_paths)
        tif_paths, seg_paths = path_intersections[0], path_intersections[1:]
    else:
        seg_paths = []
    
    show_files(tif_paths, segmentation_paths_2d=seg_paths, colormaps=colormaps, titles=titles, eval=eval, ground_truth_index=ground_truth_index, shuffle_indices=shuffle_indices)

if __name__ == "__main__":
    colormaps = ['Blues', 'hot', 'BuGn', 'Greens']
    titles = ['DAPI', 'HSP70', 'Channel 3', 'Channel 4'] 

    # tif_dir = "/mnt/DATA1/anton/data/lowres_dataset_selection/images/NF135"
    # seg_dir = ["/mnt/DATA1/anton/data/lowres_dataset_selection/annotation", 
    #         #    "/mnt/DATA1/anton/pipeline_files/results/segmentation_collection/best_yet_clusterAug/", 
    #            "/mnt/DATA1/anton/data/lowres_dataset_selection/watershed_normtest"]
    
    # tif_dir = "/mnt/DATA1/anton/data/lowres_dataset/images/NF135/D5"
    # seg_dir = ['/mnt/DATA1/anton/data/lowres_dataset/annotation', '/mnt/DATA1/anton/data/lowres_dataset/watershed_test', '/mnt/DATA1/anton/data/lowres_dataset/merozoites_test']
    # show_folder(tif_dir, seg_dir, colormaps=colormaps, titles=titles)

    image_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/images/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif"
    parasite_mask_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.png"
    merozoite_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/merozoite_watershed/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif'
    hepatocyte_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/hepatocyte_watershed/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif'
    
    show_file(image_path, [], colormaps=colormaps, titles=titles)#, save_path='/mnt/DATA1/anton/example6.png')



    