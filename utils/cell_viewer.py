import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

import imageio.v3
import matplotlib.pyplot as plt
import matplotlib as mpl

import math
import segmentation.evaluate as evaluate
import numpy as np
from random import shuffle
import random

import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from utils import data_utils
import imageio
from matplotlib.colors import LinearSegmentedColormap


def plot_augmentations(sample, draw_boxes=False, show=True, save_path=None):
    original_image = sample['original_image']

    # from the masks and bounding boxes, generate an RGB version of the mask
    # augmented_mask = torch.zeros(size=(3, *original_image[0].shape)).type(torch.uint8)
    augmented_mask = torch.full((3, *sample['image'].shape[1:]), 0, dtype=torch.uint8)
    augmented_mask = draw_segmentation_masks(augmented_mask, sample['mask_3d'].type(torch.bool), alpha=1)
    if draw_boxes:
        augmented_mask = draw_bounding_boxes(augmented_mask, sample['boxes'], width=3, colors='red')
    augmented_mask = torch.permute(augmented_mask, (1, 2, 0))

    # indices = torch.arange(sample['mask_3d'].shape[0]).unsqueeze(1).unsqueeze(2)
    # augmented_mask = torch.where(sample['mask_3d'], indices, torch.zeros(1, sample['mask_3d'].shape[1], sample['mask_3d'].shape[2]))

    # Plot the original image, original mask, augmented image and augmented mask
    fig, axs = plt.subplots(2, 3, figsize=(50,30), sharex=False, sharey=False)
    aspect = 'auto'
    axs[0, 0].imshow(original_image[0], aspect=aspect, cmap='Blues')
    axs[0, 1].imshow(original_image[1], aspect=aspect, cmap='hot')
    axs[0, 2].imshow(sample['original_mask_2d'], aspect=aspect)
    axs[1, 0].imshow(sample['image'][0], aspect=aspect, cmap='Blues')
    axs[1, 1].imshow(sample['image'][1], aspect=aspect, cmap='hot')
    axs[1, 2].imshow(augmented_mask, aspect=aspect)

    fig.tight_layout()

    # [ax.axis('off') for ax in axs.flatten()]
    axs[0, 0].set_ylabel('Original')
    axs[1, 0].set_ylabel('Augmented')
    axs[0, 0].set_title('DAPI')


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
            input('Showing image {}. Press enter to continue ...'.format(idx))

# Show a single tiff with accompanying segmentations.
def plot(tif, segmentations, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None, save_path=None, num_rows=2):
    # tif = tif[:2,:,:]
    num_channels = tif.shape[0]
    num_plots = num_channels + len(segmentations)
    num_cols = math.ceil(num_plots/num_rows)

    # fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, sharey=True, figsize=(12*num_cols, 8*num_rows))
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, sharey=True, layout='constrained', figsize=(100,50))#(11, 6))
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    # fig.canvas.layout.width = '100%'
    # fig.canvas.layout.height = '100%'

    for i in range(num_cols*num_rows):
        axs_coord = ((i+1) // ((math.ceil(num_plots/num_rows)+1)), i % (math.ceil(num_plots/num_rows))) if num_rows > 1 else i

        if i >= num_channels: # segmentations go here

            if i >= num_plots: # remove axs if unused
                fig.delaxes(axs[axs_coord])
                continue

            segmentation = segmentations[i-num_channels]

            # Perform optional evaluation
            if eval and i-num_channels+1 > 0 and i != num_channels+ground_truth_index:
                evaluate.eval(segmentations[ground_truth_index], segmentation, print_metrics=True)
                
            # Get cmap
            # cmap = mpl.cm.get_cmap('Paired', len(np.unique(segmentation)))
            # colors = cmap(np.linspace(0, 1, cmap.N))
            # np.random.shuffle(colors)
            # cmap = LinearSegmentedColormap.from_list('ShuffledCmap', colors, N=cmap.N)
            # cmap.set_under(color='black')

            # Get the 'Paired' colormap
            cmap = plt.get_cmap('Paired')

            # Calculate the number of unique values in segmentation
            num_segments = len(np.unique(segmentation))

            # Generate colors using the colormap
            colors = cmap(np.linspace(0, 1, num_segments))

            # Shuffle the colors
            np.random.shuffle(colors)

            # Create a LinearSegmentedColormap from the shuffled colors
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('ShuffledCmap', colors, N=num_segments)

            # Set the 'under' color to black
            cmap.set_under(color='black')

            # Now you can use 'shuffled_cmap' in your plotting
            
            # Show mask
            if len(np.unique(segmentation)) > 1:
                axs[axs_coord].imshow(segmentation, cmap=cmap, vmin=0.9, interpolation='nearest')
            else:
                axs[axs_coord].imshow(segmentation)

            if titles: axs[axs_coord].set_title(title)

        else: # regular channels go here
            if colormaps and len(colormaps)-1 >= i:
                cmap = colormaps[i]
            else:
                cmap = None

            axs[axs_coord].imshow(tif[i], cmap=cmap)
            axs[axs_coord].set_title(titles[i] if titles else None)
    
    if title: fig.suptitle(os.path.basename(title))

    if save_path:
        plt.savefig(save_path)
    else:
        # plt.tight_layout(h_pad=0)#3) 
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        [ax.axis('off') for ax in axs]
        plt.show()

def show_file(tif_path, segmentation_paths, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None, save_path=None, num_rows=2):
    tif = np.array(imageio.mimread(tif_path))
    segmentations = [imageio.v3.imread(segmentation_path) for segmentation_path in segmentation_paths]
    segmentations = [np.squeeze(s, 0) if len(s.shape) > 2 else s for s in segmentations]
    print(tif_path[:-4])
    plot(tif, segmentations, eval=eval, colormaps=colormaps, titles=titles, ground_truth_index=ground_truth_index, title=title, save_path=save_path, num_rows=num_rows)

def show_files(tif_paths, segmentation_paths_2d, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None, shuffle_indices=True, num_rows=2):
    tif_indices = list(range(len(tif_paths)))
    if shuffle_indices: shuffle(tif_indices)

    for i in tif_indices:
        tif_path = tif_paths[i]
        seg_paths = [segmentation_paths[i] for segmentation_paths in segmentation_paths_2d]
        show_file(tif_path, seg_paths, eval=eval, colormaps=colormaps, titles=titles, ground_truth_index=ground_truth_index, title=title, num_rows=num_rows)

# Show a tiff file in a directory. When closed keep opening the next tiff file.
def show_folder(tif_dir, seg_dirs=[], colormaps=None, titles=None, eval=False, ground_truth_index=0, shuffle_indices=True, num_rows=1, num_images=None):
    tif_paths = data_utils.get_paths(tif_dir, extension='.tif', substring=None)

    tif_paths = random.sample(tif_paths, num_images) if num_images else tif_paths
    if seg_dirs:
        seg_paths = [data_utils.get_paths(folder) for folder in seg_dirs]
        path_intersections = data_utils.intersection_of_lists([tif_paths] + seg_paths)
        tif_paths, seg_paths = path_intersections[0], path_intersections[1:]
    else:
        seg_paths = []
    
    show_files(tif_paths, segmentation_paths_2d=seg_paths, colormaps=colormaps, titles=titles, eval=eval, ground_truth_index=ground_truth_index, shuffle_indices=shuffle_indices, num_rows=num_rows)

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

    # image_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/images/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif"
    # parasite_mask_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.png"
    # merozoite_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/merozoite_watershed/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif'
    # hepatocyte_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/hepatocyte_watershed/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif'
    
    # show_file(image_path, [parasite_mask_path, merozoite_mask_path, hepatocyte_mask_path], colormaps=colormaps, titles=titles)#, save_path='/mnt/DATA1/anton/example6.png')

    # High res testing
    colormaps = ['Blues', 'hot', 'BuGn', 'Greens']
    # colormaps = ['hot', 'Greens', 'Blues']

    # img_folder = '/mnt/DATA1/anton/data/dataset/images/lowres/NF54/D7'
    # parasite_mask_folder = '/mnt/DATA1/anton/data/dataset/annotation'
    # segmentation_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/test3'
    # show_folder(img_folder, [parasite_mask_folder], colormaps=colormaps, eval=True)

    # img_folder = '/mnt/DATA1/anton/data/hepatocyte_annotated_dataset/images/lowres'
    # parasite_mask_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/annotation'
    # hepatocyte_mask_folder = '/mnt/DATA1/anton/data/hepatocyte_annotated_dataset/annotation'
    # show_folder(img_folder, [parasite_mask_folder, hepatocyte_mask_folder], colormaps=colormaps)


    lowres_tif_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/images/lowres/NF54'
    highres_tif_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/images/highres'

    lowres_target_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/watershed/lowres'
    highres_target_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/watershed/highres'

    # show_folder(lowres_tif_folder, [lowres_target_folder], colormaps=['Blues', 'hot', 'BuGn', 'Greens'])
    # show_folder(highres_tif_folder, [highres_target_folder], colormaps=['hot', 'Greens', 'Blues'])

    # tif_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/images'
    # anno_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/annotation'
    # seg_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/full_augmentation_3'
    # show_folder(tif_folder, [anno_folder, seg_folder], colormaps=['Blues', 'hot', 'BuGn', 'Greens'])


    img_folder = '/mnt/DATA3/compounds/11C-organised'
    parasite_mask_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/parasite_segmentations/6c'
    hep_mask_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/hepatocyte_segmentations/6c'

    show_file(tif_path=None, segmentation_paths=['/mnt/DATA1/anton/pipeline_files/segmentation/parasite_segmentations/6c/Exp2021025C-01-Scene-15-C7-C07_series_30_Exp2021025C-01-Scene-15-C7-C07.tif'])


    show_folder(img_folder, [parasite_mask_folder, hep_mask_folder], colormaps=['Blues', 'hot', 'BuGn', 'Greens'])
