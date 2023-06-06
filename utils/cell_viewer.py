import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[0] + 'LiverStagePipeline')

import imageio.v3
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import math
import utils.data_utils as data_utils
from pathlib import Path
import segmentation.evaluate as evaluate
import numpy as np
from random import shuffle

# Show a single tiff with accompanying segmentations.
def plot(tif, segmentations, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None):
    num_channels = tif.shape[0]
    num_plots = num_channels + len(segmentations)

    fig, axs = plt.subplots(nrows=2, ncols=math.ceil(num_plots/2), sharex=True, sharey=True, figsize=(32, 16))
    if num_plots % 2:
        fig.delaxes(axs[1, math.ceil(num_plots/2)-1])

    for i in range(num_plots):
        x, y = (i+1) // ((math.ceil(num_plots/2)+1)), i % (math.ceil(num_plots/2))

        if i >= num_channels:
            segmentation = segmentations[i-num_channels]
            rgb_seg = segmentation.copy()
            rgb_seg[rgb_seg!=0] += 40

            # Perform optional evaluation
            if eval and i-num_channels+1 > 0 and i != num_channels+ground_truth_index+1:
                evaluate.eval(segmentations[ground_truth_index], segmentation, print_metrics=True)

            # Show mask
            axs[x, y].imshow(rgb_seg)
            if titles: axs[x, y].set_title(title)
        else:
            axs[x, y].imshow(tif[i], cmap=colormaps[i])
            axs[x, y].set_title(titles[i])
    
    if title: fig.suptitle(os.path.basename(title))
    plt.tight_layout(h_pad=0)#3)

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()   
    plt.show()

def show_file(tif_path, segmentation_paths, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None):
    tif = np.array(imageio.mimread(tif_path))
    segmentations = [imageio.v3.imread(segmentation_path) for segmentation_path in segmentation_paths]
    plot(tif, segmentations, eval=eval, colormaps=colormaps, titles=titles, ground_truth_index=ground_truth_index, title=title)

def show_files(tif_paths, segmentation_paths_2d, colormaps=None, titles=None, eval=False, ground_truth_index=0, title=None, shuffle_indices=True):
    tif_indices = list(range(len(tif_paths)))
    if shuffle_indices: shuffle(tif_indices)

    for i in tif_indices:
        tif_path = tif_paths[i]
        seg_paths = [segmentation_paths[i] for segmentation_paths in segmentation_paths_2d]
        show_file(tif_path, seg_paths, eval=eval, colormaps=colormaps, titles=titles, ground_truth_index=ground_truth_index, title=title)

# Show a tiff file in a directory. When closed keep opening the next tiff file.
def show_folder(tif_dir, seg_dirs=[], colormaps=None, titles=None, eval=True, ground_truth_index=0, shuffle_indices=True):
    tif_paths = data_utils.get_paths(tif_dir)
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

    tif_dir = "/mnt/DATA1/anton/data/lowres_dataset_selection/images/NF135"
    seg_dir = ["/mnt/DATA1/anton/data/lowres_dataset_selection/annotation", "/mnt/DATA1/anton/pipeline_files/results/segmentation_collection/best_yet_clusterAug/"]
    show_folder(tif_dir, seg_dir, colormaps=colormaps, titles=titles)

    # tif_file = "/mnt/DATA1/anton/data/unformatted/GS validation data/untreated_tifs/Experiment 2019003/tif/2019003_D3_135_hsp_20x_2_series_22_TileScan_001.tif"
    # seg_file1 = "/mnt/DATA1/anton/data/unformatted/GS validation data/untreated_segmentations_2_copypaste_2905/2019003/2019003_D3_135_hsp_20x_2_series_22_TileScan_001.png"
    # show_file(tif_file, [seg_file1], colormaps=colormaps, titles=titles)
    

    