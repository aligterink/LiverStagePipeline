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

# Show a single tiff with accompanying segmentations.
# if eval=True, then first seg_path should be the ground truth annotation
def show_file(tif_path, seg_paths=[], eval=False):
    colormaps = ['Blues', 'hot', 'BuGn', 'Greens']
    titles = ['DAPI', 'HSP70', 'Channel 3', 'Channel 4']

    channels = imageio.mimread(tif_path) # read image
    num_channels = len(channels) + len(seg_paths)

    fig, axs = plt.subplots(nrows=2, ncols=math.ceil(num_channels/2), sharex=True, sharey=True, figsize=(15, 8))
    if num_channels % 2:
        fig.delaxes(axs[1, math.ceil(num_channels/2)-1])

    for i in range(num_channels):
        x, y = (i+1) // ((math.ceil(num_channels/2)+1)), i % (math.ceil(num_channels/2))

        if i >= len(channels):
            # Read mask file
            seg = imageio.v3.imread(seg_paths[i-len(channels)])
            # rgbseg = label2rgb(seg, bg_label=0) #, colors=0.25+0.5*np.random.random((len(np.unique(seg)), 3)))
            rgb_seg = seg.copy()
            rgb_seg[rgb_seg!=0] += 40

            # Perform optional evaluation
            if eval and i-len(channels) > 0:
                true_mask = imageio.v3.imread(seg_paths[0])
                # evaluate.eval([true_mask], [seg], identifier=Path(seg_paths[i-len(channels)]).stem)
                evaluate.eval(true_mask, seg, print_metrics=True)

            # Show mask
            axs[x, y].imshow(rgb_seg) #, vmin=0, vmax=16000)
            axs[x, y].set_title('/'.join(os.path.normpath(seg_paths[i-len(channels)]).split(os.sep)[-3:-1]))
        else:
            axs[x, y].imshow(channels[i], cmap=colormaps[i]) #, vmin=0, vmax=16000)
            axs[x, y].set_title(titles[i])
    
    fig.suptitle(os.path.basename(tif_path))
    plt.tight_layout(h_pad=0)#3)

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()   
    plt.show()

# Show a tiff file in a directory. When closed keep opening the next tiff file.
def show_folder(tif_dir, seg_dirs, tif_file=None, eval=True):
    
    tif_paths = data_utils.get_paths(tif_dir)
    file_index = [Path(p).stem for p in tif_paths].index(tif_file[:-4]) if tif_file else np.random.randint(0, len(tif_paths))

    for i in range(file_index, len(tif_paths)):
        tif_path = tif_paths[i]
        seg_paths = []
        
        for seg_dir in seg_dirs:
            if not isinstance(seg_dir, list):
                seg_dir = [seg_dir]
                
            seg_path = data_utils.find_stem_in_other_folder(seg_dir, Path(tif_path).stem)
            if os.path.isfile(seg_path):
                seg_paths.append(seg_path)
        
        show_file(tif_path, seg_paths, eval=eval)

if __name__ == "__main__":
    # tif_dirs = [R"C:\Users\anton\Documents\microscopy_data\dataset\images\NF54"]
    # seg_dirs = [
    #     [R"C:\Users\anton\Documents\microscopy_data\dataset\annotation"],
    #     [R"C:\Users\anton\Documents\microscopy_data\model_segmentations\segmentations_partitioned6"]
    #              ]
    # tif_file = "2019003_D7_54_hspgsb_series_10_TileScan_001.png"
    # # tif_file = '2019003_D7_175_hsp_gs_a_series_2_TileScan_001.tif'
    # # tif_dirs = [R"C:\Users\anton\Documents\microscopy_data\Stressed cells"]
    # # seg_dirs = []


    # tif_paths = sum([data_utils.get_paths(folder, extension='.tif') for folder in tif_dirs], [])
    # seg_paths_2d = [sum([data_utils.get_paths(folder, extension='.png') for folder in seg_dir], []) for seg_dir in seg_dirs]
    # visualize_files(tif_paths, seg_paths_2d, tif_file=None)

    tif_dir = "/mnt/DATA1/anton/data/lowres_dataset/images"
    seg_dir = ["/mnt/DATA1/anton/data/lowres_dataset/annotation", "/mnt/DATA1/anton/data/lowres_dataset/watershed"]
    show_folder(tif_dir, seg_dir)
    

    