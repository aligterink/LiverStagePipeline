import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

from utils import data_utils

import napari
import imageio.v3
import os
from PIL import Image
import numpy as np
from pathlib import Path

global file_index 

### Settings
tif_dir = "/mnt/DATA1/anton/data/high_res_subset_from_Felix/B04"
seg_dir = "/mnt/DATA1/anton/data/high_res_subset_from_Felix/B04_watershed/"
starting_tif = "Exp2021025A-01-Scene-03-B4-B04_series_1_Exp2021025A-01-Scene-03-B4-B04.tif"

remove_all_labels_key = 't'
remove_blob_key = 'r'
save_key = 's'
next_image_key = 'v'
previous_image_key = 'c'
new_label_key = 'f'
toggle_labels = 'e'

# Indices in these lists coincide with the indices of the layers. 
# E.g. the first layer toggle key will toggle the first layer.
layer_minimal_contrast = [100, 200, 200, 200, 200]
layer_maximal_contrast = [2000, 4000, 1200, 2000, 800]
layer_toggle_keys = ['q', 'w', 'z', 'x', 'b']
layer_names = ['blue', 'red', 'cyan', 'green', 'some other layer']
layer_initial_visibility = [False, True, False, False, False]
layer_colormaps = ['blue', 'red', 'cyan', 'green', 'yellow']


# Initialize files
tif_files, seg_files = data_utils.get_two_sets(tif_dir, seg_dir, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
file_index = tif_files.index(os.path.join(tif_dir, starting_tif))

# create viewer
viewer = napari.Viewer()

def change_image(viewer, forward): # makes the viewer go an image forwards or backwards
    viewer.layers.clear()

    global file_index
    file_index += forward

    current_tif_path = os.path.join(tif_dir, tif_files[file_index])
    current_seg_path = os.path.join(seg_dir, seg_files[file_index])
    assert Path(current_tif_path).stem == Path(current_seg_path).stem

    image = imageio.mimread(current_tif_path) # read tif

    for i in range(len(image)):
        # viewer.add_image(image[i], name=layer_names[i], blending='additive', colormap=layer_colormaps[i], visible=layer_initial_visibility[i], contrast_limits=[0, np.percentile(image[i], layer_contrast_percentiles[i])])
        viewer.add_image(image[i], name=layer_names[i], blending='additive', colormap=layer_colormaps[i], visible=layer_initial_visibility[i], contrast_limits=[layer_minimal_contrast[i], layer_maximal_contrast[i]])

    seg_layer = viewer.open(current_seg_path, layer_type='labels')

@viewer.bind_key(remove_blob_key)
def remove_blob(viewer): # remove all pixels with the currently selected label
    cur_label = viewer.layers[-1]._selected_label
    msg = 'Removed label {}'.format(cur_label)
    viewer.status = msg
    print(msg)
    viewer.layers[-1].data[viewer.layers[-1].data == viewer.layers[-1]._selected_label] = 0
    viewer.layers[-1].refresh()

@viewer.bind_key(remove_all_labels_key)
def remove_blob(viewer): # remove all labels
    viewer.layers[-1].data = np.zeros_like(viewer.layers[-1].data)
    viewer.layers[-1].refresh()
    msg = 'Removed all labels'
    viewer.status = msg
    print(msg)

@viewer.bind_key(save_key)
def save_button(viewer): # save file

    global file_index
    current_seg_path = os.path.join(seg_dir, seg_files[file_index])
    seg = viewer.layers[-1].data
    seg_img = Image.fromarray(seg.astype(np.uint8), 'L')
    seg_img.save(current_seg_path)

    msg = 'Image saved. Nice going Annie! :)'
    viewer.status = msg
    print(msg)


@viewer.bind_key(next_image_key)
def next_image(viewer): # go 1 image forward 
    change_image(viewer, forward=1)
    
    msg = 'Loading the next image ...'
    viewer.status = msg
    print(msg)

@viewer.bind_key(previous_image_key)
def previous_image(viewer): # go 1 image backward
    change_image(viewer, forward=-1)

    msg = 'Loading the previous image ...'
    viewer.status = msg
    print(msg)

@viewer.bind_key(new_label_key)
def new_label(viewer): # Switch to lowest available label
    present_labels = np.unique(viewer.layers[-1].data).tolist()
    lowest_available_label = min(set(range(max(present_labels) + 2)) - set(present_labels))
    viewer.layers[-1]._selected_label = lowest_available_label

    msg = 'Switched to lowest available label {}'.format(lowest_available_label)
    viewer.status = msg
    print(msg)

# Generate functions for toggling layers
for i in range(len(layer_toggle_keys)):
    @viewer.bind_key(layer_toggle_keys[i])
    def toggle_layer(viewer, i=i):
        viewer.layers[i].visible = not viewer.layers[i].visible

@viewer.bind_key(toggle_labels)
def toggle_label_layer(viewer):
    viewer.layers[-1].visible = not viewer.layers[-1].visible

change_image(viewer, 0)


if __name__ == '__main__':
    napari.run()
