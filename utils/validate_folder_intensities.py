import sys
import os
sys.path.append(os.path.sep + os.path.join(*(__file__.split(os.path.sep)[:next((i for i in range(len(__file__.split(os.path.sep)) -1, -1, -1) if 'LiverStagePipeline' in __file__.split(os.path.sep)[i]), None)+1])))

import os
import numpy as np
import imageio
from utils import data_utils
import matplotlib.pyplot as plt
from pathlib import Path

## Settings
folder = '/mnt/DATA1/anton/data/lowres_dataset_selection/images'
channels = [0, 1]


image_paths = data_utils.get_paths(folder, extension='.tif', recursive=True, substring=None)
folder_paths = set([os.path.dirname(path) for path in image_paths])

fig, axs = plt.subplots(ncols=len(channels), nrows=len(folder_paths), sharex=True, sharey=True)

for i, fp in enumerate(folder_paths):
    paths_subset = [path for path in image_paths if path.startswith(fp)]

    for j, channel in enumerate(channels):
        means, SDs, mins, maxs = [], [], [], []

        for path in paths_subset:
    
            image = imageio.mimread(path, memtest=False)[channel]

            means.append(np.mean(image))
            SDs.append(np.std(image))
            mins.append(np.min(image))
            maxs.append(np.max(image))

        x = np.arange(0, len(means))
        axs[i, j].plot(means, label='mean')
        axs[i, j].plot(SDs, label='SD')
        axs[i, j].plot(mins, label='min')
        axs[i, j].plot(maxs, label='max')

        axs[i, j].set_xlabel(channel)
        axs[i, j].set_ylabel(fp, rotation=45)


plt.legend()
plt.show()






# for fp in lowest_level_folders:
#     paths_subset = [path for path in image_paths if path.startswith(fp)]

#     for channel in channels:
#         images = [imageio.mimread(path, memtest=False)[channel] for path in paths_subset]


# return mean_dict, SD_dict