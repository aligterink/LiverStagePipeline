from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

from segmentation.AI import datasets
from utils import data_utils

import torch
import imageio.v3

testset_names, test_sets, X_train, y_train = [], [], [], []

img_folder = '/mnt/DATA1/anton/data/lowres_dataset_selection/images'
anno_folder = '/mnt/DATA1/anton/data/lowres_dataset_selection/annotation'

paths = data_utils.get_two_sets(img_folder, anno_folder, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)

def collate_fn(batch):
    return list(zip(*list(filter(lambda x : x if(x is not None) else None, batch))))

mds = datasets.MicroscopyDataset(paths[0], paths[1], filter_empty=True, folder_normalize=True)
# loader = torch.utils.data.DataLoader(mds, batch_size=4, num_workers=4,shuffle=True, collate_fn=collate_fn)
print(mds[0])
