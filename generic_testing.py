from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

from segmentation.AI import dataset
from utils import data_utils

import torch

testset_names, test_sets, X_train, y_train = [], [], [], []

img_folder = R"C:\Users\anton\Documents\microscopy_data\dataset\images"
anno_folder = R"C:\Users\anton\Documents\microscopy_data\dataset\annotation"

paths = data_utils.get_two_sets(img_folder, anno_folder, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)

def collate_fn(batch):
    return list(zip(*list(filter(lambda x : x if(x is not None) else None, batch))))

mds = dataset.MicroscopyDataset(paths[0], paths[1], filter_empty=True, set_folder_ranges=True)
loader = torch.utils.data.DataLoader(mds, batch_size=4, num_workers=4,shuffle=True, collate_fn=collate_fn)

