import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

# from torchvision import disable_beta_transforms_warning
# disable_beta_transforms_warning()


import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit

from segmentation.evaluate import Inferenceinator
import utils.setup as setup
import torch
import segmentation.AI.train as train
from segmentation.AI.datasets import MicroscopyDataset
from utils import data_utils, mask_utils, cell_viewer

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import get_models as get_models
import get_dataloaders as get_dataloaders
import get_transforms as get_transforms



### Settings
pipeline_files_folder = '/mnt/DATA1/anton/pipeline_files'

main_folder = "/mnt/DATA1/anton/pipeline_files"
session_name = "all_aug_final_final"

imgs_folder = "/mnt/DATA1/anton/data/parasite_annotated_dataset/images"
masks_folder = "/mnt/DATA1/anton/data/parasite_annotated_dataset/annotation"

folder_normalize = True
train_batch_size, test_batch_size = 3, 8
loader_workers, eval_workers = train_batch_size, 12

train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Set necessary paths and create folders if non-existent
paths = setup.setup(pipeline_files_folder, session_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Training on {}".format(device))

# Define the model
model, model_output_parser = get_models.maskrcnn('resnet101')
model.to(device)
model.device = device

train_transform, train_individual_transform = get_transforms.v1(crops_folder=paths['segmentation_crops_folder'])




img_paths, mask_paths = data_utils.get_two_sets(dir1=imgs_folder, dir2=masks_folder, common_subset=True, extension_dir1='.tif', extension_dir2='', return_paths=True)
groups = [os.path.dirname(path)[len(imgs_folder)+1:].replace('/', ' ') for path in img_paths]

channels = [[0, 1] if 'lowres' in img_path else [2, 0] for img_path in img_paths]

indices = list(range(len(img_paths)))
sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio + test_ratio, random_state=200)
train_indices, tmp_indices = list(sss.split(indices, groups))[0]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio / (validation_ratio + test_ratio), random_state=200)
validation_indices, test_indices = list(sss2.split(tmp_indices, [groups[i] for i in tmp_indices]))[0]


trainset = MicroscopyDataset([img_paths[i] for i in train_indices], mask_paths=[mask_paths[i] for i in train_indices], channels=[channels[i] for i in train_indices], groups=[groups[i] for i in train_indices], 
                                        transform=train_transform, individual_transform=train_individual_transform, folder_normalize=folder_normalize, rescale_img=(1040, 1392), rescale_mask=(1040, 1392))
valset = MicroscopyDataset([img_paths[i] for i in validation_indices], mask_paths=[mask_paths[i] for i in validation_indices], channels=[channels[i] for i in validation_indices], 
                                     groups=[groups[i] for i in validation_indices], folder_normalize=folder_normalize, rescale_img=(1040, 1392))
testset = MicroscopyDataset([img_paths[i] for i in test_indices], mask_paths=[mask_paths[i] for i in test_indices], channels=[channels[i] for i in test_indices], 
                                     groups=[groups[i] for i in test_indices], folder_normalize=folder_normalize, rescale_img=(1040, 1392))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=loader_workers,shuffle=True, collate_fn=data_utils.collate_fn)
validation_loader = torch.utils.data.DataLoader(valset, batch_size=test_batch_size, num_workers=loader_workers, shuffle=True, collate_fn=data_utils.collate_fn)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, num_workers=loader_workers, shuffle=True, collate_fn=data_utils.collate_fn)

mask_utils.extract_crops_from_set(dataset=trainset, folder=paths['segmentation_crops_folder']) # generate new crops of training set cells


# cell_viewer.show_dataset(trainset, show=True)#, save_path='/mnt/DATA1/anton/example.png')




validation_evaluator = Inferenceinator(validation_loader, model_output_parser, processes=eval_workers, device=device)
test_evaluator = Inferenceinator(test_loader, model_output_parser, processes=eval_workers, log_file=paths['log_file'], device=device)

optimizer = optim.Adam(model.parameters(), lr=0.0005, amsgrad=True)

train.train(model, train_loader, validation_evaluator, num_epochs=100, optimizer=optimizer, get_loss_func=model_output_parser,
            log_file=paths['log_file'], figure_file=paths['figure_file'], model_path=paths['model_file'], metric_for_best='F1@0.5', device=device)

model.load_state_dict(torch.load(paths['model_file']))
res = test_evaluator(model, store_folder=paths['parasite_masks_folder'])
print(res)
