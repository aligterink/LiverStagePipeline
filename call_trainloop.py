import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

# from torchvision import disable_beta_transforms_warning
# disable_beta_transforms_warning()

from segmentation.AI import logger

import numpy as np
import os
import glob
from sklearn.model_selection import StratifiedShuffleSplit

import segmentation.evaluate as evaluate
import utils.setup as setup
import torch
import segmentation.AI.train as train
import segmentation.AI.datasets as datasets
from utils import data_utils, mask_utils, cell_viewer

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import get_models as get_models
import get_dataloaders as get_dataloaders
import get_transforms as get_transforms



### Settings
pipeline_files_folder = '/mnt/DATA1/anton/pipeline_files'

main_folder = "/mnt/DATA1/anton/pipeline_files"
train_name = "mrcnn_test"

imgs_folder = "/mnt/DATA1/anton/data/lowres_dataset_selection/images"
masks_folder = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation"

folder_normalize = True
train_batch_size, test_batch_size = 1, 8
loader_workers, eval_workers = train_batch_size, 12
    
### Setting up paths

# Set necessary paths and create folders if non-existent
paths = setup.setup(pipeline_files_folder)

log_file = os.path.join(paths['segmentation_logs_folder'], train_name + '.log')
model_file = os.path.join(paths['segmentation_models_folder'], train_name + '.pth')
figure_file = os.path.join(paths['segmentation_figures_folder'], train_name + '.png')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Training on {}".format(device))

# Define the model
model, model_output_parser = get_models.maskrcnn('resnet101')
model.to(device)
model.device = device

train_transform, train_individual_transform = get_transforms.v1(crops_folder=paths['segmentation_crops_folder'])







img_paths, mask_paths = data_utils.get_two_sets(dir1=imgs_folder, dir2=masks_folder, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
groups = [os.path.dirname(path)[len(imgs_folder)+1:].replace('/', ' ') for path in img_paths]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_indices, test_indices = list(sss.split(list(range(len(img_paths))), groups))[0]

trainset = datasets.MicroscopyDataset([img_paths[i] for i in train_indices], mask_paths=[mask_paths[i] for i in train_indices], groups=[groups[i] for i in train_indices], 
                                        transform=train_transform, individual_transform=train_individual_transform, folder_normalize=folder_normalize)
dummy_trainset = datasets.MicroscopyDataset([img_paths[i] for i in train_indices], mask_paths=[mask_paths[i] for i in train_indices], 
                                            filter_empty=True, transform=None, individual_transform=None, folder_normalize=folder_normalize)
testset = datasets.MicroscopyDataset([img_paths[i] for i in test_indices], mask_paths=[mask_paths[i] for i in test_indices], groups=[groups[i] for i in test_indices], 
                                        folder_normalize=folder_normalize)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=loader_workers,shuffle=True, collate_fn=data_utils.collate_fn)
dummy_train_loader = torch.utils.data.DataLoader(dummy_trainset, batch_size=test_batch_size, num_workers=loader_workers, shuffle=True, collate_fn=data_utils.collate_fn)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, num_workers=loader_workers, shuffle=True, collate_fn=data_utils.collate_fn)

# Generate new crops of training set cells
crop_files = glob.glob(paths['segmentation_crops_folder'] + '/*.tif')
for f in crop_files:
    os.remove(f)
mask_utils.extract_crops_from_loader(loader=dummy_train_loader, folder=paths['segmentation_crops_folder'])


cell_viewer.show_dataset(trainset, show=False, save_path='/mnt/DATA1/anton/example.png')




# evaluator = evaluate.Inferenceinator(test_loader, model_output_parser, processes=eval_workers)
# optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.00005, amsgrad=True)
# # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

# train.train(model, train_loader, evaluator, num_epochs=10, optimizer=optimizer, get_loss_func=model_output_parser,
#             log_file=log_file, figure_file=figure_file, model_path=model_file, metric_for_best='aP')

# model.load_state_dict(torch.load(model_file))
# res = evaluator(model, store_folder=paths['segmentation_folder'])
# print(res)



