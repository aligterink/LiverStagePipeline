import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

# from torchvision import disable_beta_transforms_warning
# disable_beta_transforms_warning()

from segmentation.AI import logger

import numpy as np
import os
import glob
from sklearn.model_selection import KFold

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200000"


### Settings
pipeline_files_folder = '/mnt/DATA1/anton/pipeline_files'

main_folder = "/mnt/DATA1/anton/pipeline_files"
train_name = "mrcnn_test"

dataset_path = "/mnt/DATA1/anton/data/lowres_dataset_selection"

batch_size = 1
dataloader_workers = 4
eval_workers = 4
    
### Setting up paths


# Set necessary paths and create folders if non-existent
paths = setup.setup(pipeline_files_folder)

log_file = os.path.join(paths['segmentation_logs_folder'], train_name + '.log')
model_file = os.path.join(paths['segmentation_models_folder'], train_name + '.pth')
figure_file = os.path.join(paths['segmentation_figures_folder'], train_name + '.png')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if not torch.cuda.is_available():
    print("Using non-cuda device: {}".format(device))

train_transform, train_individual_transform = get_transforms.v1(crops_folder=paths['segmentation_crops_folder'])
train_loader, test_loaders, testset_names, dummy_loader = get_dataloaders.v3(train_transform, train_individual_transform, batch_size, dataloader_workers, paths['segmentation_crops_folder'])

evaluator = evaluate.Evaluator(device, train_loader=train_loader, test_loaders=test_loaders, store_folder=paths['segmentation_folder'], processes=eval_workers, testset_names=testset_names)

# # Generate new crops of training set cells
# crop_files = glob.glob(crops_folder + '/*.tif')
# for f in crop_files:
#     os.remove(f)
# mask_utils.extract_crops_from_loader(loader=dummy_loader, folder=crops_folder)

# Define the model
# model = get_models.maskrcnn('resnet101')
model = get_models.maskformer()
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.00005, amsgrad=True)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

train.train(model, train_loader, evaluator, num_epochs=1, optimizer=optimizer, scheduler=scheduler, print_every=1, 
            device=device, log_file=log_file, figure_file=figure_file, model_path=model_file, eval_trainloader=False, 
            metric_for_best='test_aP', 
            printed_vals=['test_aP', 'test_precision', 'test_recall', 'train_aP', 'train_precision', 'train_recall'])

# evaluator.store = True
# model.load_state_dict(torch.load(model_file))
# res = evaluator.eval_test(model)
# print(res)



