import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[0] + 'LiverStagePipeline')

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

from segmentation.AI import logger

import numpy as np
import utils.data_utils as data_utils
import os
import segmentation.evaluate as evaluate
import torch
import segmentation.AI.train as train
import segmentation.AI.dataset as dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import get_models as get_models
import get_dataloaders as get_dataloaders
import get_transforms as get_transforms
import glob
import utils.mask_utils as mask_utils
from sklearn.model_selection import KFold
torch.manual_seed(2)

### Settings
main_folder = "/mnt/DATA1/anton/pipeline_files"
train_name = "no_copypaste"

dataset_path = "/mnt/DATA1/anton/data/lowres_dataset_selection"
watershed_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/watershed"

batch_size = 4
dataloader_workers = 4
eval_workers = 4
    
### Setting up paths
results_folder = os.path.join(main_folder, "results")
segmentation_folder = os.path.join(results_folder, "segmentations")
logs_folder = os.path.join(results_folder, "logs")
figure_folder = os.path.join(results_folder, "figures")
model_folder = os.path.join(results_folder, "models")
crops_folder = os.path.join(main_folder, "crops")
watershed_crops_folder = os.path.join(main_folder, "watershed_crops")

# Create folders if non-existent
for folder in [main_folder, results_folder, crops_folder, watershed_crops_folder, segmentation_folder, logs_folder, figure_folder, model_folder]:
    os.makedirs(folder, exist_ok=True)

# Generate paths for current trainloop
log_file = os.path.join(logs_folder, train_name + '.log')
model_file = os.path.join(model_folder, train_name + '.pth')
figure_file = os.path.join(figure_folder, train_name + '.png')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if not torch.cuda.is_available():
    print("Using non-cuda device: {}".format(device))

train_transform, train_individual_transform, test_transform = get_transforms.v1(crops_folder=crops_folder, watershed_crop_folder=watershed_crops_folder)
train_loader, test_loaders, testset_names, dummy_loader, train_loader_watershed = get_dataloaders.v3(train_transform, train_individual_transform, test_transform, batch_size, dataloader_workers, crops_folder, watershed_crops_folder, show=False)






# dataset_names, path_sets = [], []

# strains, days = ['NF54', 'NF135', 'NF175'], ['D3', 'D5', 'D7']
# for strain, day in [(strain, day) for strain in strains for day in days]:
#     imgs_path = os.path.join(dataset_path, 'images', strain, day)
#     anno_path = os.path.join(dataset_path, 'annotation', strain, day)
#     paths = data_utils.get_two_sets(imgs_path, anno_path, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
#     # datasets.append(dataset.MicroscopyDataset(paths[0], paths[1], filter_empty=False))
#     dataset_names.append('{}_{}'.format(strain, day))
#     path_sets.append(paths)

# def kfold_CV(path_sets, dataset_names, kfold=5):
#     kf = KFold(n_splits=kfold)

#     for path_set in path_sets:
#         kf.get_n_splits(path_set)
            
#     strainday_X_train, strainday_X_test, strainday_y_train, strainday_y_test = train_test_split(strain_day_paths[0], strain_day_paths[1], test_size=0.2, random_state=62, shuffle=True)

#     testset_names.append(strain + '_' + day)
#     X_train += strainday_X_train
#     y_train += strainday_y_train

#     test_sets.append(dataset.MicroscopyDataset(strainday_X_test, strainday_y_test, filter_empty=False, transform=test_transform))

#     dummy_trainset = dataset.MicroscopyDataset(X_train, y_train, filter_empty=True, transform=None, individual_transform=None)
#     dummy_train_loader = torch.utils.data.DataLoader(dummy_trainset, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn)

#     X_train_watershed, y_train_watershed = data_utils.get_common_subset(X_train, data_utils.get_paths(watershed_path, extension='.png'))

#     from pathlib import Path
#     for a,b in zip(X_train_watershed, y_train_watershed):
#         assert Path(a).stem == Path(b).stem, '{} != {}'.format(a, b)

#     trainset_watershed = dataset.MicroscopyDataset(X_train_watershed, y_train_watershed, filter_empty=True, transform=None, individual_transform=None)
#     train_loader_watershed = torch.utils.data.DataLoader(trainset_watershed, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn)
    
#     trainset = dataset.MicroscopyDataset(X_train, y_train, filter_empty=True, transform=train_transform, individual_transform=train_individual_transform)
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn)

#     if show:
#         show_dataset(trainset)

#     test_loaders = [torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x:list(zip(*x))) for test_set in test_sets]
#     return train_loader, test_loaders, testset_names, dummy_train_loader, train_loader_watershed













evaluator = evaluate.Evaluator(device, train_loader=train_loader, test_loaders=test_loaders, store_folder=segmentation_folder, processes=eval_workers, testset_names=testset_names)

# Generate new crops of training set cells
crop_files = glob.glob(crops_folder + '/*.tif')
for f in crop_files:
    os.remove(f)
mask_utils.extract_crops_from_loader(loader=dummy_loader, folder=crops_folder)

# # Generate watershed augmentation crops
# watershed_crop_files = glob.glob(watershed_crops_folder + '/*.tif')
# for f in watershed_crop_files:
#     os.remove(f)
# mask_utils.extract_non_overlapping_crops_from_loader(loader=train_loader_watershed, val_loader=dummy_loader, folder=watershed_crops_folder)

# Define the model
model = get_models.maskrcnn('resnet101')
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.00005, amsgrad=True)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

train.train(model, train_loader, evaluator, num_epochs=100, optimizer=optimizer, scheduler=scheduler, print_every=1, 
            device=device, log_file=log_file, figure_file=figure_file, model_path=model_file, eval_trainloader=False, 
            metric_for_best='test_aP', 
            printed_vals=['test_aP', 'test_precision', 'test_recall', 'train_aP', 'train_precision', 'train_recall'])

evaluator.store = True
model.load_state_dict(torch.load(model_file))
res = evaluator.eval_test(model)
print(res)



