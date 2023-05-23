from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

import numpy as np
import utils.data_utils as data_utils
import os
import segmentation.evaluate as evaluate
import torch
import segmentation.AI.train as train
import segmentation.AI.dataset as dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import exploration.get_models as get_models
import exploration.get_dataloaders as get_dataloaders
import exploration.get_transforms as get_transforms
import glob
import utils.mask_utils as mask_utils

### Settings
main_folder = "/mnt/DATA1/anton/pipeline_files"
train_name = "partitioned6"

batch_size = 4
dataloader_workers = 4
eval_workers = 4

### Setting up paths
crops_folder = os.path.join(main_folder, "crops")
results_folder = os.path.join(main_folder, "results")
segmentation_folder = os.path.join(results_folder, "segmentations")
logs_folder = os.path.join(results_folder, "logs")
figure_folder = os.path.join(results_folder, "figures")
model_folder = os.path.join(results_folder, "models")

# Create folders if non-existent
for folder in [main_folder, results_folder, crops_folder, segmentation_folder, logs_folder, figure_folder, model_folder]:
    os.makedirs(folder, exist_ok=True)

# Generate paths for current trainloop
log_file = os.path.join(logs_folder, train_name + '.log')
model_file = os.path.join(model_folder, train_name + '.pth')
figure_file = os.path.join(figure_folder, train_name + '.png')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if not torch.cuda.is_available():
    print("Using non-cuda device: {}".format(device))
    
# Generate new crops of training set cells
crop_files = glob.glob(folder + '/*.tif')
for f in crop_files:
    os.remove(f)
dummy_train_loader, _, _ = get_dataloaders.v3(None, None, None, batch_size, dataloader_workers)
mask_utils.extract_crops_from_loader(loader=dummy_train_loader, folder=crops_folder)

train_transform, train_individual_transform, test_transform = get_transforms.v1(crops_folder=crops_folder)
train_loader, test_loader, testset_names = get_dataloaders.v3(train_transform, train_individual_transform, test_transform, batch_size, dataloader_workers, show=False)
evaluator = evaluate.Evaluator(device, train_loader=train_loader, test_loaders=test_loader, store_folder=segmentation_folder, processes=eval_workers, testset_names=testset_names)

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



