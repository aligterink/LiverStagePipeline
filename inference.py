import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

import numpy as np
import utils.data_utils as data_utils
import os
import segmentation.evaluate as evaluate
import torch
import segmentation.AI.dataset as dataset
from PIL import Image
from pathlib import Path

import get_models as get_models
import get_dataloaders as get_dataloaders
import get_transforms as get_transforms



tif_folder = "/mnt/DATA1/anton/data/unformatted/GS validation data/untreated_tifs/Experiment 2019014"
results_folder = "/mnt/DATA1/anton/data/unformatted/GS validation data/untreated_segmentations_2_copypaste_2905/2019014"
model_file = "/mnt/DATA1/anton/pipeline_files/results/models/2_copypaste_2905.pth"

os.makedirs(results_folder, exist_ok=True)

tif_paths = data_utils.get_paths(tif_folder, extension='.tif')
seg_stems = [Path(path).stem for path in data_utils.get_paths(results_folder, extension='.png')]
tif_paths = [path for path in tif_paths if Path(path).stem not in seg_stems]
print(len(tif_paths))

transform = get_transforms.get_test_transform()
ds = dataset.MicroscopyDatasetNoTarget(tif_paths, transform=transform, set_folder_ranges=False)
loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, shuffle=True, collate_fn=lambda x:list(zip(*x)))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
device = torch.device("cpu")
if not torch.cuda.is_available():
    print("Using non-cuda device: {}".format(device))

model = get_models.maskrcnn('resnet101')
model.load_state_dict(torch.load(model_file))
model.to(device)
model.eval()


for images, filenames in loader:
    
    images = list(image.to(device) for image in images)
    outputs = model(images)
    pred_masks = [evaluate.collapse_3dmasks(output['masks'], device=device) for output in outputs]

    for pred_mask, filename in zip(pred_masks, filenames):
        pred_img = Image.fromarray(pred_mask.astype(np.uint8), 'L')
        pred_img.save(os.path.join(results_folder, filename + '.png'))

