import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30"


from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

import numpy as np
import utils.data_utils as data_utils
import os
import segmentation.evaluate as evaluate
import torch
import segmentation.AI.datasets as datasets
from PIL import Image
from pathlib import Path

import get_models as get_models
import get_dataloaders as get_dataloaders
import get_transforms as get_transforms

old = 9999999999
new = 1
while old != new:
    old = new
    try:
        tif_folder = '/mnt/DATA1/anton/data/unformatted/GS validation data/untreated_tifs'
        results_folder = "/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/GS_validation_all_untreated_2_copypaste_1806"
        # tif_folder = "/mnt/DATA1/anton/data/unformatted/force_of_infection/tifs"
        # results_folder = "/mnt/DATA1/anton/data/unformatted/force_of_infection/segmentations_2_copypaste_2905"
        model_file = "/mnt/DATA1/anton/pipeline_files_old/results/models/2_copypaste_2905.pth"

        os.makedirs(results_folder, exist_ok=True)

        tif_paths = data_utils.get_paths(tif_folder, extension='.tif')
        seg_stems = [Path(path).stem for path in data_utils.get_paths(results_folder, extension='.png')]
        tif_paths = [path for path in tif_paths if Path(path).stem not in seg_stems]
        print(len(tif_paths))
        new = len(tif_paths)

        ds = datasets.MicroscopyDataset(tif_paths, transform=None, folder_normalize=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, shuffle=True, collate_fn=get_dataloaders.collate_fn_MRCNN_test)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        if not torch.cuda.is_available():
            print("Using non-cuda device: {}".format(device))

        model = get_models.maskrcnn('resnet101')
        model.load_state_dict(torch.load(model_file))
        model.to(device)

        with torch.no_grad():
            model.eval()

            for batch in loader:    
                images = list(image.to(device) for image in batch['X'])

                outputs = model(images)
                pred_masks = [evaluate.collapse_3dmasks(output['masks'], device=device) for output in outputs]

                for pred_mask, file_path in zip(pred_masks, batch['file_paths']):
                    pred_img = Image.fromarray(pred_mask.astype(np.uint8), 'L')
                    pred_img.save(os.path.join(results_folder, Path(file_path).stem + '.png'))

                torch.cuda.empty_cache()
    except:
        pass
