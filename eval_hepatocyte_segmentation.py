from segmentation.evaluate import evaluate_images
from utils import data_utils
from segmentation.conventional.cell_watershed import segment_cells_in_folder as seg
import imageio.v3
import numpy as np

tif_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/images'
gt_hepatocyte_folder = '/mnt/DATA1/anton/data/hepatocyte_annotated_dataset/annotation'
gt_parasite_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/annotation'
hepatocyte_seg_folder = '/mnt/DATA1/anton/data/hepatocyte_annotated_dataset/hepatocyte_watershed'

# segment
# tif_paths, parasite_seg_paths = data_utils.get_two_sets(tif_folder, gt_parasite_folder, common_subset=True, return_paths=True)
# channels = [0 if 'lowres' in tp else 2 for tp in tif_paths]
# seg(image_paths=tif_paths, segmentation_folder=hepatocyte_seg_folder, parasite_mask_paths=parasite_seg_paths, threads=40, channels=channels, resize_shape=None, normalize=False, equalize_adapthist=None)

# eval
gt_mask_paths, hep_pred_paths = data_utils.get_two_sets(gt_hepatocyte_folder, hepatocyte_seg_folder, common_subset=True, return_paths=True)
metadata = ['lowres' if 'lowres' in gt_path else 'highres' for gt_path in gt_mask_paths]
evaluate_images(gt_mask_paths, predicted_masks=hep_pred_paths, metadata=metadata)

# _, hep_seg_paths = data_utils.get_two_sets(tif_folder, gt_hepatocyte_folder, common_subset=True, return_paths=True)

# for p in hep_seg_paths:
#     m = imageio.imread(p)
#     print(len(np.unique(m)))