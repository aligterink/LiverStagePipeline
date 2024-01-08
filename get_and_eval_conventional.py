from segmentation.conventional.cell_watershed import segment_cells_in_folder as seg
from segmentation.evaluate import evaluate_folder
from utils import data_utils
import os

watershed_seg_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/watershed'
AI_segmentation_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/full_augmentation_2'
annotation_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/annotation'

lowres_tif_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/images/lowres'
lowres_target_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/watershed/lowres'
# seg(image_folder=lowres_tif_folder, segmentation_folder=lowres_target_folder, threads=40, channel=1, resize_shape=None, normalize=False, equalize_adapthist=None)


highres_tif_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/images/highres'
highres_target_folder = '/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/watershed/highres'
# seg(image_folder=highres_tif_folder, segmentation_folder=highres_target_folder, threads=40, channel=0, resize_shape=None, normalize=False, equalize_adapthist=None)

annotation_paths = data_utils.get_paths(annotation_folder)
AI_segmentations_folder_paths = data_utils.get_paths(AI_segmentation_folder)

annotation_paths, _ = data_utils.get_common_subset(annotation_paths, AI_segmentations_folder_paths)
watershed_paths = data_utils.get_paths(watershed_seg_folder)

watershed_paths, annotation_paths = data_utils.get_common_subset(watershed_paths, annotation_paths)
groups = [os.path.dirname(path)[len(annotation_folder)+1:].replace('/', ' ') for path in annotation_paths]
evaluate_folder(annotation_paths, watershed_paths, metadata=groups)