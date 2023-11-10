import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

import numpy as np
from utils import data_utils
import os
from collections import OrderedDict
import torch
from PIL import Image
import multiprocessing
import imageio.v3
from tqdm import tqdm
from pathlib import Path
from segmentation.AI import logger


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_iou_matrix(predicted_mask, ground_truth_mask):
    # ax1 = GT, ax2 = pred
    true_cell_ids = np.unique(ground_truth_mask)[1:]
    pred_cell_ids = np.unique(predicted_mask)[1:]
    num_true_objects = len(true_cell_ids)
    num_pred_objects = len(pred_cell_ids)

    iou_matrix = np.zeros(shape=(num_true_objects, num_pred_objects))

    for pred_id_index, pred_id in enumerate(pred_cell_ids):
        for true_id in np.setdiff1d(np.unique(ground_truth_mask[(predicted_mask == pred_id)]), np.array([0])):
            true_id_index = np.where(true_cell_ids == true_id)
            iou_matrix[true_id_index, pred_id_index] = calculate_iou((predicted_mask == pred_id), (ground_truth_mask == true_id))
    return iou_matrix

def evaluate_iou_matrix(iou_matrix):
    ious = []
    while np.prod(iou_matrix.shape) > 0 and np.max(iou_matrix) > 0:
        max_iou = np.max(iou_matrix)
        ious.append(max_iou)

        max_indices = [oi[0] for oi in np.where(iou_matrix == max_iou)]
        

        iou_matrix = np.delete(iou_matrix, max_indices[0], axis=0)
        iou_matrix = np.delete(iou_matrix, max_indices[1], axis=1)

    fn, fp = iou_matrix.shape
    return ious, fp, fn

def evaluate_image(gt_mask, pred_mask, metadata):
    iou_matrix = calculate_iou_matrix(pred_mask, gt_mask)
    ious, fp, fn = evaluate_iou_matrix(iou_matrix)
    return (metadata, ious, fp, fn)

def compute_results(ious, fp, fn, iou_thresholds=[0.25, 0.5, 0.75], decimals=3):
    metrics = {}
    for iou_threshold in iou_thresholds:
        iou_tp = np.sum(ious >= iou_threshold)
        iou_fp = fp + np.sum(ious < iou_threshold)
        iou_fn = fn + np.sum(ious < iou_threshold)

        precision = iou_tp / (iou_tp + iou_fp)
        recall = iou_tp / (iou_tp + iou_fn)
        f1 = (2*iou_tp) / ((2*iou_tp) + iou_fp + iou_fn)

        metrics.update({
            'precision@{}'.format(iou_threshold): round(precision, decimals),
            'recall@{}'.format(iou_threshold): round(recall, decimals),
            'F1@{}'.format(iou_threshold): round(f1, decimals),
            'mmIoU': round(np.mean(ious), decimals)
                        })

    return metrics


def aggregate_results(results):
    groups = np.array([r[0] for r in results])
    ious = [r[1] for r in results]
    fp = np.array([r[2] for r in results])
    fn = np.array([r[3] for r in results])
    unique_groups = np.unique(groups)

    metrics = {}

    for group in unique_groups:
        group_mask = groups == group
        group_ious = np.concatenate([ious[i] for i in range(len(ious)) if group_mask[i]])
        group_fp = np.sum(fp[group_mask])
        group_fn = np.sum(fn[group_mask])
        metrics.update({group: compute_results(group_ious, group_fp, group_fn)}) # compute results per group

    metrics.update(compute_results(np.concatenate(ious), np.sum(fp), np.sum(fn))) # compute global results


    return metrics

def evaluate_images(ground_truth_masks, predicted_masks, metadata=None):
    results = []
    for i in range(len(ground_truth_masks)):
        gt_mask = imageio.v3.imread(ground_truth_masks[i])
        pred_mask = imageio.v3.imread(predicted_masks[i])

        results += [evaluate_image(gt_mask=gt_mask, pred_mask=pred_mask, metadata=metadata[i] if metadata else None)]
    metrics = aggregate_results(results)
    print(metrics)
    return metrics
    

class Inferenceinator():
    def __init__(self, loader, model_output_parser, device, log_file=None, IoU_threshold=0.5, processes=12, evaluate=True):
        self.loader = loader
        self.model_output_parser = model_output_parser
        self.IoU_threshold = IoU_threshold
        self.processes = processes
        self.logger = logger.Logger(log_file=log_file, overwrite=False, track_metric=False) if log_file else None
        self.evaluate = evaluate
        self.device = device

    def __call__(self, model, store_folder=None):
        results = []
        model.eval()
        with torch.no_grad():

            for batch in tqdm(self.loader, leave=False, desc='Inference' if store_folder else 'Evaluating'):
                batch = data_utils.move_to_device(batch, self.device)

                pred_masks_2d = self.model_output_parser(model, {'images': batch['images']})

                # Rescale predicted mask if ground truth mask is of different shape
                for i, pred_mask_2d in enumerate(pred_masks_2d):
                    if batch['original_sizes'][i] != pred_mask_2d.shape:
                        pred_masks_2d[i] = data_utils.resize_mask(pred_mask_2d, shape=batch['original_sizes'][i])
                if store_folder:
                    for pred_mask, image_path in zip(pred_masks_2d, batch['file_paths']):
                        save_path = os.path.join(store_folder, Path(image_path).stem + '.tif')
                        data_utils.save_image(pred_mask.astype(np.uint16), save_path)

                if self.evaluate and batch['masks_2d']:
                    with multiprocessing.Pool(processes=self.processes) as pool: # create a pool of worker processes
                        # Submit the processing function for each mask in the batch
                        batch_results = [pool.apply_async(evaluate_image, args=(batch['masks_2d'][i].cpu().numpy(), pred_masks_2d[i], batch['groups'][i])) for i in range(len(batch['masks_2d']))]
                        # for i in range(len(batch['masks_2d'])): # for debugging purposes
                        #     results += [evaluate_image(batch['masks_2d'][i].cpu().numpy(), pred_masks_2d[i], batch['groups'][i])]
                
                        # Get the results from the worker processes and append them to the batch results list
                        results += [res.get() for res in batch_results]

        if self.evaluate:
            metrics = aggregate_results(results)
            if self.logger:
                self.logger.end()
                self.logger.log(metrics)
            return metrics
    
    





if __name__ == "__main__":
    seg_dir = "/mnt/DATA1/anton/pipeline_files/results/segmentation_collection/partitioned7/"

    true_dir = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation/NF175"
    # paths_masks_true, paths_masks_pred = data_utils.get_two_sets(true_dir, seg_dir, common_subset=True, return_paths=True, extension_dir1='.png', extension_dir2='.png')
    # results = [eval(imageio.v3.imread(true_path), imageio.v3.imread(pred_path)) for true_path, pred_path in zip(paths_masks_true, paths_masks_pred)]
    # metrics = compute_metrics(results)

    m = compute_iou_matrix(np.array([[0, 1, 2], [0, 0, 0]]), np.array([[0, 1, 2], [3, 4, 5]]))
    tp, fp, fn = evaluate_iou_matrix(m)
    print(tp, fp, fn)


