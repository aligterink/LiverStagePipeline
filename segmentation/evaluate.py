import numpy as np
import utils.data_utils as data_utils
import os
from collections import OrderedDict
import torch
from PIL import Image
import multiprocessing

# Evaluate true masks vs predicted masks at provided IoU threshold. 
def eval(true_mask, pred_mask, iou_thres=0.5, print_metrics=False):
    true_cell_ids = np.unique(true_mask)[1:]
    num_true_cells = len(true_cell_ids)
    invalid_true_cell_ids = np.array([0])

    pred_cell_ids = np.unique(pred_mask)[1:]
    num_pred_cells = len(pred_cell_ids)

    tp, IoUs, omission_ratios, oversegmentation_ratios = 0, [], [], []

    for pred_cell_id in pred_cell_ids:
        pred_cell_bool_mask = pred_mask == pred_cell_id
        overlapping_true_cell_ids = np.unique(true_mask[pred_cell_bool_mask])

        overlapping_true_cell_ids = np.setdiff1d(overlapping_true_cell_ids, invalid_true_cell_ids)

        overlapping_true_cell_IoUs = []

        for overlapping_true_cell_id in overlapping_true_cell_ids:
            overlapping_true_cell_mask = true_mask == overlapping_true_cell_id
            
            intersection = np.sum(np.logical_and(pred_cell_bool_mask, overlapping_true_cell_mask))
            union = np.sum(np.logical_or(pred_cell_bool_mask, overlapping_true_cell_mask))
            iou = intersection / union

            overlapping_true_cell_IoUs.append(iou)

        if len(overlapping_true_cell_ids) >= 1:
        
            best_overlapping_true_cell_IoU = max(overlapping_true_cell_IoUs)
            best_overlapping_true_cell_id = overlapping_true_cell_ids[overlapping_true_cell_IoUs.index(best_overlapping_true_cell_IoU)]

            if best_overlapping_true_cell_IoU >= iou_thres:
                overlapping_true_cell_mask = true_mask == best_overlapping_true_cell_id

                omission_ratio = np.sum(np.logical_and(overlapping_true_cell_mask, np.logical_not(pred_cell_bool_mask))) / np.sum(overlapping_true_cell_mask)
                oversegmentation_ratio = np.sum(np.logical_and(pred_cell_bool_mask, np.logical_not(overlapping_true_cell_mask))) / np.sum(pred_cell_bool_mask)

                tp += 1
                invalid_true_cell_ids = np.append(invalid_true_cell_ids, best_overlapping_true_cell_id)
                IoUs.append(best_overlapping_true_cell_IoU)
                omission_ratios.append(omission_ratio)
                oversegmentation_ratios.append(oversegmentation_ratio)

    res = {
        'TP': tp,
        'FP': num_pred_cells - tp,
        'FN': num_true_cells - tp,
        'IoUs': IoUs,
        'Omission ratios': omission_ratios,
        'Oversegmentation ratios': oversegmentation_ratios
    }

    if print_metrics:
        compute_metrics([res])

    return res

def eval_folder(pred_folder, true_folder, results_file, substring=None, seg_channel=None):
    masks_true, masks_pred = data_utils.get_two_sets(true_folder, pred_folder, substring=substring, channel_dir2=seg_channel)
    eval(masks_true, masks_pred, identifier=substring, results_file=results_file)

def eval_multiple_folders(pred_folder, true_folder, results_file, substrings):
    for substring in substrings:
        eval_folder(pred_folder, true_folder, results_file=results_file, substring=substring)
        print('\n')

# For metrics computed over multiple batches, return the totals and averages
def average_evals(res_dicts, decimals=5, print_metrics=True, prefix=None):
    total_cells = sum([res_dict['cells'] for res_dict in res_dicts])
    total_images = sum([res_dict['imgs'] for res_dict in res_dicts])

    total_tp = sum([res_dict['TP'] for res_dict in res_dicts])
    total_fp = sum([res_dict['FP'] for res_dict in res_dicts])
    total_fn = sum([res_dict['FN'] for res_dict in res_dicts])

    precision = round(total_tp / (total_tp + total_fp), decimals) if total_tp else 0
    recall = round(total_tp / (total_tp + total_fn), decimals) if total_tp else 0
    average_precision = round(total_tp / (total_tp + total_fp + total_fn), decimals) if total_tp else 0

    # average_omission_ratio = [res_dict['omission_ratio'] * ]

    res_dict = OrderedDict([('imgs', total_images), ('cells', total_cells), ('TP', total_tp), 
                ('FP', total_fp), ('FN', total_fn), ('precision', precision), ('recall', recall), 
                ('aP', average_precision)])
    
    if prefix:
        res_dict = {f'{prefix}_{k}': v for k, v in res_dict.items()}
    
    if print_metrics:
        print(', '.join([': '.join([k, str(res_dict[k])]) for k in res_dict.keys()]))

    return res_dict

# For metrics computed over multiple batches, return the totals and averages
def compute_metrics(results_2d, decimals=2, print_metrics=True, prefix=None):
    tp = sum([img_res['TP'] for img_res in results_2d])
    fp = sum([img_res['FP'] for img_res in results_2d])
    fn = sum([img_res['FN'] for img_res in results_2d])

    cells = tp + fn
    images = len(results_2d)

    omission_ratios = sum([img_res['Omission ratios'] for img_res in results_2d], [])
    avg_omission_ratio = round(sum(omission_ratios) / cells, decimals) if cells else 0

    oversegmentation_ratios = sum([img_res['Oversegmentation ratios'] for img_res in results_2d], [])
    avg_oversegmentation_ratio = round(sum(oversegmentation_ratios) / cells, decimals) if cells else 0

    precision = round(tp / (tp + fp), decimals) if tp else 0
    recall = round(tp / (tp + fn), decimals) if tp else 0
    aP = round(tp / (tp + fp + fn), decimals) if tp else 0

    res_dict = OrderedDict([('imgs', images), ('cells', cells), ('TP', tp), 
                ('FP', fp), ('FN', fn), ('precision', precision), ('recall', recall), 
                ('aP', aP), ('omission_ratio', avg_omission_ratio), ('oversegmentation_ratio', avg_oversegmentation_ratio)])
    
    if prefix:
        res_dict = {f'{prefix}_{k}': v for k, v in res_dict.items()}
    
    if print_metrics:
        print(', '.join([': '.join([k, str(res_dict[k])]) for k in res_dict.keys()]))

    return res_dict

# Mask R-CNN output contains a dictionary for each image, which contains a [N, 1, H, W] tensor
# where N is the number of masks. Values are between 0-1 and need to be thresholded.
# The code below converts the 3d tensor to a 2d numpy array with labeled masks.
def collapse_3dmasks(tensor, device):
    # convert bool tensor to int tensor
    # tensor = tensor.astype(int)
    tensor = tensor > 0.5
    
    # initialize output tensor with zeros
    out_tensor = torch.zeros(tensor.shape[2:], dtype=int, device=device)
    
    # sort layers by descending size
    layer_sizes = torch.sum(tensor, axis=(2, 3))[:, 0]
    sorted_indices = torch.argsort(layer_sizes, stable=True, descending=True)

    sorted_tensor = tensor[sorted_indices]

    for i in range(sorted_tensor.shape[0]):
        # create a mask for the current layer
        mask = sorted_tensor[i, 0]
        
        # compare overlap of current layer with output tensor
        intersection = torch.sum(torch.logical_and(mask, out_tensor))
        overlap = intersection / torch.sum(mask)
        
        # if there's less than 0.3 IoU, add the layer to the output tensor
        if overlap <= 0.3:
            out_tensor[mask] = i + 1
    
    return out_tensor.cpu().numpy()

class Evaluator():
    def __init__(self, device, test_loaders, store_folder=None, store=False, train_loader=None, iou_thres=0.5, processes=1, testset_names=None):
        self.device = device

        if not isinstance(test_loaders, list):
            self.test_loaders = [test_loaders]
        else:
            self.test_loaders = test_loaders

        self.train_loader = train_loader

        self.store_folder = store_folder
        self.store = store
        self.iou_thres = iou_thres
        self.processes = processes
        self.testset_names = testset_names

    def eval_batch(self, true_masks, outputs, filenames):
        pred_masks = [collapse_3dmasks(output['masks'], device=self.device) for output in outputs]
        
        if self.store:
            for pred_mask, filename in zip(pred_masks, filenames):
                pred_img = Image.fromarray(pred_mask.astype(np.uint8), 'L')
                pred_img.save(os.path.join(self.store_folder, filename + '.png'))

        # Create a pool of worker processes equal to the number of CPUs on the system
        with multiprocessing.Pool(processes=self.processes) as pool: 
            # Submit the processing function for each mask in the batch
            batch_results = [pool.apply_async(eval, args=(true_masks[i].numpy(), pred_masks[i], self.iou_thres)) 
                    for i in range(len(outputs))]

            # Get the results from the worker processes and append them to the batch results list
            batch_results = [res.get() for res in batch_results]

        return batch_results

    def eval_loader(self, model, loader, prefix=''):
        model.to(self.device)
        with torch.no_grad():
            model.eval()

            results = []

            for images, _, true_masks, filenames in loader:
                images = list(image.to(self.device) for image in images)
                outputs = model(images)

                del images
                torch.cuda.empty_cache()

                results += self.eval_batch(true_masks, outputs, filenames)
               
            loader_results = compute_metrics(results, print_metrics=False, prefix=prefix)
        return loader_results

    def eval_train(self, model):
        results = self.eval_loader(model, self.train_loader, prefix='train')
        return results
    
    def eval_test(self, model):
        test_loaders_results = [self.eval_loader(model, test_loader) for test_loader in self.test_loaders]
    
        combined_results = average_evals(test_loaders_results, print_metrics=False, prefix='test') # get totals and averages of results for different loaders

        if len(test_loaders_results) > 1:
            test_loaders_results = [{f'{prefix}_{k}': v for k, v in test_loader_results.items()} for test_loader_results, prefix in zip(test_loaders_results, self.testset_names)]
            for test_loader_results in test_loaders_results:
                combined_results.update(test_loader_results)

        return combined_results
    

    
if __name__ == "__main__":
    true_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_4thchannel_experiments\EXP2_annotation"
    mask_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_4thchannel_experiments\EXP2_watershed"

    results_file = R"C:\Users\anton\Documents\microscopy_data\results\channel4_experiments.csv"

    # masks_true, masks_pred = dataloader.get_two_mask_arrays(true_dir, mask_dir, dir1_is_leading=True)
    # eval(masks_true, masks_pred)

    # eval_files(mask_dir, true_dir, results_file=None)

    # substrings = ["NF135_D3_EXP2_hsp70", "NF135_D5_EXP2_hsp70", "NF135_D7_EXP2_hsp70", "NF175_D3_EXP2_hsp70", "NF175_D5_EXP2_hsp70", "NF175_D7_EXP2_hsp70"]
    substrings = ["D3_135_exp2_hsp", "D5_NF135_exp2_hsp", "D7_135_exp2_hsp", "D3_175_exp2_hsp", "D5_NF175_exp2_hsp", "D7_175_exp2_hsp"]










# import metrics
# import numpy as np
# import dataloader
# import os
# from collections import OrderedDict
# import torch
# from PIL import Image
# import multiprocessing

# # # Evaluate true masks vs predicted masks at provided IoU threshold. 
# # # Expects lists of 2d arrays, not 3d arrays.
# # def eval(masks_true, masks_pred, IoU_threshold=0.5, results_file=None, identifier=' ', decimals=5, print_metrics=True):
# #     res = metrics.average_precision(masks_true, masks_pred, threshold=[IoU_threshold])
# #     print('true masks: {}, pred masks: {}'.format(len(masks_true), len(masks_pred)))

# #     # import matplotlib.pyplot as plt

# #     # for t, p in zip(masks_true, masks_pred):

# #     #     print(np.unique(t), np.unique(p))

# #     #     fig, axs = plt.subplots(2)
# #     #     axs[0].imshow(t)
# #     #     axs[1].imshow(p)

# #     #     plt.savefig("/home/anton/Documents/results/figures/example.png")
# #     #     input("Pres enter...")

# #     total_tp = int(np.sum(res[1]))
# #     total_fp = int(np.sum(res[2]))
# #     total_fn = int(np.sum(res[3]))



# #     precision = round(total_tp / (total_tp + total_fp), decimals) if total_tp else 0
# #     recall = round(total_tp / (total_tp + total_fn), decimals) if total_tp else 0
# #     average_precision = round(total_tp / (total_tp + total_fp + total_fn), decimals) if total_tp else 0
# #     total_imgs = int(len(masks_true))
# #     total_cells = total_tp + total_fn

# #     res_dict = OrderedDict([('identifier', identifier), ('imgs', total_imgs), ('cells', total_cells), ('TP', total_tp), 
# #                 ('FP', total_fp), ('FN', total_fn), ('precision', precision), ('recall', recall), 
# #                 ('aP', average_precision)])

# #     if print_metrics:
# #         print(', '.join([': '.join([k, str(res_dict[k])]) for k in res_dict.keys()]))

# #     if results_file:
# #         with open(results_file, 'r') as f:
# #             is_empty = f.readlines()

# #         with open(results_file, 'a') as f:
# #             if not is_empty:
# #                 f.write(','.join(res_dict.keys()) + '\n')

# #             f.write(','.join(str(r) for r in res_dict.values()) + '\n')
    
# #     return res_dict

# def eval(true_mask, pred_mask, iou_thres=0.5, print_metrics=False):
#     true_cell_ids = np.unique(true_mask)[1:]
#     num_true_cells = len(true_cell_ids)
#     invalid_true_cell_ids = np.array([0])

#     pred_cell_ids = np.unique(pred_mask)[1:]
#     # pred_cell_sizes = [np.sum((pred_mask == x)) for x in pred_cell_ids]
#     num_pred_cells = len(pred_cell_ids)

#     tp, IoUs, omission_ratios, oversegmentation_ratios = 0, [], [], []

#     for pred_cell_id in pred_cell_ids:
#         pred_cell_bool_mask = pred_mask == pred_cell_id
#         overlapping_true_cell_ids = np.unique(true_mask[pred_cell_bool_mask])

#         overlapping_true_cell_ids = np.setdiff1d(overlapping_true_cell_ids, invalid_true_cell_ids)

#         overlapping_true_cell_IoUs = []

#         for overlapping_true_cell_id in overlapping_true_cell_ids:
#             overlapping_true_cell_mask = true_mask == overlapping_true_cell_id
            
#             intersection = np.sum(np.logical_and(pred_cell_bool_mask, overlapping_true_cell_mask))
#             union = np.sum(np.logical_or(pred_cell_bool_mask, overlapping_true_cell_mask))
#             iou = intersection / union

#             overlapping_true_cell_IoUs.append(iou)

#         if len(overlapping_true_cell_ids) >= 1:
        
#             best_overlapping_true_cell_IoU = max(overlapping_true_cell_IoUs)
#             best_overlapping_true_cell_id = overlapping_true_cell_ids[overlapping_true_cell_IoUs.index(best_overlapping_true_cell_IoU)]

#             if best_overlapping_true_cell_IoU >= iou_thres:
#                 overlapping_true_cell_mask = true_mask == best_overlapping_true_cell_id

#                 omission_ratio = np.sum(np.logical_and(overlapping_true_cell_mask, np.logical_not(pred_cell_bool_mask))) / np.sum(overlapping_true_cell_mask)
#                 oversegmentation_ratio = np.sum(np.logical_and(pred_cell_bool_mask, np.logical_not(overlapping_true_cell_mask))) / np.sum(pred_cell_bool_mask)

#                 tp += 1
#                 invalid_true_cell_ids = np.append(invalid_true_cell_ids, best_overlapping_true_cell_id)
#                 IoUs.append(best_overlapping_true_cell_IoU)
#                 omission_ratios.append(omission_ratio)
#                 oversegmentation_ratios.append(oversegmentation_ratio)

#     res = {
#         'TP': tp,
#         'FP': num_pred_cells - tp,
#         'FN': num_true_cells - tp,
#         'IoUs': IoUs,
#         'Omission ratios': omission_ratios,
#         'Oversegmentation ratios': oversegmentation_ratios
#     }

#     if print_metrics:
#         compute_metrics([res])

#     return res

# def eval_folder(pred_folder, true_folder, results_file, substring=None, seg_channel=None):
#     masks_true, masks_pred = dataloader.get_two_sets(true_folder, pred_folder, substring=substring, channel_dir2=seg_channel)
#     eval(masks_true, masks_pred, identifier=substring, results_file=results_file)

# def eval_multiple_folders(pred_folder, true_folder, results_file, substrings):
#     for substring in substrings:
#         eval_folder(pred_folder, true_folder, results_file=results_file, substring=substring)
#         print('\n')

# # # For metrics computed over multiple batches, return the totals and averages
# # def average_evals(res_dicts, decimals=5, print_metrics=True, prefix=None):
# #     total_cells = sum([res_dict['cells'] for res_dict in res_dicts])
# #     total_images = sum([res_dict['imgs'] for res_dict in res_dicts])

# #     total_tp = sum([res_dict['TP'] for res_dict in res_dicts])
# #     total_fp = sum([res_dict['FP'] for res_dict in res_dicts])
# #     total_fn = sum([res_dict['FN'] for res_dict in res_dicts])

# #     precision = round(total_tp / (total_tp + total_fp), decimals) if total_tp else 0
# #     recall = round(total_tp / (total_tp + total_fn), decimals) if total_tp else 0
# #     average_precision = round(total_tp / (total_tp + total_fp + total_fn), decimals) if total_tp else 0

# #     res_dict = OrderedDict([('imgs', total_images), ('cells', total_cells), ('TP', total_tp), 
# #                 ('FP', total_fp), ('FN', total_fn), ('precision', precision), ('recall', recall), 
# #                 ('aP', average_precision)])
    
# #     if prefix:
# #         res_dict = {f'{prefix}_{k}': v for k, v in res_dict.items()}
    
# #     if print_metrics:
# #         print(', '.join([': '.join([k, str(res_dict[k])]) for k in res_dict.keys()]))

# #     return res_dict

# # For metrics computed over multiple batches, return the totals and averages
# def compute_metrics(results_2d, decimals=2, print_metrics=True, prefix=None):
#     tp = sum([img_res['TP'] for img_res in results_2d])
#     fp = sum([img_res['FP'] for img_res in results_2d])
#     fn = sum([img_res['FN'] for img_res in results_2d])

#     cells = tp + fn
#     images = len(results_2d)

#     omission_ratios = sum([img_res['Omission ratios'] for img_res in results_2d], [])
#     avg_omission_ratio = round(sum(omission_ratios) / cells, decimals) if cells else 0

#     oversegmentation_ratios = sum([img_res['Oversegmentation ratios'] for img_res in results_2d], [])
#     avg_oversegmentation_ratio = round(sum(oversegmentation_ratios) / cells, decimals) if cells else 0

#     precision = round(tp / (tp + fp), decimals) if tp else 0
#     recall = round(tp / (tp + fn), decimals) if tp else 0
#     aP = round(tp / (tp + fp + fn), decimals) if tp else 0

#     res_dict = OrderedDict([('imgs', images), ('cells', cells), ('TP', tp), 
#                 ('FP', fp), ('FN', fn), ('precision', precision), ('recall', recall), 
#                 ('aP', aP), ('omission_ratio', avg_omission_ratio), ('oversegmentation_ratio', avg_oversegmentation_ratio)])
    
#     if prefix:
#         res_dict = {f'{prefix}_{k}': v for k, v in res_dict.items()}
    
#     if print_metrics:
#         print(', '.join([': '.join([k, str(res_dict[k])]) for k in res_dict.keys()]))

#     return res_dict

# def collapse_3dmasks(tensor):
#     # Concatenate a tensor of zeros of shape [1, 1, H, W] as images where no masks are predicted
#     # will otherwise throw an error.
#     zeros_tensor = torch.zeros((1, 1, output['masks'].shape[-2], output['masks'].shape[-1]), device=device)
#     mask_3d = torch.cat((output['masks'], zeros_tensor), 0)

#     # Threshold segmentation at 0.5 as per mask R-CNN documentation, then multiply each mask with
#     # its index +1. Then collapse 3d matrix by taking the maximum value along 
#     mask_3d_with_indices = (mask_3d >= 0.5) * torch.arange(1, mask_3d.shape[0] + 1, device=device)[:, None, None, None]
#     mask_2d = torch.amax(mask_3d_with_indices, dim=(0,1))
#     mask_2d = mask_2d.cpu()

# # Mask R-CNN output contains a dictionary for each image, which contains a [N, 1, H, W] tensor
# # where N is the number of masks. Values are between 0-1 and need to be thresholded.
# # The code below converts the 3d tensor to a 2d numpy array with labeled masks.
# def collapse_3dmasks_v2(tensor, device):
#     # convert bool tensor to int tensor
#     # tensor = tensor.astype(int)
#     tensor = tensor > 0.5
    
#     # initialize output tensor with zeros
#     out_tensor = torch.zeros(tensor.shape[2:], dtype=int, device=device)
    
#     # sort layers by descending size
#     layer_sizes = torch.sum(tensor, axis=(2, 3))[:, 0]
#     sorted_indices = torch.argsort(layer_sizes, stable=True, descending=True)

#     sorted_tensor = tensor[sorted_indices]

#     for i in range(sorted_tensor.shape[0]):
#         # create a mask for the current layer
#         mask = sorted_tensor[i, 0]
        
#         # compare overlap of current layer with output tensor
#         intersection = torch.sum(torch.logical_and(mask, out_tensor))
#         # union = torch.sum(torch.logical_or(mask, out_tensor))
#         # iou = intersection / union
#         overlap = intersection / torch.sum(mask)
        
#         # if there's no overlap, add the layer to the output tensor
#         if overlap <= 0.3:
#             out_tensor[mask] = i + 1
    
#     return out_tensor.cpu().numpy()

# class Evaluator():
#     def __init__(self, device, test_loader, store_folder=None, store=False, train_loader=None, iou_thres=0.5, processes=1):
#         self.device = device

#         self.test_loader = test_loader
#         self.train_loader = train_loader

#         self.store_folder = store_folder
#         self.store = store
#         self.iou_thres = iou_thres
#         self.processes = processes

#     def eval_batch(self, true_masks, outputs, filenames):
#         pred_masks = [collapse_3dmasks_v2(output['masks'], device=self.device) for output in outputs]
        
#         if self.store:
#             for pred_mask, filename in zip(pred_masks, filenames):
#                 pred_img = Image.fromarray(pred_mask.astype(np.uint8), 'L')
#                 pred_img.save(os.path.join(self.store_folder, filename + '.png'))

#         # Create a pool of worker processes equal to the number of CPUs on the system
#         with multiprocessing.Pool(processes=self.processes) as pool: 
#             # Submit the processing function for each mask in the batch
#             batch_results = [pool.apply_async(eval, args=(true_masks[i].numpy(), pred_masks[i], self.iou_thres)) 
#                     for i in range(len(outputs))]

#             # Get the results from the worker processes and append them to the batch results list
#             batch_results = [res.get() for res in batch_results]

#         return batch_results

#     def eval_loader(self, model, loader, prefix=''):
#         model.to(self.device)
#         with torch.no_grad():
#             model.eval()

#             results = []

#             for images, _, true_masks, filenames in loader:
#                 images = list(image.to(self.device) for image in images)
#                 outputs = model(images)

#                 del images
#                 torch.cuda.empty_cache()

#                 results += self.eval_batch(true_masks, outputs, filenames)
               
#             loader_results = compute_metrics(results, print_metrics=False, prefix=prefix)
#         return loader_results

#     def eval_train(self, model):
#         results = self.eval_loader(model, self.train_loader, prefix='train')
#         return results
    
#     def eval_test(self, model):
#         results = self.eval_loader(model, self.test_loader, prefix='test')
#         return results
    
#         # combined_results = average_evals(test_loaders_results, print_metrics=False, prefix='test') # get totals and averages of results for different loaders

#         # if len(test_loaders_results) > 1:
#         #     test_loaders_results = [{f'{prefix}_{k}': v for k, v in test_loader_results.items()} for test_loader_results, prefix in zip(test_loaders_results, self.testset_names)]
#         #     for test_loader_results in test_loaders_results:
#         #         combined_results.update(test_loader_results)

#         # return combined_results
    

    
# if __name__ == "__main__":
#     true_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_4thchannel_experiments\EXP2_annotation"
#     mask_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_4thchannel_experiments\EXP2_watershed"

#     results_file = R"C:\Users\anton\Documents\microscopy_data\results\channel4_experiments.csv"

#     # masks_true, masks_pred = dataloader.get_two_mask_arrays(true_dir, mask_dir, dir1_is_leading=True)
#     # eval(masks_true, masks_pred)

#     # eval_files(mask_dir, true_dir, results_file=None)

#     # substrings = ["NF135_D3_EXP2_hsp70", "NF135_D5_EXP2_hsp70", "NF135_D7_EXP2_hsp70", "NF175_D3_EXP2_hsp70", "NF175_D5_EXP2_hsp70", "NF175_D7_EXP2_hsp70"]
#     substrings = ["D3_135_exp2_hsp", "D5_NF135_exp2_hsp", "D7_135_exp2_hsp", "D3_175_exp2_hsp", "D5_NF175_exp2_hsp", "D7_175_exp2_hsp"]

