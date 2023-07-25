import numpy as np
import utils.data_utils as data_utils
import os
import segmentation.evaluate as evaluate
import torch
import segmentation.AI.train as train
import segmentation.AI.datasets as datasets
from utils import mask_utils, cell_viewer
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import get_models as get_models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
from transformers import MaskFormerImageProcessor

processor = MaskFormerImageProcessor()

# Filters out images with empty masks which are not accepted by mask R-CNN
def collate_fn_MRCNN_train(batch):
    # return list(zip(*list(filter(lambda x : x if(x is not None) else None, batch))))
    filtered_batch = [sample for sample in batch if 'masks' in sample.keys() and len(sample['masks'].size()) > 0]

    new_batch = {
        'X': [sample['image'] for sample in filtered_batch],
        'y': [{k: sample[k] for k in ('boxes','labels','masks')} for sample in filtered_batch],
        'file_paths': [sample['file_path'] for sample in filtered_batch],
        'masks_2d': [sample['mask_2d'] for sample in filtered_batch]
    }
    return new_batch

def collate_fn_MRCNN_test(batch):
    # new_batch = {
    #     'X': [sample['image'] for sample in batch],
    #     'y': [{k: sample[k] for k in ('boxes','labels','masks')} for sample in batch],
    #     'file_names': [sample['name'] for sample in batch],
    #     'masks_2d': [sample['mask_2d'] for sample in batch]
    # }
    new_batch = {
        'X': [sample['image'] for sample in batch if 'image' in sample],
        'y': [{k: sample[k] for k in ('boxes', 'labels', 'masks') if k in sample} for sample in batch],
        'file_paths': [sample['file_path'] for sample in batch if 'file_path' in sample],
        'masks_2d': [sample['mask_2d'] for sample in batch if 'mask_2d' in sample]
    }
    return new_batch

def collate_fn_maskformer(batch):
    pixel_values = torch.stack([example["image"] for example in batch])
    pixel_mask = torch.stack([example["mask_2d"] for example in batch])
    class_labels = [example["labels"] for example in batch]
    mask_labels = [example["mask_3d"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}


def v3(train_transform, train_individual_transform, batch_size, num_workers, crops_folder):
    collate_fn_train = collate_fn_maskformer
    collate_fn_test = collate_fn_maskformer
    compute_3d_mask = True

    testset_names, test_sets, X_train, y_train = [], [], [], []

    strains = ['NF54', 'NF135', 'NF175']
    days = ['D3', 'D5', 'D7']

    dataset_path = "/mnt/DATA1/anton/data/lowres_dataset_selection"

    for strain in strains:
        imgs_path = dataset_path + '/' + 'images' + '/' + strain
        anno_path = dataset_path + '/' + 'annotation' + '/' + strain
        for day in days:
            day_imgs_path = imgs_path + '/' + day
            day_anno_path = anno_path + '/' + day

            strain_day_paths = data_utils.get_two_sets(day_imgs_path, day_anno_path, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
            strainday_X_train, strainday_X_test, strainday_y_train, strainday_y_test = train_test_split(strain_day_paths[0], strain_day_paths[1], test_size=0.2, random_state=64, shuffle=True)

            testset_names.append(strain + '_' + day)
            X_train += strainday_X_train
            y_train += strainday_y_train

            test_sets.append(datasets.MicroscopyDataset(strainday_X_test, strainday_y_test, filter_empty=False, transform=None, folder_normalize=True, compute_3d_mask=compute_3d_mask))

    dummy_trainset = datasets.MicroscopyDataset(X_train, y_train, filter_empty=True, transform=None, individual_transform=None, compute_3d_mask=compute_3d_mask)
    dummy_train_loader = torch.utils.data.DataLoader(dummy_trainset, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn_train)

    trainset = datasets.MicroscopyDataset(X_train, y_train, filter_empty=True, transform=train_transform, individual_transform=train_individual_transform, folder_normalize=True, compute_3d_mask=compute_3d_mask)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn_train)

    trainset[0]
    # cell_viewer.show_dataset(trainset)

    test_loaders = [torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn_test) for test_set in test_sets]
    return train_loader, test_loaders, testset_names, dummy_train_loader


def plot_set(set):
    minlist_blue, p2plist_blue, minlist_red, p2plist_red = [], [], [], []
    median_blue, median_red = [], []
    maxlist_blue, maxlist_red = [], []

    for x in trainset:
        bluechan, redchan = x[0][0,:,:].numpy(), x[0][1,:,:].numpy()
        minlist_blue.append(bluechan.min())
        minlist_red.append(redchan.min())
        p2plist_blue.append(bluechan.ptp())
        p2plist_red.append(redchan.ptp())

        median_blue.append(np.median(bluechan))
        median_red.append(np.median(redchan))
        maxlist_red.append(redchan.max())
        maxlist_blue.append(bluechan.max())

        plt.hist(redchan.flatten(), bins=200)
        plt.ylim([0,2000])
        plt.savefig("/home/anton/Documents/results/figures/example.png")
        input('Pres enter...')

    print('Red min: {}, red max: {}, blue min: {}, blue max: {}'.format(min(minlist_red), max(maxlist_red), min(minlist_blue), max(maxlist_blue)))


    plt.plot(minlist_blue, label='blue min')
    plt.plot(minlist_red, label='red min')
    plt.plot(p2plist_blue, label='p2p blue')
    plt.plot(p2plist_red, label='p2p red')
    plt.plot(median_blue, label='median blue')
    plt.plot(median_red, label='median red')


    plt.legend()
    plt.savefig("/home/anton/Documents/results/figures/example.png")
    




