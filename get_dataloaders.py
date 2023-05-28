import numpy as np
import utils.data_utils as data_utils
import os
import segmentation.evaluate as evaluate
import torch
import segmentation.AI.train as train
import segmentation.AI.dataset as dataset
from utils import mask_utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import get_models as get_models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob

# folder_4thexp = "/home/anton/Documents/microscopy_data/fourth_channel_experiments"
# folder_NF54 = "/home/anton/Documents/microscopy_data/NF54_annotation"

# folder_3564_tifs = "/home/anton/Documents/microscopy_data/3564_annotated_subset/Annie_subset_tiffs"
# folder_3564_anno = "/home/anton/Documents/microscopy_data/3564_annotated_subset/Annie_subset_annotations"

# Filters out images with empty masks which are not accepted by mask R-CNN
def collate_fn(batch):
    return list(zip(*list(filter(lambda x : x if(x is not None) else None, batch))))

def show_dataset(dataset):
    ids = np.arange(0, dataset.__len__())
    np.random.shuffle(ids)
    for idx in ids:
        dataset.__getitem__(idx, savepath="/mnt/DATA1/anton/pipeline_files/results/figures/dataset.png")
        # input('Showing image {}. Press enter to continue...'.format(idx))

def get_meanstd(dataset, channel):
    ids = np.arange(0, dataset.__len__())
    np.random.shuffle(ids)
    c = []
    for idx in ids:
        image, target, labeled_mask, name = dataset.__getitem__(idx)
        print(image.shape)
        c.append(image[channel,:,:])

    cstack = torch.stack(c, 0)
    print(cstack.shape, torch.mean(cstack), torch.std(cstack))

def v1(train_transform, train_individual_transform, test_transform, batch_size, num_workers, show=False):

    train_paths_4thexp = data_utils.get_two_sets(folder_4thexp, folder_4thexp, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
    train_paths_NF54 = data_utils.get_two_sets(folder_NF54, folder_NF54, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
    trainset = dataset.MicroscopyDataset(train_paths_4thexp[0] + train_paths_NF54[0], train_paths_4thexp[1] + train_paths_NF54[1], 
                                        filter_empty=True, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn)

    substrings = {'D3_54': 'NF54 D3', 'D5_54': 'NF54 D5', 'D7_54': 'NF54 D7',
                'D3_135': 'NF135 D3', 'D5_135': 'NF135 D5', 'D7_135': 'NF135 D7',
                'D3_175': 'NF175 D3', 'D5_175': 'NF175 D5', 'D7_175': 'NF175 D7'}
    test_path_sets = [data_utils.get_two_sets(folder_3564_tifs, folder_3564_anno, common_subset=True, substring=k, extension_dir1='.tif', 
                                            extension_dir2='.png', return_paths=True) for k in substrings.keys()]
    test_sets = [dataset.MicroscopyDataset(tps[0], tps[1], filter_empty=False, transform=test_transform) for tps in test_path_sets]
    test_loaders = [torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x:list(zip(*x))) for test_set in test_sets]
    return train_loader, test_loaders


def v2(train_transform, train_individual_transform, test_transform, batch_size, num_workers, show=False):

    paths_4thexp = data_utils.get_two_sets(folder_4thexp, folder_4thexp, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
    paths_NF54 = data_utils.get_two_sets(folder_NF54, folder_NF54, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
    paths_3654 = data_utils.get_two_sets(folder_3564_tifs, folder_3564_anno, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
    
    tif_paths = paths_4thexp[0] + paths_NF54[0] + paths_3654[0]
    seg_paths = paths_4thexp[1] + paths_NF54[1] + paths_3654[1]

    tif_train, tif_test, seg_train, seg_test = train_test_split(tif_paths, seg_paths, test_size=0.2, random_state=43, shuffle=True)
    tif_train, seg_train = tif_train[0:10], seg_train[0:10] ### for testing purposes
    trainset = dataset.MicroscopyDataset(tif_train, seg_train, filter_empty=True, transform=train_transform, individual_transform=train_individual_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn)
    
    if show:
        show_dataset(trainset)

    tif_test, seg_test = tif_test[0:10], seg_test[0:10]
    testset = dataset.MicroscopyDataset(tif_test, seg_test, filter_empty=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x:list(zip(*x)))
    test_loaders = [test_loader, test_loader]

    return train_loader, test_loaders

def v3(train_transform, train_individual_transform, test_transform, batch_size, num_workers, crops_folder, watershed_crops_folder, show=False):

    testset_names, test_sets, X_train, y_train, y_train_watershed = [], [], [], [], []

    strains = ['NF54', 'NF135', 'NF175']
    days = ['D3', 'D5', 'D7']

    dataset_path = "/mnt/DATA1/anton/data/lowres_dataset_selection"
    watershed_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/watershed"

    for strain in strains:
        imgs_path = dataset_path + '/' + 'images' + '/' + strain
        anno_path = dataset_path + '/' + 'annotation' + '/' + strain
        for day in days:
            day_imgs_path = imgs_path + '/' + day
            day_anno_path = anno_path + '/' + day

            strain_day_paths = data_utils.get_two_sets(day_imgs_path, day_anno_path, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)
            strainday_X_train, strainday_X_test, strainday_y_train, strainday_y_test = train_test_split(strain_day_paths[0], strain_day_paths[1], test_size=0.2, random_state=62, shuffle=True)

            testset_names.append(strain + '_' + day)
            X_train += strainday_X_train
            y_train += strainday_y_train

            test_sets.append(dataset.MicroscopyDataset(strainday_X_test, strainday_y_test, filter_empty=False, transform=test_transform))

    dummy_trainset = dataset.MicroscopyDataset(X_train, y_train, filter_empty=True, transform=None, individual_transform=None)
    dummy_train_loader = torch.utils.data.DataLoader(dummy_trainset, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn)

    X_train_watershed, y_train_watershed = data_utils.get_common_subset(X_train, data_utils.get_paths(watershed_path, extension='.png'))

    from pathlib import Path
    for a,b in zip(X_train_watershed, y_train_watershed):
        assert Path(a).stem == Path(b).stem, '{} != {}'.format(a, b)

    trainset_watershed = dataset.MicroscopyDataset(X_train_watershed, y_train_watershed, filter_empty=True, transform=None, individual_transform=None)
    train_loader_watershed = torch.utils.data.DataLoader(trainset_watershed, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn)
    
    trainset = dataset.MicroscopyDataset(X_train, y_train, filter_empty=True, transform=train_transform, individual_transform=train_individual_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,shuffle=True, collate_fn=collate_fn)

    if show:
        show_dataset(trainset)

    test_loaders = [torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x:list(zip(*x))) for test_set in test_sets]
    return train_loader, test_loaders, testset_names, dummy_train_loader, train_loader_watershed


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
    




