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
    




