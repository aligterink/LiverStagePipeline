import os
import imageio.v3
import utils.data_utils as data_utils
from skimage.measure import regionprops
from skimage import data, util, measure
import numpy as np
import pandas as pd  
from pathlib import Path
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import math
from skimage import filters, morphology, measure, segmentation
from skimage.measure import _regionprops

available_regionprops = _regionprops.PROPS.values()

# Class representing a single image. Methods retrieve cell features.
# We use a class here so we can recycle computed channel- or image-wide features.
class Extractor:
    def __init__(self, mask, intensity_image, name):
        self.mask = mask
        self.intensity_image = intensity_image
        self.name = name
        self.labels = np.unique(mask)[1:].tolist()
        self.num_channels = intensity_image.shape[-1]
        
        self.max_intensities = [None] * self.num_channels

    def get_channel(self, channel):
        if channel == -1:
            return self.mask
        else: 
            return self.intensity_image[:, :, channel]

    # Method for calling our own features
    def call_feature(self, feature, channel):
        feature_method = getattr(self, 'get_' + feature) # let's not name features after other attributes or imports here
        return [feature_method(channel, label) for label in self.labels]
    
    def get_features(self, feature_dict):

        props = pd.DataFrame(columns=['file', 'label'], data=[[self.name, label] for label in self.labels])
        
        for channel in feature_dict.keys():
            channel_img = self.get_channel(channel)
            channel_name = feature_dict[channel][0]
            channel_prop_names = feature_dict[channel][1]

            skimage_channel_prop_names = list(set(channel_prop_names).intersection(available_regionprops))
            if len(skimage_channel_prop_names) > 0:
                skimage_channel_props = pd.DataFrame(measure.regionprops_table(self.mask, channel_img, properties=['label'] + skimage_channel_prop_names))
                if not skimage_channel_props.empty:
                    if channel_name:
                        skimage_channel_props.rename({f: '{}_{}'.format(f, channel_name) for f in skimage_channel_prop_names}, axis=1, inplace=True)
                    props = props.merge(skimage_channel_props, on='label')

            our_prop_names = list(set(channel_prop_names) - set(skimage_channel_prop_names))
            if len(our_prop_names) > 0:
                our_channel_props = pd.DataFrame({f: vals for f, vals in zip(['label'] + our_prop_names, [self.labels] + [self.call_feature(f, channel) for f in our_prop_names])})
                if not our_channel_props.empty:
                    if channel_name:
                        our_channel_props.rename({f: '{}_{}'.format(f, channel_name) for f in our_prop_names}, axis=1, inplace=True)
                    props = props.merge(our_channel_props, on='label')
        
        return props
    
    ##### Channel-wide and image-wide features
    def get_avg_channel_intensity(self, channel):
        if not self.max_intensities[channel]:
            self.max_intensities[channel] = np.average(self.get_channel(channel))
        return self.max_intensities[channel]


    ##### Cell-specific features
    def get_std_intensity(self, channel, label):
        return np.std(self.get_channel(channel)[(self.mask == label)])
    
    def get_max_intensity(self, channel, label):
        return np.max(self.get_channel(channel)[(self.mask == label)])

    def get_num_local_maxima(self, regionmask, intensity_image):
        region_intensity = intensity_image
        region_intensity[~regionmask] = 0

        # side_length = round(math.sqrt(np.sum(regionmask)))
        image_max = ndi.maximum_filter(region_intensity, size=3, mode='constant')

        # coordinates = peak_local_max(region_intensity)

        # # display results
        # fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        # ax = axes.ravel()
        # ax[0].imshow(region_intensity, cmap=plt.cm.gray)
        # ax[0].axis('off')
        # ax[0].set_title('Original')

        # ax[1].imshow(image_max, cmap=plt.cm.gray)
        # ax[1].axis('off')
        # ax[1].set_title('Maximum filter')

        # ax[2].imshow(region_intensity, cmap=plt.cm.gray)
        # ax[2].autoscale(False)
        # ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        # ax[2].axis('off')
        # ax[2].set_title('Peak local max')

        # fig.tight_layout()

        # plt.show()

        # return len(coordinates)

    # 1: skimage features (regionmask, intensityimage)
    # 2: homebrew features
        # 2a: compatible with skimage (regionmask)
        # 2b: not compatible with skimage (x)
    # 3: cell profiler ?

    # This method is overwritten by the regionprops area feature
    def get_area(self, label):
        return np.sum(self.mask == label)

    def get_intensity_sum(self, channel, label):
        return np.sum(self.get_channel(channel)[(self.mask == label)])
    
    # Zonal paper feature: sum(HGs cell intensity) - average_image_hgs_background * cell_area
    def get_normalized_intensity_sum(self, channel, label):
        return self.get_intensity_sum(channel, label) - self.get_avg_channel_intensity(channel) * self.get_area(label)


if __name__ == '__main__':
        img_path = R"C:\Users\anton\Documents\microscopy_data\dataset\images\NF135\D5\2019003_D5_135_hsp_20x_2_series_1_TileScan_001.tif"
        mask_path = R"C:\Users\anton\Documents\microscopy_data\dataset\annotation\NF135\D5\2019003_D5_135_hsp_20x_2_series_1_TileScan_001.png"

        img = np.array(imageio.mimread(img_path)).transpose(1, 2, 0)
        mask = imageio.v3.imread(mask_path)

        x = Extractor(mask, img, '2019003_D5_135_hsp_20x_2_series_1_TileScan_001')

        mask_features = ['area', 'area_convex'] #, 'area_filled', 'axis_major_length', 'axis_minor_length',
                    #  'eccentricity', 'equivalent_diameter_area', 'extent', 'feret_diameter_max', 'orientation',
                    #  'perimeter', 'perimeter_crofton', 'solidity']
        default_channel_features = ['normalized_intensity_sum']

        feature_dict = {-1: ('', ['area', 'area_convex']), 0: ('dapi', ['normalized_intensity_sum']), 1: ('hsp', [])}
        features1 = x.get_features(feature_dict)
        print(features1)
