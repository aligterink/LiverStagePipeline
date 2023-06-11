import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

import imageio.v3
from utils import data_utils, mask_utils
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
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, distance_transform_cdt
from scipy.spatial import distance
import re
import math
from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # disables the warning

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
        self.avg_intensities = [None] * self.num_channels
        self.parasite_centre_distance_matrix = None

    def get_channel(self, channel):
        if channel == 'mask':
            return self.mask
        else: 
            return self.intensity_image[:, :, channel]

    # Method for calling our own features
    def call_feature(self, feature, channel):
        # Check if there is a '(n)' pattern, if so supply feature method with additional n parameter
        pattern = r'\((\d+)\)'
        match = re.search(pattern, feature)
        if match:
            n = match.group(1)  # Extract the value of n from the match
            feature = re.sub(pattern, 'n', feature)  # Replace (n) with n

        feature_method = getattr(self, 'get_' + feature) # let's not name features after other attributes or imports here
        return [feature_method(channel, label, int(n)) if match else feature_method(channel, label) for label in self.labels]
    
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
                our_channel_props = pd.DataFrame({f.replace('(', '').replace(')', ''): vals for f, vals in zip(['label'] + our_prop_names, [self.labels] + [self.call_feature(f, channel) for f in our_prop_names])})
                if not our_channel_props.empty:
                    if channel_name:
                        our_channel_props.rename({f.replace('(', '').replace(')', ''): '{}_{}'.format(f.replace('(', '').replace(')', ''), channel_name) for f in our_prop_names}, axis=1, inplace=True)
                    props = props.merge(our_channel_props, on='label')
        
        return props
    
    ########################## Channel-wide and image-wide features

    ######## Mask features
    def _get_parasite_centre_distance_matrix(self, channel):
        if self.parasite_centre_distance_matrix is None:
            coords = [self.get_centre_coords('mask', label) for label in self.labels]
            self.parasite_centre_distance_matrix = distance.cdist(coords, coords, 'euclidean')
        return self.parasite_centre_distance_matrix
    
    ###### Non-mask features
    def _get_avg_channel_intensity(self, channel):
        if not self.avg_intensities[channel]:
            self.avg_intensities[channel] = np.average(self.get_channel(channel))
        return self.avg_intensities[channel]
    

    ########################## Cell-specific features

    ###### Mask features
    def get_centre_coords(self, channel, label):
        return ndimage.center_of_mass(self.mask == label)
    
    def get_avg_n_NN_distance(self, channel, label, n):
        cell_distances = self._get_parasite_distances(channel, label)
        return np.mean(np.sort(cell_distances)[:n])
    
    def get_parasites_within_npx(self, channel, label, n):
        cell_distances = self._get_parasite_distances(channel, label)
        return np.sum(cell_distances <= n)

    # This method is overwritten by the regionprops area feature
    def get_area(self, channel, label):
        assert channel == 'mask', 'channel should be \'mask\''
        return np.sum(self.mask == label)

    def _get_parasite_distances(self, channel, label):
        assert channel == 'mask', 'channel should be \'mask\''
        distance_matrix = self._get_parasite_centre_distance_matrix('mask')
        label_index = self.labels.index(label)
        distances_to_cell = distance_matrix[label_index]
        distances_to_cell = np.delete(distances_to_cell, label_index) # remove distance of parasite to itself
        return distances_to_cell
    

    ###### Non-mask features
    def get_avg_intensity(self, channel, label):
        return np.mean(self.get_channel(channel)[(self.mask == label)])
    
    def get_std_intensity(self, channel, label):
        return np.std(self.get_channel(channel)[(self.mask == label)])
    
    def get_max_intensity(self, channel, label):
        return np.max(self.get_channel(channel)[(self.mask == label)])
    
    def get_min_intensity(self, channel, label):
        return np.min(self.get_channel(channel)[(self.mask == label)])
    
    def get_intensity_sum(self, channel, label):
        return np.sum(self.get_channel(channel)[(self.mask == label)])

    def get_num_local_maxima(self, channel, label):
        cell_mask = (self.mask == label)
        intensity_image = self.get_channel(channel).copy()
        region_intensity = intensity_image
        region_intensity[~cell_mask] = 0
        print(label, np.sum(cell_mask))

        # side_length = round(math.sqrt(np.sum(cell_mask)))
        image_max = ndi.maximum_filter(region_intensity, size=3, mode='constant')

        coordinates = peak_local_max(region_intensity)

        # display results
        fig, axes = plt.subplots(1, 3, figsize=(24,8), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(cell_mask, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')

        ax[1].imshow(image_max, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Maximum filter')

        ax[2].imshow(region_intensity, cmap=plt.cm.gray)
        ax[2].autoscale(False)
        ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        ax[2].axis('off')
        ax[2].set_title('Peak local max')

        fig.tight_layout()

        plt.show()

        return len(coordinates)

    def get_avg_n_NN_distance(self, channel, label, n):
        cell_distances = self._get_parasite_distances(channel, label)
        if cell_distances.any():
            return np.mean(np.sort(cell_distances)[:n]) 
        else:
            return np.nan
    
    # Zonal paper feature: sum(HGs cell intensity) - average_image_hgs_background * cell_area
    def get_normalized_intensity_sum(self, channel, label):
        return self.get_intensity_sum(channel, label) - (self._get_avg_channel_intensity(channel) * self.get_area('mask', label))

    def _get_npx_radius_intensity(self, channel, label, radius):
        cell_mask = (self.mask == label)
        target_area = (math.pi * radius ** 2) + np.sum(cell_mask) # target area = pixels due to radius + pixels due to cell mask
        adjusted_radius = (target_area / math.pi) ** 0.5 # compute radius that most accurately represents target area
        cell_centre = self.get_centre_coords(channel, label)

        circular_mask = mask_utils.create_circular_mask(h=self.mask.shape[0], w=self.mask.shape[1], centre=(cell_centre[1], cell_centre[0]), radius=adjusted_radius)
        radius_mask = circular_mask & ~cell_mask # create a mask for the radius that excludes the cell

        radius_intensity = self.get_channel(channel)[radius_mask]
        return radius_intensity

    def get_sum_npx_radius_intensity(self, channel, label, radius):
        return np.sum(self._get_npx_radius_intensity(channel, label, radius))

    def get_avg_npx_radius_intensity(self, channel, label, radius):
        return np.mean(self._get_npx_radius_intensity(channel, label, radius))

def collect_features_from_path(tif_path, seg_path, feature_dict):
    mask = imageio.v3.imread(seg_path)
    tif = np.array(imageio.mimread(tif_path, memtest=False)).transpose(1, 2, 0)
    extractor = Extractor(mask, tif, Path(tif_path).stem)
    features = extractor.get_features(feature_dict)
    df = pd.concat([df, features])
    return df

def collect_features_from_paths(tif_paths, seg_paths, feature_dict, csv_path=None, append=False, overwrite=False):
    df = pd.DataFrame()
    for tif_path, seg_path in zip(tif_paths, seg_paths):
        df = collect_features_from_path(tif_path, seg_path, feature_dict)

    if csv_path:
        isfile = Path(csv_path).is_file()
        if (isfile and overwrite) or not isfile:
            df.to_csv(csv_path, index=False)

    print('{} features extracted from {} cells in {} images.'.format(sum([len(x[1]) for x in feature_dict.values()]), len(df.index), len(tif_paths)))
    return df

def collect_features_from_folder(tif_folder, seg_folder, feature_dict, csv_path=None, append=False, overwrite=False):
    tif_paths, seg_paths = data_utils.get_two_sets(tif_folder, seg_folder, extension_dir1='.tif', extension_dir2='.png', common_subset=True, return_paths=True)
    df = collect_features_from_paths(tif_paths, seg_paths, feature_dict, csv_path, append, overwrite)
    return df


if __name__ == '__main__':
    # mask_features = ['area', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 
    #                  'extent', 'feret_diameter_max', 'orientation', 'perimeter', 'perimeter_crofton', 'solidity', 
    #                  'avg_(1)_NN_distance', 'avg_(3)_NN_distance', 'avg_(5)_NN_distance',
    #                  'parasites_within_(300)px', 'parasites_within_(600)px']
    # default_channel_features = ['avg_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'intensity_sum',
    #                             'avg_(200)px_radius_intensity']

    # feature_dict = {
    #     'mask': ('', mask_features), 
    #     0: ('dapi', default_channel_features), 
    #     1: ('hsp', default_channel_features)
    # }

    # tif_folder = "/mnt/DATA1/anton/data/lowres_dataset_selection/images"
    # seg_folder = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation"
    # csv_file = "/mnt/DATA1/anton/pipeline_files/results/features/lowres_dataset_selection_features.csv"

    # collect_features_from_folder(tif_folder, seg_folder, feature_dict, csv_file, overwrite=True)

    feature_dict = {
        'mask': ('', []), 
        0: ('dapi', ['num_local_maxima']), 
        1: ('hsp', [])
    }

    tif_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/images/NF135/D5/2019003_D5_135_hsp_20x_2_series_6_TileScan_001.tif"
    seg_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation/NF135/D5/2019003_D5_135_hsp_20x_2_series_6_TileScan_001.png"

    df = collect_features_from_path(tif_path, seg_path, feature_dict)
    print(df)