import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[0] + 'LiverStagePipeline')


# lowest_folder = next((root for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))) if 'requirements.txt' in files), None)
# print(lowest_folder, os.path.dirname(os.path.abspath(__file__)))
# for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
#     print(root, dirs, files)

import imageio.v3
from utils import data_utils, mask_utils
from skimage.measure import regionprops
from skimage import data, util, measure
import numpy as np
import pandas as pd  
from pathlib import Path
from skimage.feature import peak_local_max, canny
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
from skimage.segmentation import find_boundaries
import math

Image.MAX_IMAGE_PIXELS = None # disables warning
available_regionprops = _regionprops.PROPS.values()

# Class representing a single image. Methods retrieve cell features.
# We use a class here so we can recycle computed channel- or image-wide features.
class Extractor:
    def __init__(self, image, parasite_mask, name, hepatocyte_mask=None, merozoite_mask=None):
        self.image = image
        self.name = name
        self.num_channels = image.shape[-1]
        
        self.parasite_mask = parasite_mask
        self.labels = np.unique(parasite_mask)[1:].tolist()

        self.hepatocyte_mask = hepatocyte_mask
        self.hepatocyte_labels = np.unique(hepatocyte_mask)[1:].tolist()

        self.merozoite_mask = merozoite_mask
        
        # Intermediate variables that are stored because they can be used to compute multiple features
        self.max_intensities = [None] * self.num_channels
        self.avg_intensities = [None] * self.num_channels
        self.parasite_centre_distance_matrix = None
        self.hepatocyte_coords = None

    def get_channel(self, channel):
        if channel == None:
            return None
        elif channel == 'mask':
            return self.parasite_mask
        
        else: 
            return self.image[:, :, channel]

    def get_mask(self, mask):
        if mask == 'merozoite':
            return self.merozoite_mask
        elif mask == 'hepatocyte':
            return self.hepatocyte_mask
        elif mask == 'parasite':
            return self.parasite_mask
        else:
            print('Unknown mask')
        
    def get_indices(self, mask, labels):
        if isinstance(labels, int):
            return (mask == labels)
        else:
            return np.isin(mask, labels)


    # Method for calling our own features
    def call_feature(self, feature, channel):
        N_match = None

        # Check if there is a '(N)' pattern, if so supply feature method with additional N parameter
        N_pattern = r'\((\d+)\)'
        N_match = re.search(N_pattern, feature)
        if N_match:
            n = N_match.group(1)  # Extract the value of N from the match
            feature = re.sub(N_pattern, 'N', feature)  # Replace (N) with N

        feature_method = getattr(self, 'get_' + feature) # let's not name features after other attributes or imports here

        if N_match:
            x = [feature_method(channel, label, int(n)) for label in self.labels]
            return x
        else:
            return [feature_method(channel, label) for label in self.labels]
    
    def get_features(self, feature_dict):

        props = pd.DataFrame(columns=['file', 'label'], data=[[self.name, label] for label in self.labels])
        
        for channel in feature_dict.keys():
            channel_img = self.get_channel(channel)
            channel_name = feature_dict[channel][0]
            channel_prop_names = feature_dict[channel][1]

            skimage_channel_prop_names = list(set(channel_prop_names).intersection(available_regionprops))
            if len(skimage_channel_prop_names) > 0:
                skimage_channel_props = pd.DataFrame(measure.regionprops_table(self.parasite_mask, channel_img, properties=['label'] + skimage_channel_prop_names))
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
    def _get_parasite_centre_distance_matrix(self, channel, label):
        if self.parasite_centre_distance_matrix is None:
            coords = [self.get_centre_coords(channel, label) for label in self.labels]
            self.parasite_centre_distance_matrix = distance.cdist(coords, coords, 'euclidean')
        return self.parasite_centre_distance_matrix
    
    ###### Non-mask features
    def _get_avg_channel_intensity(self, channel):
        if not self.avg_intensities[channel]:
            self.avg_intensities[channel] = np.average(self.get_channel(channel))
        return self.avg_intensities[channel]
    
    ########################## Parasite-specific features

    ###### Mask features
    def get_centre_coords(self, channel, label, mask_name='parasite'):
        return ndimage.center_of_mass(self.get_mask(mask_name) == label)
    
    def get_avg_N_neighbours_distance(self, channel, label, n):
        parasite_distances = self._get_parasite_distances(self.parasite_mask, channel, label)
        return np.mean(np.sort(parasite_distances)[:n])
    
    def get_parasites_within_Npx(self, channel, label, n):
        parasite_distances = self._get_parasite_distances(channel, label)
        return np.sum(parasite_distances <= n)

    # This method is overwritten by the regionprops area feature
    def get_area(self, channel, labels, mask_name='parasite'):
        indices = self.get_indices(self.get_mask(mask_name), labels)
        return np.sum(indices)

    # For a given parasite label, get a (1,N) numpy array containing the distances to other parasites
    def _get_parasite_distances(self, channel, label):
        distance_matrix = self._get_parasite_centre_distance_matrix(channel, label)
        label_index = self.labels.index(label)
        distances_to_cell = distance_matrix[label_index]
        distances_to_cell = np.delete(distances_to_cell, label_index) # remove distance of parasite to itself
        return distances_to_cell
    

    ###### Non-mask features
    def get_avg_intensity(self, channel, labels, mask_name='parasite'):
        indices = self.get_indices(self.get_mask(mask_name), labels)
        return np.mean(self.get_channel(channel)[indices])
    
    def get_std_intensity(self, channel, labels, mask_name='parasite'):
        indices = self.get_indices(self.get_mask(mask_name), labels)
        return np.std(self.get_channel(channel)[indices])
    
    def get_max_intensity(self, channel, labels, mask_name='parasite'):
        indices = self.get_indices(self.get_mask(mask_name), labels)
        return np.max(self.get_channel(channel)[indices])
    
    def get_min_intensity(self, channel, labels, mask_name='parasite'):
        indices = self.get_indices(self.get_mask(mask_name), labels)
        return np.min(self.get_channel(channel)[indices])
    
    def get_intensity_sum(self, channel, labels, mask_name='parasite'):
        indices = self.get_indices(self.get_mask(mask_name), labels)
        return np.sum(self.get_channel(channel)[indices])

    def get_avg_N_neighbours_distance(self, channel, label, n):
        cell_distances = self._get_parasite_distances(channel, label)
        if cell_distances.any():
            return np.mean(np.sort(cell_distances)[:n]) 
        else:
            return np.nan
    
    # Zonal paper feature: sum(HGs cell intensity) - average_image_hgs_background * cell_area
    def get_normalized_intensity_sum(self, channel, label):
        return self.get_intensity_sum(channel, label) - (self._get_avg_channel_intensity(channel) * self.get_area('mask', label))

    def _get_Npx_radius_intensity(self, channel, label, radius):
        cell_mask = (self.parasite_mask == label)
        target_area = (math.pi * radius ** 2) + np.sum(cell_mask) # target area = pixels due to radius + pixels due to cell mask
        adjusted_radius = (target_area / math.pi) ** 0.5 # compute radius that most accurately represents target area
        cell_centre = self.get_centre_coords(channel, label)

        circular_mask = mask_utils.create_circular_mask(h=self.parasite_mask.shape[0], w=self.parasite_mask.shape[1], centre=(cell_centre[1], cell_centre[0]), radius=adjusted_radius)
        radius_mask = circular_mask & ~cell_mask # create a mask for the radius that excludes the cell

        radius_intensity = self.get_channel(channel)[radius_mask]
        return radius_intensity

    def get_sum_Npx_radius_intensity(self, channel, label, radius):
        return np.sum(self._get_Npx_radius_intensity(channel, label, radius))

    def get_avg_Npx_radius_intensity(self, channel, label, radius):
        return np.mean(self._get_Npx_radius_intensity(channel, label, radius))
    
    ########## Hepatocyte features
    def _get_hepatocyte_coords(self):
        if not self.hepatocyte_coords:
            self.hepatocyte_coords = [self.get_centre_coords(None, label, 'hepatocyte') for label in self.hepatocyte_labels]
        return self.hepatocyte_coords

    # For a parasite label, return a list containing the distances from that parasite to all hepatocytes
    def _get_parasite_hepatocyte_distances(self, label):
        hepatocyte_coords = self._get_hepatocyte_coords()
        parasite_coord = self.get_centre_coords(None, label, 'parasite')
        return [math.dist(parasite_coord, hepatocyte_coord) for hepatocyte_coord in hepatocyte_coords]
    
    def _get_hepatocyte_labels_in_Npx_radius(self, label, radius):
        distances = self._get_parasite_hepatocyte_distances(label)
        return [self.hepatocyte_labels[i] for i in range(len(self.hepatocyte_labels)) if distances[i] <= radius]
    
    def _get_hepatocyte_N_neighbours_labels(self, label, n):
        parasite_hepatocyte_distances = self._get_parasite_hepatocyte_distances(label)

        indexed_dists = list(enumerate(parasite_hepatocyte_distances))
        sorted_indices = sorted(indexed_dists, key=lambda x: x[1]) # sort the list based on the integer values
        lowest_indices = [index for index, _ in sorted_indices[:n]] # get the first N indices from the sorted list
        # print(label, [self.hepatocyte_labels[i] for i in lowest_indices])
        return [self.hepatocyte_labels[i] for i in lowest_indices]
    
    def get_avg_N_nearest_hepatocytes_distance(self, channel, label, n):
        parasite_to_hepatocyte_distances = self._get_parasite_hepatocyte_distances(label)
        return np.mean(np.sort(parasite_to_hepatocyte_distances)[:n]) 

    def get_num_hepatocytes_in_Npx_radius(self, channel, label, radius):
        return len(self._get_hepatocyte_labels_in_Npx_radius(label, radius))
    
    def get_avg_hepatocyte_intensity_N_px_radius(self, channel, label, radius):
        hepatocyte_labels = self._get_hepatocyte_labels_in_Npx_radius(label, radius)
        return self.get_avg_intensity(channel, hepatocyte_labels, 'hepatocyte')

    def get_avg_N_nearest_hepatocytes_intensity(self, channel, label, n):
        hepatocyte_labels = self._get_hepatocyte_N_neighbours_labels(label, n)
        return self.get_avg_intensity(channel, hepatocyte_labels, 'hepatocyte')
    
    def get_avg_hepatocyte_area_N_px_radius(self, channel, label, radius):
        hepatocyte_labels = self._get_hepatocyte_labels_in_Npx_radius(label, radius)
        return self.get_area(None, hepatocyte_labels, 'hepatocyte') / len(hepatocyte_labels)

    def get_avg_N_nearest_hepatocytes_area(self, channel, label, n):
        hepatocyte_labels = self._get_hepatocyte_N_neighbours_labels(label, n)
        return self.get_area(None, hepatocyte_labels, 'hepatocyte') / len(hepatocyte_labels)
    
    # (sum intensity parasite) / ((sum intensity nearest host) + sum intensity parasite)
    # Potentially interesting, as computing this for DAPI might give an indication as to how much resources 
    # from the host are being taken away by the parasite.
    def get_cell_intensity_ratio(self, channel, label):
        nearest_hepatocyte_label = self._get_hepatocyte_N_neighbours_labels(label, 1)
        sum_nearest_hepatocyte_intensity = self.get_intensity_sum(channel, nearest_hepatocyte_label, mask_name='hepatocyte')
        sum_parasite_intensity = self.get_intensity_sum(channel, label, mask_name='parasite')
        return sum_parasite_intensity / (sum_parasite_intensity + sum_nearest_hepatocyte_intensity)




    ########## Merozoite features
    def get_num_merozoites(self, channel, label):
        cell_mask = (self.parasite_mask == label)
        cell_merozoite_mask = self.merozoite_mask.copy()
        cell_merozoite_mask[~cell_mask] = 0
        merozoite_labels = np.unique(cell_merozoite_mask)
        merozoite_labels = np.delete(merozoite_labels, np.where(merozoite_labels == 0)) # remove background label 0 from labels
        return len(merozoite_labels)
    
    def get_merozoite_fullness_ratio(self, channel, label):
        cell_mask = (self.parasite_mask == label)
        cell_merozoite_mask = np.array(self.merozoite_mask.copy(), dtype=bool)
        cell_merozoite_mask[~cell_mask] = 0
        merozoite_fullness_ratio = np.sum(cell_merozoite_mask) / np.sum(cell_mask)
        return merozoite_fullness_ratio
    
    def get_avg_merozoite_border_distance(self, channel, label):
        cell_mask = (self.parasite_mask == label)
        cell_bbox = mask_utils.get_bbox_from_mask(cell_mask, padding=1)
        cell_mask_crop = mask_utils.get_crop(cell_mask, cell_bbox)
        merozoite_mask_crop = np.array(mask_utils.get_crop(self.merozoite_mask, cell_bbox), dtype=bool)

        merozoite_mask_crop_distance = distance_transform_edt(~merozoite_mask_crop)
        cell_mask_crop_outline = find_boundaries(cell_mask_crop, mode='inner')

        avg_distance = np.mean(merozoite_mask_crop_distance[cell_mask_crop_outline])

        # import matplotlib.pyplot as plt
        # imgs = {'cell': cell_mask_crop, 'merozoite': merozoite_mask_crop, 'merozoite_mask_crop_distance': merozoite_mask_crop_distance,
        #         'cell_mask_crop_outline': cell_mask_crop_outline}
        # fig, axes = plt.subplots(ncols=len(imgs.items())+1, figsize=(20,5), sharex=True, sharey=True)
        # ax = axes.ravel()
        # for i,(k,v) in enumerate(imgs.items()):
        #     ax[i].imshow(v, cmap=plt.cm.gray)
        #     ax[i].set_title(k)
        # fig.tight_layout()
        # # plt.show()
        # plt.savefig('/mnt/DATA1/anton/example5.png')
        # input('waiting for input ...')
        return avg_distance

####### Extractor calling methods
def collect_features_from_path(image_path, feature_dict, parasite_mask_path, hepatocyte_mask_path=None, merozoite_mask_path=None, metadata_func=None):
    image = np.array(imageio.mimread(image_path, memtest=False)).transpose(1, 2, 0)
    parasite_mask = imageio.v3.imread(parasite_mask_path)
    hepatocyte_mask = imageio.v3.imread(hepatocyte_mask_path)
    merozoite_mask = imageio.v3.imread(merozoite_mask_path)

    extractor = Extractor(image=image, name=Path(image_path).stem, parasite_mask=parasite_mask, hepatocyte_mask=hepatocyte_mask, merozoite_mask=merozoite_mask)
    features = extractor.get_features(feature_dict)

    if metadata_func:
        metadata = metadata_func(image_path, parasite_mask_path)
        metadata = {k: [v]*len(features) for k,v in metadata.items()}

        metadata = pd.DataFrame(metadata)
        features = pd.concat([metadata, features], ignore_index=False, axis=1)
    return features

def collect_features_from_paths(tif_paths, seg_paths, feature_dict, csv_path=None, append=False, overwrite=False, metadata_func=None):
    df = pd.DataFrame()
    for tif_path, seg_path, i in zip(tif_paths, seg_paths, range(len(tif_paths))):
        print('{} / {}'.format(i, len(tif_paths)))
        file_props = collect_features_from_path(tif_path, seg_path, feature_dict, metadata_func)
        df = pd.concat([df, file_props])

    if csv_path:
        isfile = Path(csv_path).is_file()
        if (isfile and overwrite) or not isfile:
            df.to_csv(csv_path, index=False)

    print('{} features extracted from {} cells in {} images.'.format(sum([len(x[1]) for x in feature_dict.values()]), len(df.index), len(tif_paths)))
    return df

def collect_features_from_folder(tif_folder, seg_folder, feature_dict, csv_path=None, append=False, overwrite=False, metadata_func=None):
    tif_paths, seg_paths = data_utils.get_two_sets(tif_folder, seg_folder, extension_dir1='.tif', extension_dir2='.png', common_subset=True, return_paths=True)
    df = collect_features_from_paths(tif_paths, seg_paths, feature_dict, csv_path, append, overwrite, metadata_func)
    return df

#### Metadata functions
def FoI_metadata(tif_path, seg_path):
    
    foi_substring = Path(tif_path).stem.split('_')[2]
    if foi_substring.startswith('D'):
        foi_substring = Path(tif_path).stem.split('_')[3]

    foi_substring = foi_substring.strip('h')

    x = re.split('to|s|-', foi_substring)
    x = list(filter(None, x))

    # Extract the substring between the first and second '_' in the file name
    match = re.search(r"_(.*?)_", Path(tif_path).stem)
    if match:
        substring = match.group(1)

        # Detect the number using regular expressions
        number_match = re.search(r"(nf|NF)?(175|135|54)", substring, re.IGNORECASE)
        if number_match:
            detected_number = int(number_match.group(2))
        elif substring == "L1":
            detected_number = 54
        else:
            print("Number not found.")
    else:
        print("Substring not found.")

    meta_data = {
        'force_of_infection': '{}s-{}h'.format(x[0], x[1]),
        'force_of_infection_ratio': int(x[0]) / int(x[1]),
        # 'day': int(re.search(r"D(\d+)", tif_path).group(1)),
        # 'strain': int(re.search(r"NF(\d+)", tif_path).group(1))
        'strain': detected_number
    }
    return meta_data


def GS_metadata(tif_path, seg_path):

    file = Path(tif_path).stem
    splits = file.split('_')

    if 'D3' in file:
        day = 3
    elif 'D5' in file:
        day = 5
    elif 'D7' in file:
        day = 7
    else:
        day = 3

    if any(substring in file for substring in ['_54', '_NF54', '_nf54']):
        strain = 54
    elif any(substring in file for substring in ['_135', '_NF135', '_nf135']):
        strain = 135
    elif any(substring in file for substring in ['_175', '_NF175', '_nf175']):
        strain = 175

    return {'day': day, 'strain': strain}





if __name__ == '__main__':
    mask_features = [ # skimage
        'area', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'eccentricity', 
        'equivalent_diameter_area', 'extent', 'feret_diameter_max', 'perimeter', 'perimeter_crofton', 'solidity', 

        # parasite density-based
        'avg_(1)_neighbours_distance', 'avg_(3)_neighbours_distance', 'avg_(5)_neighbours_distance',
        'parasites_within_(300)px', 'parasites_within_(600)px',

        # hepatocyte features
        'num_hepatocytes_in_(300)px_radius', 'avg_hepatocyte_area_(300)_px_radius', 'avg_(5)_nearest_hepatocytes_area',

        # merozoite features
        'avg_merozoite_border_distance', 'num_merozoites', 'merozoite_fullness_ratio']
    
    default_channel_features = ['avg_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'intensity_sum',
                                'avg_(100)px_radius_intensity', 'avg_(200)px_radius_intensity', 'avg_(300)px_radius_intensity']

    feature_dict = {
        'mask': ('', mask_features), 
        0: ('dapi', default_channel_features + ['avg_hepatocyte_intensity_(300)_px_radius', 'cell_intensity_ratio']), 
        1: ('hsp', default_channel_features)
        # -1: ('hgs', default_channel_features)
    }

    # tif_folder = "/mnt/DATA1/anton/data/unformatted/GS validation data/untreated_tifs"
    # seg_folder = "/mnt/DATA1/anton/pipeline_files/segmentation/segmentations/GS_validation_all_untreated_2_copypaste_1806"
    # csv_file = "/mnt/DATA1/anton/pipeline_files/feature_analysis/features/untreated_GS_validation_features.csv"

    # x = collect_features_from_folder(tif_folder, seg_folder, feature_dict, csv_file, overwrite=True, metadata_func=GS_metadata)
    # feature_dict = {
    #     'mask': ('', ['avg_(5)_neighbours_distance']), 
    #     0: ('dapi', []), 
    #     1: ('hsp', [])
    # }

    image_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/images/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif"
    parasite_mask_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.png"
    merozoite_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/merozoite_watershed/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif'
    hepatocyte_mask_path = '/mnt/DATA1/anton/data/lowres_dataset/hepatocyte_watershed/NF135/D5/2019003_D5_135_hsp_20x_2_series_11_TileScan_001.tif'

    df = collect_features_from_path(image_path=image_path, feature_dict=feature_dict, parasite_mask_path=parasite_mask_path, hepatocyte_mask_path=hepatocyte_mask_path, merozoite_mask_path=merozoite_mask_path)
    df = df.drop('file', axis=1)
    print(df)

def get_indices(mask, labels):
    if isinstance(labels, np.ndarray):
        return np.isin(mask, labels)
    else:
        return (mask == labels)
    
# mask = np.array([[1, 2], [3, 4]])
# labels = [1, 3]
# print(get_indices(mask, labels))