import sys
import os
sys.path.append(os.path.sep + os.path.join(*(__file__.split(os.path.sep)[:next((i for i in range(len(__file__.split(os.path.sep)) -1, -1, -1) if 'LiverStagePipeline' in __file__.split(os.path.sep)[i]), None)+1])))


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
from segmentation.AI.datasets import MicroscopyDataset

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math
from skimage import measure
from skimage.measure import _regionprops
from scipy import ndimage
from scipy.spatial import distance
import re
import math
from PIL import Image
import math
from multiprocessing import Pool

# np.seterr('raise')

def process_sample(args):
    sample, feature_dict, metadata_func = args
    results = collect_features_from_sample(sample, feature_dict, metadata_func)
    return results

Image.MAX_IMAGE_PIXELS = None # disables warning
available_regionprops = _regionprops.PROPS.values()

# Class representing a single image. Methods retrieve cell features.
# We use a class here so we can recycle computed channel- or image-wide features.
class Extractor:
    def __init__(self, image, parasite_mask, name, hepatocyte_mask=None, merozoite_mask=None):
        self.image = np.array(image)
        self.name = name
        self.num_channels = image.shape[-1] if np.size(image) > 1 else None
        
        self.parasite_mask = np.array(parasite_mask, dtype=np.uint32)
        self.labels = np.unique(parasite_mask)[1:].tolist()

        self.merozoite_mask = merozoite_mask
        
        ## Intermediate variables that are stored because they can be used to compute multiple features
        # Parasite
        self.max_intensities = [None] * self.num_channels if np.size(image) > 1 else None
        self.avg_intensities = [None] * self.num_channels if np.size(image) > 1 else None
        self.parasite_centre_distance_matrix = None
        self.parasite_hepatocyte_centre_distance_matrix = None

        # Hepatocyte
        # if hepatocyte_mask:
        self.hepatocyte_mask = np.array(hepatocyte_mask, dtype=np.uint32)
        self.hepatocyte_labels = np.unique(hepatocyte_mask)[1:].tolist()
        self.hepatocyte_coords = None

        self.hepatocyte_eccentricity = [None] * len(self.hepatocyte_labels) 
        self.hepatocyte_area = [None] * len(self.hepatocyte_labels) 
        self.hepatocyte_min_intensity = {c: [None] * len(self.hepatocyte_labels) for c in range(self.num_channels)}
        self.hepatocyte_max_intensity = {c: [None] * len(self.hepatocyte_labels) for c in range(self.num_channels)}
        self.hepatocyte_avg_intensity = {c: [None] * len(self.hepatocyte_labels) for c in range(self.num_channels)}

        # template = {'parasite': [None]*len(self.labels), 'hepatocyte': [None]*len(self.hepatocyte_labels)}

        # self.cell_eccentricities = template
        # self.cell_areas = template
        # self.cell_min_intensities = template
        # self.cell_max_intensities = template
        # self.cell_avg_intensities = template

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
            if np.size(self.hepatocyte_mask) < 1:
                sys.exit('No hepatocyte mask present, but hepatocyte features were requested')
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
        if len(self.hepatocyte_labels) < 1: # in cases where no hepatocytes were detected, we skip the image
            return None
        
        props = pd.DataFrame(columns=['file', 'label'], data=[[self.name, label] for label in self.labels])
        x = sorted([c for c in feature_dict.keys() if isinstance(c, int)])

        for channel in feature_dict.keys():
            channel_name = feature_dict[channel][0]
            channel_prop_names = feature_dict[channel][1]
            channel = x.index(channel) if isinstance(channel, int) else channel
            channel_img = self.get_channel(channel)


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
    
    def _get_parasite_hepatocyte_centre_distance_matrix(self):
        if self.parasite_hepatocyte_centre_distance_matrix is None:
            parasite_coords = [self.get_centre_coords(None, label, 'parasite') for label in self.labels]
            hepatocyte_coords = [self.get_centre_coords(None, label, 'hepatocyte') for label in self.hepatocyte_labels]
            if len(hepatocyte_coords) > 0:
                self.parasite_hepatocyte_centre_distance_matrix = distance.cdist(parasite_coords, hepatocyte_coords, 'euclidean') 
            else:
                self.parasite_hepatocyte_centre_distance_matrix = np.nan
        return self.parasite_hepatocyte_centre_distance_matrix

    ###### Non-mask features
    def _get_avg_channel_intensity(self, channel):
        if not self.avg_intensities[channel]:
            self.avg_intensities[channel] = np.average(self.get_channel(channel))
        return self.avg_intensities[channel]
    
    ########################## Parasite-specific features

    ###### Mask features
    def get_centre_coords(self, channel, label, mask_name='parasite'):
        try:
            center = ndimage.center_of_mass(self.get_mask(mask_name) == label)
        except:
            x = self.get_mask(mask_name) == label
            print(np.sum(x), label, len(self.labels), self.labels)
        return [round(c) for c in center]
    
    def get_avg_N_neighbours_distance(self, channel, label, n):
        parasite_distances = self._get_parasite_distances(self.parasite_mask, channel, label)
        return np.mean(np.sort(parasite_distances)[:n])
    
    def get_parasites_within_Npx(self, channel, label, n):
        parasite_distances = self._get_parasite_distances(channel, label)
        return np.sum(parasite_distances <= n)

    # This method is overwritten by the regionprops area feature
    def get_area(self, channel, label, mask_name='parasite'):
        if mask_name == 'parasite':
            return np.sum(self.get_indices(self.get_mask(mask_name), label))
        elif mask_name == 'hepatocyte':
            label_index = self.hepatocyte_labels.index(label)
            if not self.hepatocyte_area[label_index]:
                 self.hepatocyte_area[label_index] = np.sum(self.get_indices(self.get_mask(mask_name), label))
            return self.hepatocyte_area[label_index]


    # For a given parasite label, get a (1,N) numpy array containing the distances to other parasites
    def _get_parasite_distances(self, channel, label):
        distance_matrix = self._get_parasite_centre_distance_matrix(channel, label)
        label_index = self.labels.index(label)
        distances_to_cell = distance_matrix[label_index]
        distances_to_cell = np.delete(distances_to_cell, label_index) # remove distance of parasite to itself
        return distances_to_cell
    
    def _get_eccentricity(self, channel, label, mask_name='parasite'):
        if mask_name == 'parasite':
            return measure.regionprops((self.get_mask(mask_name) == label).astype(np.uint8))[0].eccentricity
        elif mask_name == 'hepatocyte':
            label_index = self.hepatocyte_labels.index(label)
            if not self.hepatocyte_eccentricity[label_index]:
                 self.hepatocyte_eccentricity[label_index] = measure.regionprops((self.get_mask(mask_name) == label).astype(np.uint8))[0].eccentricity
            return self.hepatocyte_eccentricity[label_index]

    ###### Non-mask features
    def get_avg_intensity(self, channel, label, mask_name='parasite'):
        if mask_name == 'parasite':
            return np.mean(self.get_channel(channel)[self.get_indices(self.get_mask(mask_name), label)])
        elif mask_name == 'hepatocyte':
            label_index = self.hepatocyte_labels.index(label)
            if not self.hepatocyte_avg_intensity[channel][label_index]:
                self.hepatocyte_avg_intensity[channel][label_index] = np.mean(self.get_channel(channel)[self.get_indices(self.get_mask(mask_name), label)])
            return self.hepatocyte_avg_intensity[channel][label_index]
    
    def get_std_intensity(self, channel, labels, mask_name='parasite'):
        indices = self.get_indices(self.get_mask(mask_name), labels)
        return np.std(self.get_channel(channel)[indices])
    
    def get_max_intensity(self, channel, label, mask_name='parasite'):
        if mask_name == 'parasite':
            return np.max(self.get_channel(channel)[self.get_indices(self.get_mask(mask_name), label)])
        elif mask_name == 'hepatocyte':
            label_index = self.hepatocyte_labels.index(label)
            if not self.hepatocyte_max_intensity[channel][label_index]:
                self.hepatocyte_max_intensity[channel][label_index] = np.max(self.get_channel(channel)[self.get_indices(self.get_mask(mask_name), label)])
            return self.hepatocyte_max_intensity[channel][label_index]
    
    def get_min_intensity(self, channel, label, mask_name='parasite'):
        if mask_name == 'parasite':
            return np.min(self.get_channel(channel)[self.get_indices(self.get_mask(mask_name), label)])
        elif mask_name == 'hepatocyte':
            label_index = self.hepatocyte_labels.index(label)
            if not self.hepatocyte_min_intensity[channel][label_index]:
                self.hepatocyte_min_intensity[channel][label_index] = np.min(self.get_channel(channel)[self.get_indices(self.get_mask(mask_name), label)])
            return self.hepatocyte_min_intensity[channel][label_index]
    
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
    # def _get_parasite_hepatocyte_distances(self, label):
    #     hepatocyte_coords = self._get_hepatocyte_coords()
    #     parasite_coord = self.get_centre_coords(None, label, 'parasite')
    #     return [math.dist(parasite_coord, hepatocyte_coord) for hepatocyte_coord in hepatocyte_coords]
    
    def _get_hepatocyte_labels_in_Npx_radius(self, label, radius):
        parasite_hepatocyte_dist_matrix = self._get_parasite_hepatocyte_centre_distance_matrix()
        if np.isnan(parasite_hepatocyte_dist_matrix).any():
            return []
        else:
            distances = parasite_hepatocyte_dist_matrix[self.labels.index(label), :] 
            return [self.hepatocyte_labels[i] for i in range(len(self.hepatocyte_labels)) if distances[i] <= radius]
        # hepatocyte_labels = [self.hepatocyte_labels[i] for i in range(len(self.hepatocyte_labels)) if distances[i] <= radius]
        # hepatocyte_indices = [self.hepatocyte_labels.index(l) for l in hepatocyte_labels]
        # return hepatocyte_indices
    
    def _get_hepatocyte_N_neighbours_labels(self, label, n):
        parasite_hepatocyte_dist_matrix = self._get_parasite_hepatocyte_centre_distance_matrix()
        if np.isnan(parasite_hepatocyte_dist_matrix).any():
            return []
        else:
            distances = parasite_hepatocyte_dist_matrix[self.labels.index(label), :]
            indexed_dists = list(enumerate(distances))
            sorted_indices = sorted(indexed_dists, key=lambda x: x[1]) # sort the list based on the integer values
            lowest_indices = [index for index, _ in sorted_indices[:n]] # get the first N indices from the sorted list
            return [self.hepatocyte_labels[i] for i in lowest_indices]
    
    def get_avg_N_nearest_hepatocytes_distance(self, channel, label, n):
        parasite_hepatocyte_dist_matrix = self._get_parasite_hepatocyte_centre_distance_matrix()
        if np.isnan(parasite_hepatocyte_dist_matrix).any():
            return []
        else:
            distances = parasite_hepatocyte_dist_matrix[self.labels.index(label), :]
            return np.mean(np.sort(distances)[:n]) 

    def get_num_hepatocytes_in_Npx_radius(self, channel, label, radius):
        return len(self._get_hepatocyte_labels_in_Npx_radius(label, radius))
    
    ### Intensity
    def get_avg_hepatocyte_intensity_Npx_radius(self, channel, label, radius):
        hepatocyte_labels = self._get_hepatocyte_labels_in_Npx_radius(label, radius)
        avg_intensities = [self.get_avg_intensity(channel, l, 'hepatocyte') for l in hepatocyte_labels]
        return np.mean(avg_intensities) if avg_intensities else np.nan
    
    def get_min_hepatocyte_intensity_Npx_radius(self, channel, label, radius):
        hepatocyte_labels = self._get_hepatocyte_labels_in_Npx_radius(label, radius)
        min_intensities = [self.get_min_intensity(channel, l, 'hepatocyte') for l in hepatocyte_labels]
        return np.mean(min_intensities) if min_intensities else np.nan

    def get_max_hepatocyte_intensity_Npx_radius(self, channel, label, radius):
        hepatocyte_labels = self._get_hepatocyte_labels_in_Npx_radius(label, radius)
        # for l in hepatocyte_labels:
        #     print(l, channel)
        max_intensities = [self.get_max_intensity(channel, l, 'hepatocyte') for l in hepatocyte_labels]
        # print(hepatocyte_labels)
        # print(max_intensities)
        return np.mean(max_intensities) if max_intensities else np.nan
    
    def get_avg_hepatocyte_area_Npx_radius(self, channel, label, radius):
        hepatocyte_labels = self._get_hepatocyte_labels_in_Npx_radius(label, radius)
        hepatocyte_areas = [self.get_area(None, l, 'hepatocyte') for l in hepatocyte_labels]
        return np.mean(hepatocyte_areas) if hepatocyte_areas else np.nan
    
    def get_avg_hepatocyte_eccentricity_Npx_radius(self, channel, label, n):
        hepatocyte_labels = self._get_hepatocyte_labels_in_Npx_radius(label, n)
        eccentricities = [self._get_eccentricity(None, h, mask_name='hepatocyte') for h in hepatocyte_labels]
        return np.mean(eccentricities) if eccentricities else np.nan
    
    # (sum intensity parasite) / ((sum intensity nearest host) + sum intensity parasite)
    # Potentially interesting, as computing this for DAPI might give an indication as to how much resources 
    # from the host are being taken away by the parasite.
    def get_cell_intensity_ratio(self, channel, label):
        nearest_hepatocyte_label = self._get_hepatocyte_N_neighbours_labels(label, 1)
        sum_nearest_hepatocyte_intensity = self.get_intensity_sum(channel, nearest_hepatocyte_label, mask_name='hepatocyte')
        sum_parasite_intensity = self.get_intensity_sum(channel, label, mask_name='parasite')
        return sum_parasite_intensity / (sum_parasite_intensity + sum_nearest_hepatocyte_intensity)


def collect_features_from_sample(sample, feature_dict, metadata_func=None):
    extractor = Extractor(image=np.array(sample['image']).transpose(1, 2, 0) if 'image' in sample else None, name=Path(sample['file_path']).stem, parasite_mask=sample['mask_2d'], hepatocyte_mask=sample['hepatocyte_mask'] if 'hepatocyte_mask' in sample.keys() else None)
    features = extractor.get_features(feature_dict)

    if metadata_func:
        metadata = metadata_func(sample['file_path'])
        metadata = {k: [v]*len(features) for k,v in metadata.items()}

        metadata = pd.DataFrame(metadata)
        features = pd.concat([metadata, features], ignore_index=False, axis=1)
    return features

def collect_features_from_dataset(dataset, feature_dict, csv_path=None, metadata_func=None, workers=1, batch_size=96):
    
    # no multithreading, for debugging purposes
    # for sample in tqdm(dataset, desc='Extracting features'):
    #     file_props = collect_features_from_sample(sample, feature_dict, metadata_func)
    #     df = pd.concat([df, file_props])

    with Pool(workers) as pool:
        for batch_start in tqdm(range(0, len(dataset), batch_size), total=math.ceil(len(dataset)/batch_size), desc='Total progression', leave=False):
            batch_end = min(batch_start + batch_size, len(dataset))
            df = pd.DataFrame() 

            batch_range = range(batch_start, batch_end)
            results = pool.imap(process_sample, ((dataset[i], feature_dict, metadata_func) for i in batch_range))
            # results = pool.imap(process_sample, ((sample, feature_dict, metadata_func) for sample in dataset))

            for result in tqdm(results, total=len(batch_range), desc='Batch', leave=False):
                
                # Concatenate the results into the DataFrame
                df = pd.concat([df, result]) if result is not None else df

            # Reset the index of the DataFrame
            df.reset_index(drop=True, inplace=True)

            if csv_path:
                if Path(csv_path).is_file():
                    old_df = pd.read_csv(csv_path)
                    df = pd.concat([old_df, df])
                df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    print('{} features extracted from {} cells in {} images.'.format(sum([len(x[1]) for x in feature_dict.values()]), len(df.index), len(df['file'].unique())))
    return df

def collect_features_from_folder(tif_folder, parasite_mask_folder, feature_dict, hepatocyte_mask_folder=None, csv_path=None, metadata_func=None, workers=1):
    if csv_path and os.path.exists(csv_path):
        completed_paths = pd.read_csv(csv_path)['file'].unique().tolist()
        if len(completed_paths) > 0:
            print('Features found of {} files. Skipping those.'.format(len(completed_paths)))
    else:
        completed_paths = []

    tif_paths, parasite_mask_paths = data_utils.get_two_sets(tif_folder, parasite_mask_folder, extension_dir1='.tif', extension_dir2='', common_subset=True, return_paths=True, exclude=completed_paths)
    
    if hepatocyte_mask_folder:
        hepatocyte_mask_paths = data_utils.get_paths(hepatocyte_mask_folder)
        tif_paths, hepatocyte_mask_paths = data_utils.get_common_subset(tif_paths, hepatocyte_mask_paths)
        data_utils.compare_path_lists(tif_paths, parasite_mask_paths)
    else:
        hepatocyte_mask_paths = None

    channels = [sorted([x for x in feature_dict.keys() if isinstance(x, int)])]*len(tif_paths)
    dataset = MicroscopyDataset(image_paths=tif_paths, channels=channels, mask_paths=parasite_mask_paths, hepatocyte_mask_paths=hepatocyte_mask_paths, folder_normalize=True)
    df = collect_features_from_dataset(dataset, feature_dict, csv_path, metadata_func, workers=workers)
    return df

def default_feature_dict(hsp_channel, dapi_channel):
    parasite_mask_features = [
        # Area
        # Convex area
        # Filled area
        # Axis major length
        # Axis minor length
        # Eccentricity
        # Equivalent diameter area
        # Extent
        # Maximum Feret diameter
        # Perimeter
        # Crofton's perimeter
        # Solidity
        'area', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'eccentricity', 
        'equivalent_diameter_area', 'extent', 'feret_diameter_max', 'perimeter', 'perimeter_crofton', 'solidity', 

        # Average N nearest parasites distance
        # Parasites within N pixels
        'avg_(1)_neighbours_distance', 'avg_(3)_neighbours_distance', 'avg_(5)_neighbours_distance', 'avg_(7)_neighbours_distance',
        'parasites_within_(100)px', 'parasites_within_(300)px', 'parasites_within_(600)px', 
        'parasites_within_(900)px', 'parasites_within_(1200)px', 'parasites_within_(2000)px'   
        ]
    
    hepatocyte_mask_features = [
        # Average N nearest hepatocytes distance
        'avg_(1)_nearest_hepatocytes_distance', 'avg_(3)_nearest_hepatocytes_distance', 'avg_(5)_nearest_hepatocytes_distance', 
        'avg_(7)_nearest_hepatocytes_distance', 'avg_(10)_nearest_hepatocytes_distance', 'avg_(15)_nearest_hepatocytes_distance', 
        'avg_(20)_nearest_hepatocytes_distance', 'avg_(50)_nearest_hepatocytes_distance', 'avg_(100)_nearest_hepatocytes_distance', 

        # Hepatocytes within N pixel radius
        'num_hepatocytes_in_(100)px_radius', 'num_hepatocytes_in_(300)px_radius', 'num_hepatocytes_in_(600)px_radius', 'num_hepatocytes_in_(900)px_radius', 

        # Hepatocytes within N pixel radius average area
        'avg_hepatocyte_area_(100)px_radius', 'avg_hepatocyte_area_(300)px_radius', 'avg_hepatocyte_area_(600)px_radius', 'avg_hepatocyte_area_(900)px_radius',

        # Hepatocytes within N pixel radius average eccentricity
        'avg_hepatocyte_eccentricity_(100)px_radius', 'avg_hepatocyte_eccentricity_(300)px_radius', 'avg_hepatocyte_eccentricity_(600)px_radius', 
        'avg_hepatocyte_eccentricity_(900)px_radius', 
    ]
    

    dapi_channel_features = [
        # Average intensity, intensity standard deviation, minimum intensity, maximum intensity, intensity sum
        'avg_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'intensity_sum',

        # N pixel radius average intensity
        'avg_(50)px_radius_intensity', 'avg_(100)px_radius_intensity', 'avg_(300)px_radius_intensity',
                                
        # Hepatocytes within N pixel radius average intensity
        'avg_hepatocyte_intensity_(100)px_radius', 'avg_hepatocyte_intensity_(300)px_radius', 'avg_hepatocyte_intensity_(600)px_radius', 'avg_hepatocyte_intensity_(900)px_radius', 

        # # Hepatocytes within N pixel radius average minimum intensity
        'min_hepatocyte_intensity_(100)px_radius', 'min_hepatocyte_intensity_(300)px_radius', 'min_hepatocyte_intensity_(600)px_radius', 'min_hepatocyte_intensity_(900)px_radius', 

        # # Hepatocytes within N pixel radius average maximum intensity
        'max_hepatocyte_intensity_(100)px_radius', 'max_hepatocyte_intensity_(300)px_radius', 'max_hepatocyte_intensity_(600)px_radius', 'max_hepatocyte_intensity_(900)px_radius']
    
    hsp_channel_features = [
        # Average intensity, intensity standard deviation, minimum intensity, maximum intensity, intensity sum
        'avg_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'intensity_sum',

        # N pixel radius average intensity
        'avg_(50)px_radius_intensity', 'avg_(100)px_radius_intensity', 'avg_(300)px_radius_intensity'
    ]

    # Cell intensity ratio
    # feature_dict = {'mask': ('', parasite_mask_features + hepatocyte_mask_features)}
    feature_dict = {
        'mask': ('', parasite_mask_features + hepatocyte_mask_features),
        hsp_channel: ('hsp', hsp_channel_features),
        dapi_channel: ('dapi', dapi_channel_features + ['cell_intensity_ratio'])
    }

    return feature_dict


if __name__ == '__main__':
    
    feature_dict = default_feature_dict(hsp_channel=1, dapi_channel=0)

    tif_folder = "/mnt/DATA1/anton/data/parasite_annotated_dataset/images/lowres/NF54/D7"
    parasite_folder = "/mnt/DATA1/anton/data/parasite_annotated_dataset/annotation/lowres/NF54/D7"
    hepatocyte_folder = '/mnt/DATA1/anton/data/basically_trash'
    csv_file = '/mnt/DATA1/anton/pipeline_files/feature_analysis/features/hihellohi.csv'

    df = collect_features_from_folder(tif_folder, parasite_folder, feature_dict, hepatocyte_folder, csv_file, overwrite=True, metadata_func=GS_metadata, workers=44)

    # df = df.drop('file', axis=1)
    print(df)
    print(len(df.columns))

