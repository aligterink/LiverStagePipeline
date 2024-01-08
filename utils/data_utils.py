import numpy as np
import imageio
import os
import glob
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import cv2
from skimage.filters import threshold_otsu
from tqdm.auto import tqdm
        
# Get a list of paths from files in specified directory
def get_paths(folder, extension='', recursive=True, substring=None):
    paths = [p for p in glob.glob(folder + '/**/*{}'.format(extension), recursive=recursive) if p.startswith('.') == False]
    paths = [path for path in paths if not os.path.isdir(path)]
    if substring:
        rx = re.compile(substring)
        paths = [path for path in paths if rx.search(Path(path).stem)]
    paths.sort(key=lambda path: Path(path).stem)
    return paths

# # Get subset of paths in paths_to_subset of which the stem also occurs in leading_paths
# def get_paths_subset(leading_paths, paths_to_subset):
#     subset = [path for path in paths_to_subset if Path(path).stem in [Path(path).stem for path in leading_paths]]
#     subset.sort(key=lambda path: Path(path).stem)
#     return subset

# Verify if two lists of paths contain identical file names
def compare_path_lists(pathlist1, pathlist2):
    for a,b in zip(pathlist1, pathlist2):
        if Path(a).stem != Path(b).stem:
            print(a, b)
    assert [Path(p).stem for p in pathlist1] == [Path(p).stem for p in pathlist2], "folders do not contain identical filenames"

def get_image_array(dir, files, channel):
    return np.array([imageio.mimread(os.path.join(dir, f))[channel] for f in files])

# def get_mask_array(dir):
#     paths = get_paths(dir, extension='.png', recursive=True)
#     return np.array([imageio.mimread(paths) for p in paths])

# def get_mask_array_from_paths(paths):
#     return np.array([imageio.mimread(paths) for p in paths])

def get_data_from_paths(paths, extension, channel=None, array=False):
    if extension == '.tif':
        if channel:
            data = [imageio.mimread(p)[channel] for p in paths]
        else: 
            data = [imageio.mimread(p) for p in paths]
    elif extension == '.png':
        data = [np.array(Image.open(p).convert('L')) for p in paths]

    if array:
        data = np.array(data)
    return data

def get_common_subset(paths_dir1, paths_dir2):
    common_stems = list(set([Path(p).stem for p in paths_dir1]).intersection([Path(p).stem for p in paths_dir2]))
    paths_dir1 = sorted([p for p in paths_dir1 if Path(p).stem in common_stems], key=lambda x: common_stems.index(Path(x).stem))
    paths_dir2 = sorted([p for p in paths_dir2 if Path(p).stem in common_stems], key=lambda x: common_stems.index(Path(x).stem))
    return paths_dir1, paths_dir2


def intersection_of_lists(tuples_of_lists):
    dicts = [{Path(path).stem: path for path in path_list} for path_list in tuples_of_lists]
    common_stems = list(set([stem for sublist in dicts for stem in sublist.keys()]))
    intersections = [[d[stem] for d in dicts] for stem in common_stems if all([stem in d.keys() for d in dicts])]
    transposed_intersections = np.array(intersections).T.tolist()
    return transposed_intersections



# From two folders, return files with corresponding stems
def get_two_sets(dir1, dir2, common_subset=True, substring=None, extension_dir1='', extension_dir2='', 
                 array=False, channel_dir1=None, channel_dir2=None, split=None, exclude=None, return_paths=True, max_imgs=None):
    
    paths_dir1 = get_paths(dir1, extension=extension_dir1, recursive=True, substring=substring)
    paths_dir2 = get_paths(dir2, extension=extension_dir2, recursive=True, substring=substring)

    if exclude:
        exclusion_stems = [Path(p).stem for p in exclude]
        paths_dir1 = [p for p in paths_dir1 if Path(p).stem not in exclusion_stems]
        paths_dir2 = [p for p in paths_dir2 if Path(p).stem not in exclusion_stems]

    if common_subset:
        paths_dir1, paths_dir2 = get_common_subset(paths_dir1, paths_dir2)

    compare_path_lists(paths_dir1, paths_dir2)

    if max_imgs:
        paths_dir1 = paths_dir1[0:max_imgs]
        paths_dir2 = paths_dir2[0:max_imgs]

    if return_paths:
        return paths_dir1, paths_dir2
    
    else:
        X1 = get_data_from_paths(paths_dir1, array=array, channel=channel_dir1, extension=extension_dir1)
        X2 = get_data_from_paths(paths_dir2, array=array, channel=channel_dir2, extension=extension_dir2)

        if split:
            X1_train, X1_test, X2_train, X2_test = train_test_split(X1, X2, test_size=split, random_state=42, shuffle=True)
            return X1_train, X1_test, X2_train, X2_test
        
        else:
            return X1, X2

# # Returns a list of images
# def get_image_list(paths):
#     return [imageio.mimread(p) for p in paths]

# Given a folder and a file stem, return paths of all files in folder with said stem
def find_stem_in_other_folder(folder, stem):
    folder_paths = get_paths(folder)
    try:
        corresponding_path = folder_paths[[Path(f).stem for f in folder_paths].index(stem)]
    except:
        corresponding_path = None
    return corresponding_path

# Given some paths in a dir, generate corresponding paths in another dir
def get_corresponding_paths(paths, old_dir, new_dir):
    return [os.path.join(new_dir, os.path.relpath(path, old_dir)) for path in paths]

# scales an image from an old range to a new range
def rescale(img, old_range, new_range=(0, +1)):
    img -= old_range[0]
    img = img / (old_range[1] / (new_range[1] - new_range[0]))
    img += new_range[0]
    return img

# rescale an image to a new range, but also makes the standard deviation 1
def normalize(img, old_mean, old_SD):
    img -= old_mean
    img /= old_SD
    return img

def count_cells(annotation_folder, substring=None, extension='.png'):
    paths = get_paths(annotation_folder, extension=extension, recursive=True, substring=substring)
    imgs, cells = len(paths), 0
    for path in paths:
        img = imageio.v2.imread(path)
        cells += len(np.unique(img)) - 1
    print('{} cells in {} images for {}'.format(cells, imgs, annotation_folder))
    return imgs, cells

def recursively_count_cells(folder, extension='.png'):
    count_cells(folder, extension=extension)
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    for subfolder in subfolders:
        recursively_count_cells(subfolder, extension=extension)

# for a list of images, and for each channel within those images, find the range (min and max) of intensity values
def find_folder_range(image_paths, channels, otsu=False):
    folder_paths = set([os.path.dirname(path) for path in image_paths])
    range_dict = {fp: {} for fp in folder_paths}

    hist_dict = {fp: {str(channel): np.zeros(shape=(20000)) for channel in channels[0]} for fp in folder_paths} if otsu else None

    for i, path in enumerate(tqdm(image_paths, desc='Finding folder ranges', leave=False)):
        folder = os.path.dirname(path)
        # for channel in channels:
        #     range_dict[folder][channel] = (min(range_dict[folder][channel][0], np.min(imageio.mimread(path, memtest=False)[channel])), 
        #                                    max(range_dict[folder][channel][1], np.max(imageio.mimread(path, memtest=False)[channel])))
        image = imageio.mimread(path, memtest=False)
        for channel in channels[i]:
            if str(channel) not in range_dict[folder].keys():
                range_dict[folder][str(channel)] = [[], []]
            range_dict[folder][str(channel)][0].append(np.min(image[channel]))
            range_dict[folder][str(channel)][1].append(np.max(image[channel]))

            if otsu:
                hist, _ = np.histogram(image, bins=20000, range=(0, 20000))
                hist_dict[folder][str(channel)] += hist

    for fp in range_dict:
        for channel in range_dict[fp]:
            range_dict[fp][str(channel)][0] = min(range_dict[fp][str(channel)][0])
            range_dict[fp][str(channel)][1] = max(range_dict[fp][str(channel)][1])

    if otsu:
        for fp in hist_dict:
            for channel in hist_dict[fp]:
                hist_dict[fp][str(channel)] = threshold_otsu(hist=hist_dict[fp][str(channel)], nbins=20000)
        return range_dict, [hist_dict[os.path.dirname(path)][str(channels[0][0])] for path in image_paths]
    else:
        return range_dict

def find_mean_and_SD(image_paths, channels):
    num_channels = len(channels[0])
    means = [[] for c in range(num_channels)]
    SDs = [[] for c in range(num_channels)]

    for i, p in enumerate(tqdm(image_paths, leave=False, desc='Computing image means')):
        img = imageio.mimread(p)
        for j, c in enumerate(channels[i]):
            means[j].append(np.mean(img[c]))
            # print(np.min(img[c]), np.max(img[c]), np.mean(img[c]), np.sqrt(np.mean((img[c] - np.mean(img[c])) ** 2)))

    channel_means = [np.mean(x) for x in means]

    for i, p in enumerate(tqdm(image_paths, leave=False, desc='Computing image SDs')):
        img = imageio.mimread(p)
        for j, c in enumerate(channels[i]):
            SDs[j].append(np.sqrt(np.mean((img[c] - channel_means[j]) ** 2)))

    channel_SDs = [np.mean(x) for x in SDs]

    return channel_means, channel_SDs


def resize(img, shape):
    return cv2.resize(img, dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)

# Rescale appropriate for masks since it uses INTER_NEAREST as interpolation method
def resize_mask(img, shape):
    return cv2.resize(img, dsize=shape[::-1], interpolation=cv2.INTER_NEAREST)
 

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    else:
        return obj

# Function that prepares a list of samples to be run through an AI model
def collate_fn(batch):
    batch_dict = {}
    batch_dict.update({'images': [sample["image"] for sample in batch]})
    batch_dict.update({'masks_2d': [sample["mask_2d"] for sample in batch]}) if 'mask_2d' in batch[0].keys() else None
    batch_dict.update({'labels': [sample["labels"] for sample in batch]}) if 'labels' in batch[0].keys() else None
    batch_dict.update({'boxes': [sample['boxes'] for sample in batch]}) if 'boxes' in batch[0].keys() else None
    batch_dict.update({'masks_3d': [sample["mask_3d"] for sample in batch]}) if 'mask_3d' in batch[0].keys() else None
    batch_dict.update({'file_paths': [sample['file_path'] for sample in batch]})
    batch_dict.update({'original_sizes': [sample["original_size"] for sample in batch]})
    batch_dict.update({'groups': [sample['group'] for sample in batch]}) if 'group' in batch[0].keys() else None
    return batch_dict

def parse_image(path, channels=None, numpy_dtype=None, torch_dtype=None):
    image = imageio.mimread(path)
    channels = [channels] if isinstance(channels, int) else channels
    
    if channels:
        image = [image[channel] for channel in channels]

    if numpy_dtype:
        image = np.array(image, dtype=numpy_dtype)
    elif torch_dtype:
        image = torch.Tensor(image, dtype=torch_dtype)
    if image.shape[0] == 1:
        try:
            image = image[0, :, :]
        except:
            print(path, image.shape)
    return image

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=0)
    imageio.mimwrite(path, image)

# For a given path (source_path), change the shared prefix of source_path and target_folder into the full path of target_folder.
def change_relative_path(source_path, target_folder):
    common_prefix = os.path.commonprefix([source_path, target_folder])
    relative_source_path = os.path.relpath(source_path, common_prefix)
    new_path = os.path.join(target_folder, relative_source_path)
    return new_path

def get_common_suffix(path1, path2):
    # Split paths into directories and filenames
    dirs1, file1 = os.path.split(path1)
    dirs2, file2 = os.path.split(path2)
    
    # Split directories into components
    components1 = dirs1.split(os.path.sep)
    components2 = dirs2.split(os.path.sep)
    
    # Reverse the lists of components
    reversed_components1 = components1[::-1]
    reversed_components2 = components2[::-1]
    
    # Find the length of the common suffix
    common_length = 0
    while common_length < min(len(reversed_components1), len(reversed_components2)) and reversed_components1[common_length] == reversed_components2[common_length]:
        common_length += 1
    
    # Reverse the common part back to the correct order
    common_suffix_components = reversed_components1[:common_length][::-1]
    
    # Join the components to form the common suffix path
    common_suffix = os.path.sep.join(common_suffix_components)
    
    return common_suffix

def find_common_substring(strings):
    if not strings:
        return ""
    
    min_len = min(len(s) for s in strings)
    
    common_substring = ""
    for i in range(min_len):
        if all(s[i] == strings[0][i] for s in strings):
            common_substring += strings[0][i]
        else:
            break
    return common_substring


if __name__ == "__main__":
    recursively_count_cells(folder='/mnt/DATA1/anton/data/hepatocyte_annotated_dataset/annotation/highres/', extension='')
