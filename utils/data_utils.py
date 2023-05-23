import numpy as np
import imageio
import os
import glob
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

        
# Get a list of paths from files in specified directory
def get_paths(folder, extension='', recursive=True, substring=None):
    paths = [p for p in glob.glob(folder + '/**/*{}'.format(extension), recursive=recursive) if p.startswith('.') == False]
    if substring:
        rx = re.compile(substring)
        paths = [path for path in paths if rx.search(Path(path).stem)]
    paths.sort(key=lambda path: Path(path).stem)
    return paths

# Get subset of paths in paths_to_subset of which the stem also occurs in leading_paths
def get_paths_subset(leading_paths, paths_to_subset):
    subset = [path for path in paths_to_subset if Path(path).stem in [Path(path).stem for path in leading_paths]]
    subset.sort(key=lambda path: Path(path).stem)
    return subset

# Verify if two lists of paths contain identical file names
def compare_path_lists(pathlist1, pathlist2):
    assert [Path(p).stem for p in pathlist1] == [Path(p).stem for p in pathlist2], "folders do not contain identical filenames"

def get_image_array(dir, files, channel):
    return np.array([imageio.mimread(os.path.join(dir, f))[channel] for f in files])

def get_mask_array(dir):
    paths = get_paths(dir, extension='.png', recursive=True)
    return np.array([imageio.mimread(paths) for p in paths])

def get_mask_array_from_paths(paths):
    return np.array([imageio.mimread(paths) for p in paths])

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
    paths_dir1 = [p for p in paths_dir1 if Path(p).stem in common_stems]
    paths_dir2 = [p for p in paths_dir2 if Path(p).stem in common_stems]
    return paths_dir1, paths_dir2

# From two folders, return files with corresponding stems
def get_two_sets(dir1, dir2, common_subset=False, substring=None, extension_dir1='.tif', extension_dir2='.png', 
                 array=False, channel_dir1=None, channel_dir2=None, split=None, exclude=None, return_paths=False, max_imgs=None):
    
    paths_dir1 = get_paths(dir1, extension=extension_dir1, recursive=True, substring=substring)
    paths_dir2 = get_paths(dir2, extension=extension_dir2, recursive=True, substring=substring)

    if common_subset:
        paths_dir1, paths_dir2 = get_common_subset(paths_dir1, paths_dir2)

    if exclude:
        exclusion_stems = [Path(p).stem for p in exclude]
        paths_dir1 = [p for p in paths_dir1 if Path(p).stem not in exclusion_stems]
        paths_dir2 = [p for p in paths_dir2 if Path(p).stem not in exclusion_stems]

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

# Returns a list of images
def get_image_list(paths):
    return [imageio.mimread(p) for p in paths]

# Given a folder and a file stem, return paths of all files in folder with said stem
def find_stem_in_other_folder(folder, stem):
    folder_paths = get_paths(folder)
    corresponding_path = folder_paths[[Path(f).stem for f in folder_paths].index(stem)]
    return corresponding_path

# Given some paths in a dir, generate corresponding paths in another dir
def get_corresponding_paths(paths, old_dir, new_dir):
    return [os.path.join(new_dir, os.path.relpath(path, old_dir)) for path in paths]

def resize(img, shape):
    return cv2.resize(img, dsize=shape, interpolation=cv2.INTER_LINEAR)

# Rescale appropriate for masks since it uses INTER_NEAREST as interpolation method
def resize_mask(img, shape):
    return cv2.resize(img, dsize=shape, interpolation=cv2.INTER_NEAREST)

def normalize(img, old_range, new_range=(-1, +1)):
    img -= old_range[0]
    img /= old_range[1] / (new_range[1] - new_range[0])
    img += new_range[0]
    return img

if __name__ == "__main__":
    tifdir = "/mnt/DATA2/anton/3564_low_res_imgs/tiffs"
    segdir = "/mnt/DATA2/anton/3564_low_res_imgs/tiffs_watersheds"
    # # x1, x2 = get_two_sets(tifdir, segdir, extension_dir1='.tif')
    # # print(len(x1))
    # x = get_image_array(tifdir, get_paths(tifdir), 1)
    # print(len(x))

    tifpaths, segpaths = get_two_sets(tifdir, segdir, common_subset=True, extension_dir1='.tif', return_paths=True, max_imgs=12)
    train_loader = MicroscopyDataset(tifpaths, segpaths, batch_size=4)
    # print(next(iter(train_loader))[0].shape)
    train_loader.__getitem__(2)