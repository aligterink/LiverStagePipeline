from feature_analysis import features
import utils.data_utils as data_utils
import pandas as pd
import imageio.v3
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # disables the warning


def collect_features(tif_paths, seg_paths, feature_dict, csv_path=None, append=False, overwrite=False):
    df = pd.DataFrame()
    for tif_path, seg_path in zip(tif_paths, seg_paths):
        mask = imageio.v3.imread(seg_path)
        tif = np.array(imageio.mimread(tif_path, memtest=False)).transpose(1, 2, 0)
        extractor = features.Extractor(mask, tif, Path(tif_path).stem)
        features = extractor.get_features(feature_dict)
        df = pd.concat([df, features])

    if csv_path:
        isfile = Path(csv_path).is_file()
        if (isfile and overwrite) or not isfile:
            df.to_csv(csv_path, index=False)
    return df

if __name__ == "__main__":
    # tif_dir = R"C:\Users\anton\Documents\microscopy_data\dataset\images\NF54"
    # seg_dir = R"C:\Users\anton\Documents\microscopy_data\dataset\annotation\NF54"

    # csv_path = R"C:\Users\anton\Documents\microscopy_data\results\features.csv"
                    
    # paths = data_utils.get_two_sets(tif_dir, seg_dir, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)

    # feature_dict = {-1: ('', ['area', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 'extent', 'feret_diameter_max', 'orientation', 'perimeter', 'perimeter_crofton', 'solidity']), 
    #                 0: ('dapi', ['normalized_intensity_sum']), 
    #                 1: ('hsp', ['normalized_intensity_sum'])}

    # x = collect_features(paths[0], paths[1], feature_dict, csv_path, append=False, overwrite=False)
    # print(x)



    img_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/images/NF135/D5/2019003_D5_135_hsp_20x_2_series_1_TileScan_001.tif"
    mask_path = "/mnt/DATA1/anton/data/lowres_dataset_selection/annotation/NF135/D5/2019003_D5_135_hsp_20x_2_series_1_TileScan_001.png"

    img = np.array(imageio.mimread(img_path)).transpose(1, 2, 0)
    mask = imageio.v3.imread(mask_path)

    x = features.Extractor(mask, img, '2019003_D5_135_hsp_20x_2_series_1_TileScan_001')

    mask_features = ['area', 'area_convex'] #, 'area_filled', 'axis_major_length', 'axis_minor_length',
                #  'eccentricity', 'equivalent_diameter_area', 'extent', 'feret_diameter_max', 'orientation',
                #  'perimeter', 'perimeter_crofton', 'solidity']
    default_channel_features = ['normalized_intensity_sum']

    feature_dict = {-1: ('', ['area', 'area_convex']), 0: ('dapi', ['num_local_maxima']), 1: ('hsp', [])}
    features1 = x.get_features(feature_dict)
    print(features1)
