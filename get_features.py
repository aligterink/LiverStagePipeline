import feature_analysis.features as F
import utils.data_utils as data_utils
import pandas as pd
import imageio.v3
from pathlib import Path
import numpy as np
tif_dir = R"C:\Users\anton\Documents\microscopy_data\dataset\images\NF54"
seg_dir = R"C:\Users\anton\Documents\microscopy_data\dataset\annotation\NF54"

csv_path = R"C:\Users\anton\Documents\microscopy_data\results\features.csv"
                
paths = data_utils.get_two_sets(tif_dir, seg_dir, common_subset=True, extension_dir1='.tif', extension_dir2='.png', return_paths=True)


def collect_features(tif_paths, seg_paths, feature_dict, csv_path=None, append=False, overwrite=False):
    df = pd.DataFrame()
    for tif_path, seg_path in zip(tif_paths, seg_paths):
        mask = imageio.v3.imread(seg_path)
        tif = np.array(imageio.mimread(tif_path)).transpose(1, 2, 0)
        extractor = F.Extractor(mask, tif, Path(tif_path).stem)
        features = extractor.get_features(feature_dict)
        df = pd.concat([df, features])

    if csv_path:
        isfile = Path(csv_path).is_file()
        if (isfile and overwrite) or not isfile:
            df.to_csv(csv_path, index=False)
    return df

feature_dict = {-1: ('', ['area', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 'extent', 'feret_diameter_max', 'orientation', 'perimeter', 'perimeter_crofton', 'solidity']), 
                0: ('dapi', ['normalized_intensity_sum']), 
                1: ('hsp', ['normalized_intensity_sum'])}

x = collect_features(paths[0], paths[1], feature_dict, csv_path, append=False, overwrite=False)
print(x)