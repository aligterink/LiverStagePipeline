import utils.data_utils as data_utils
import segmentation.evaluate as evaluate
import watershed_segmentation_v3 as ws
import multiprocessing
import glob
import os

if __name__ == '__main__':
    multiprocessing.freeze_support()


    tif_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_3564_tiffs\Annie_subset_tiffs"
    true_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_3564_tiffs\Annie_subset_annotations"

    subset = R"\NF135_D3"
    tif_dir, true_dir = tif_dir + subset, true_dir + subset

    seg_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_3564_tiffs\Annie_subset_watershed"


    files = glob.glob(seg_dir + R'\*')
    for f in files:
        os.remove(f)

    ws.segment_dir(tif_dir, seg_dir, threads=8)

    masks_true, masks_pred = data_utils.get_two_sets(true_dir, seg_dir)
    evaluate.eval(masks_true, masks_pred)
