import watershed_segmentation_v2 as ws
import segmentation.evaluate as evaluate
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    tif_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_4thchannel_experiments\tifs_and_annotation"
    mask_dir = R"C:\Users\anton\Documents\microscopy_data\Annie_4thchannel_experiments\watershed"
    true_dir = tif_dir

    results_file = R"C:\Users\anton\Documents\microscopy_data\results\all_channel4_experiments_WSv2.csv"

    substrings = {'EXP2': ["D3_135_exp2_hsp", "D5_NF135_exp2_hsp", "D7_135_exp2_hsp", "D3_175_exp2_hsp", "D5_NF175_exp2_hsp", "D7_175_exp2_hsp"],
                    'CSP': ['D3_135_csphsp', 'd5_135_csp_hsp', 'D7_nf135_csphsp', 'D3_175_csphsp', 'd5_175_csp_hsp', 'D7_nf175_csphsp'],
                    'GAPDH': ['D3_135_gapdh_hsp', 'd5_135_Gapdh_hsp', 'D7_nf135_gapdhhsp', 'D3_175_gapdh_hsp', 'D5_NF175_gapdh_hsp', 'D7_175_gapdh_hsp'],
                    'hGS': ['D3_135_HSPGS', 'D5_135_HSPGS', 'D7_nf135_gshsp', 'D3_175_HSPGS', 'D5_175_HSPGS', 'D7_175_HSPGS']                  
    }

    substrings = ["D3_135_exp2_hsp", "D5_NF135_exp2_hsp", "D7_135_exp2_hsp", "D3_175_exp2_hsp", "D5_NF175_exp2_hsp", "D7_175_exp2_hsp",
                    'D3_135_csphsp', 'd5_135_csp_hsp', 'D7_nf135_csphsp', 'D3_175_csphsp', 'd5_175_csp_hsp', 'D7_nf175_csphsp',
                    'D3_135_gapdh_hsp', 'd5_135_Gapdh_hsp', 'D7_nf135_gapdhhsp', 'D3_175_gapdh_hsp', 'D5_NF175_gapdh_hsp', 'D7_175_gapdh_hsp',
                    'D3_135_HSPGS', 'D5_135_HSPGS', 'D7_nf135_gshsp', 'D3_175_HSPGS', 'D5_175_HSPGS', 'D7_175_HSPGS']                  


    ws.segment_dir(tif_dir, mask_dir, threads=8, channel=3)
    evaluate.multiple_evaluations(mask_dir, true_dir, results_file, substrings)