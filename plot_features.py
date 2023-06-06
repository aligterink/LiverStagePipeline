import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import get_features
from utils import data_utils
plt.rcParams.update({'font.size': 16})

def read_features(path, features):
    values_dict = {}
    df = pd.read_csv(path)
    for feature in features:
        values_dict[feature] = df[feature].tolist()
    return values_dict

def pixel2micrometer2(row):
    return row['area'] * ((640.32 * 478.40) / (1392 * 1040))

                
def zonal_plot():
    tif_folder = "/mnt/DATA1/anton/data/unformatted/GS validation data/untreated_tifs"
    seg_folder = "/mnt/DATA1/anton/data/unformatted/GS validation data/untreated_segmentations_2_copypaste_2905"

    feature_dict = {'mask': ('', ['area', 'centre_coords']), -1: ('hsp', ['normalized_intensity_sum'])}
    fig, axs = plt.subplots(ncols=3, sharex=False, sharey=False, figsize=(17,5))

    strains, days = ['NF135', 'NF175', 'NF54'], ['D3', 'D5', 'D7']
    for i, strain in enumerate(strains):
        strain_nr = ''.join(filter(str.isdigit, strain))
        n_cells = 0
        for j, day in enumerate(days):

            substrings = ['{}_{}'.format(day, strain_nr),
                          '{}_{}'.format(day, strain.lower()),
                          '{}_{}'.format(day, strain.upper())]
            
            paths = [data_utils.get_two_sets(tif_folder, seg_folder, substring=substring, extension_dir1='.tif', extension_dir2='.png', common_subset=True, return_paths=True, max_imgs=None) for substring in substrings]
            tif_paths, seg_paths = sum([p[0] for p in paths], []), sum([p[1] for p in paths], [])
            # tif_paths, seg_paths = tif_paths[0:10], seg_paths[0:10]
            df = get_features.collect_features(tif_paths, seg_paths, feature_dict)
            # df = df[df.normalized_intensity_sum_hsp < df.normalized_intensity_sum_hsp.quantile(.95)]
            # df = df[df.area < df.area.quantile(.95)]

            df['micrometer2'] = df.apply(pixel2micrometer2, axis=1)

            n_cells += len(df.index)

            axs[i].scatter(df['micrometer2'], df['normalized_intensity_sum_hsp'], s=0.25)

            # Trendline
            z = np.polyfit(df['micrometer2'], df['normalized_intensity_sum_hsp'], 1) 
            p = np.poly1d(z)
            axs[i].plot(df['micrometer2'], p(df['micrometer2']), label=day)

            subset = df[df.normalized_intensity_sum_hsp > 200000]
            # print(subset)
            print(strain, day, 'total:', len(df.index), '>200k:', len(subset.index))


        # axs[i].annotate(str(n_cells) + ' cells', xy=(500, 10000))
        # axs[i].set_title('{} ({} cells)'.format(strain, n_cells))
        axs[i].set_xlabel('Schizont size (micrometre^2)')
        axs[i].set_ylabel('Intra-schizont hGS level (Raw Int Den unit)')
        axs[i].set_xlim([0,500])
        axs[i].set_ylim([0,0.5e6])
        axs[i].legend()
        
    plt.tight_layout()
    plt.savefig("/mnt/DATA1/anton/pipeline_files/results/figures/hgs_plot.png")
    plt.show()


if __name__ == '__main__':
    zonal_plot()

