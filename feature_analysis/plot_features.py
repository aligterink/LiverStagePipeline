import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from utils import data_utils
import math
from matplotlib.lines import Line2D


plt.rcParams.update({'font.size': 16, 'figure.autolayout': True})


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

def plot_annie_needs_in_30_min():
    df = pd.read_csv("/mnt/DATA1/anton/pipeline_files/results/features/FoI_features.csv")
    df = df.interpolate()
    print(len(df))
    features = ['area', 'avg_1_NN_distance', 'avg_3_NN_distance', 'avg_5_NN_distance']
    FoI_rates = df['force_of_infection'].unique()
    strains = [54, 135, 175]
    for strain in strains:
        strain_df = df[df['strain'] == strain]
        print(len(strain_df))
        fig, axs = plt.subplots(ncols=len(features), nrows=len(FoI_rates), figsize=(24,8))
        for i, rate in enumerate(FoI_rates):
            sub_df = strain_df[strain_df['force_of_infection'] == rate]

            for j, feature in enumerate(features):
                # axs[i,j].set_xlim(sub_df[feature].quantile(0.9))
                axs[i,j].hist(sub_df[feature], bins=150)
                
                # group data & plot histogram
                # sub_df.pivot(columns='strain', values=feature).plot.hist(ax=axs[i,j], bins=300)
                
        for ax, col in zip(axs[0], features):
            ax.set_title(col)

        for ax, row in zip(axs[:,0], FoI_rates):
            ax.set_ylabel(row, rotation=90, size='medium')
        
        fig.suptitle('NF{} - n={}'.format(strain, len(strain_df)))
        
        plt.show()

# def multiplot(features, labels, nrows=4, nbins=300, save_path=None, show=True):
#     labels = labels.tolist() if not isinstance(labels, list) else labels
#     num_features = len(features.columns)
#     unique_labels = list(set(labels))
#     colors = cm.tab10(range(len(unique_labels)))

#     fig, axs = plt.subplots(nrows=nrows, ncols=math.ceil(num_features/nrows), figsize=(100,50))#figsize=(50,25))
#     axs = axs.flatten()

#     for i, ax in enumerate(axs):

#         if i >= num_features:
#             ax.remove()

#         else:
#             patches = []
#             ax.set_xlim([features[features.columns[i]].quantile(0.01), features[features.columns[i]].quantile(0.99)])

#             for j, label in enumerate(unique_labels):
#                 mask = [l == label for l in labels]
#                 sub_df = features[mask][features.columns[i]]
#                 bins = min(len(sub_df.unique()), nbins, max(int(len(sub_df)/10), 10))
#                 patches.append(ax.hist(sub_df, bins=bins, label=label, color=colors[j], density=True, histtype='step', linewidth=4)[2])
#                 ax.set_title(features.columns[i], fontsize=14)
#                 ax.set_yticks([])
#                 ax.tick_params(axis="x", labelsize=8)
#                 xax = ax


#     handles, labels = xax.get_legend_handles_labels()
#     fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper right')#, bbox_to_anchor=(0.965, 0.18))
#     plt.tight_layout(rect=[0.01, 0.01, 0.95, 0.99])  # Adjust layout to make space for the legend
#     plt.subplots_adjust(hspace=0.35)

#     # plt.legend(loc="upper right")
    

#     if save_path:
#         plt.savefig(save_path)
#     if show:
#         plt.show()

# def multiplot(features, labels, nrows=4, nbins=300, save_path=None, show=True, figsize=(100,50)):
#     labels = labels.tolist() if not isinstance(labels, list) else labels
#     num_features = len(features.columns)
#     unique_labels = list(set(labels))
#     colors = cm.tab10(range(len(unique_labels)))

#     fig, axs = plt.subplots(nrows=nrows, ncols=math.ceil(num_features/nrows), figsize=figsize)
#     axs = axs.flatten()


#     for i, ax in enumerate(axs):

#         if i >= num_features:
#             ax.remove()

#         else:
#             patches = []
#             ax.set_xlim([features[features.columns[i]].quantile(0.01), features[features.columns[i]].quantile(0.99)])

#             for j, label in enumerate(unique_labels):
#                 mask = [l == label for l in labels]
#                 sub_df = features[mask][features.columns[i]]
#                 bins = min(len(sub_df.unique()), nbins, max(int(len(sub_df)/10), 10))
#                 patches.append(ax.hist(sub_df, bins=bins, label=label, color=colors[j], density=True, histtype='step', linewidth=4)[2])
#                 ax.set_title(features.columns[i], fontsize=34)
#                 ax.set_yticks([])
#                 ax.tick_params(axis="x", labelsize=18)

#     legend_lines = [Line2D([0], [0], color=colors[i], label=unique_labels[i], linewidth=7) for i in range(len(unique_labels))]
#     fig.legend(handles=legend_lines, labels=unique_labels, bbox_to_anchor=(1, 1), loc='upper right', fontsize=48)

#     plt.tight_layout(rect=[0.01, 0.01, 0.95, 0.99])  # Adjust layout to make space for the legend
#     plt.subplots_adjust(hspace=0.25)

#     # plt.legend(loc="upper right")
    

#     if save_path:
#         plt.savefig(save_path)
#     if show:
#         plt.show()


# def multiplot(features, labels, nrows=4, nbins=300, save_path=None, show=True, figsize=(50,25), label_settings=None):
#     labels = labels.tolist() if not isinstance(labels, list) else labels

#     if label_settings:
#         if isinstance(label_settings[0], list):
#             label_dict = {l[0]: l[1] for l in label_settings}
#             labels = [label_dict[l] for l in labels]
#             unique_labels = [l[1] for l in label_settings if l[1] in labels]
#         else:
#             unique_labels = [l for l in label_settings if l in labels]
#     else:
#         unique_labels = list(set(labels))

#     num_features = len(features.columns)
#     colors = cm.tab20(range(len(unique_labels)))
#     ncols = math.ceil(num_features/nrows)

#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
#     axs = axs.flatten()


#     for i, ax in enumerate(axs):

#         if i >= num_features:
#             ax.remove()

#         else:
#             ax.set_xlim([features[features.columns[i]].quantile(0.01), features[features.columns[i]].quantile(0.99)])

#             for j, label in enumerate(unique_labels):
#                 mask = [l == label for l in labels]
#                 sub_df = features[mask][features.columns[i]]
#                 bins = len(sub_df.unique()) if len(sub_df.unique()) < 1000 else nbins
#                 ax.hist(sub_df, bins=bins, label=label, color=colors[j], density=True, histtype='step', linewidth=2.5)[2]
#             ax.set_title(features.columns[i], fontsize=10 if len(features.columns[i]) < 35 else 12)
#             ax.set_yticks([])
#             ax.tick_params(axis="x", labelsize=8)

#             if (i % ncols) == 0:
#                 ax.set_ylabel('Probability density', fontsize=8)

#     legend_lines = [Line2D([0], [0], color=colors[i], label=unique_labels[i], linewidth=7) for i in range(len(unique_labels))]
#     fig.legend(handles=legend_lines, labels=unique_labels, loc='upper right', ncol=round(len(legend_lines)/1), fontsize=10, bbox_to_anchor=(0.95, 1))
#     plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.90])  # Adjust layout to make space for the legend

#     # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])  # Adjust layout to make space for the legend
#     # plt.subplots_adjust(hspace=0.6, wspace=0.04)

#     # plt.legend(loc="upper right")
    

#     if save_path:
#         plt.savefig(save_path)
#     if show:
#         plt.show()



def multiplot(features, labels, nrows=None, nbins=300, save_path=None, show=True, figsize=(30,20), label_settings=None):
    labels = labels.tolist() if not isinstance(labels, list) else labels


    if label_settings:
        if isinstance(label_settings[0], list):
            label_dict = {l[0]: l[1] for l in label_settings}
            labels = [label_dict[l] for l in labels]
            unique_labels = [l[1] for l in label_settings if l[1] in labels]
        else:
            unique_labels = [l for l in label_settings if l in labels]
    else:
        unique_labels = list(set(labels))

    num_features = len(features.columns)
    colors = cm.tab20(range(len(unique_labels)))
    nrows = math.ceil(num_features/6)
    ncols = math.ceil(num_features/nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()


    for i, ax in enumerate(axs):

        if i >= num_features:
            ax.remove()

        else:
            ax.set_xlim([features[features.columns[i]].quantile(0.01), features[features.columns[i]].quantile(0.99)])

            for j, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                sub_df = features[mask][features.columns[i]]
                bins = len(sub_df.unique()) if len(sub_df.unique()) < 1000 else nbins
                ax.hist(sub_df, bins=bins, label=label, color=colors[j], density=True, histtype='step', linewidth=2.5)[2]
            ax.set_title(features.columns[i], fontsize=14 if len(features.columns[i]) < 35 else 12)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=8)

            if (i % ncols) == 0:
                ax.set_ylabel('Probability density', fontsize=8)

    legend_lines = [Line2D([0], [0], color=colors[i], label=unique_labels[i], linewidth=7) for i in range(len(unique_labels))]
    fig.legend(handles=legend_lines, labels=unique_labels, loc='upper right', ncol=round(len(legend_lines)/1), fontsize=14, bbox_to_anchor=(0.95, 1))
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])  # Adjust layout to make space for the legend
    plt.subplots_adjust(hspace=0.6, wspace=0.04)

    # plt.legend(loc="upper right")
    

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('/mnt/DATA1/anton/pipeline_files/feature_analysis/features/6c.csv').head(1000)
    labels = list(df['drug'])
    features = df.drop(['file', 'label', 'drug'], axis=1)
    features = features.interpolate()

    multiplot(features, labels, nrows=11, nbins=120)

