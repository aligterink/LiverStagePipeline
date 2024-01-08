import sys
import os
sys.path.append(os.path.abspath('').split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import get_models as get_models
from segmentation.AI.datasets import MicroscopyDataset
from segmentation.evaluate import Inferenceinator
from segmentation.conventional.cell_watershed import segment_cells_in_folder as conv_seg


from utils import setup, data_utils, cell_viewer
from feature_analysis import features
from feature_analysis.cell_dataset import CellDataset
from feature_analysis.VAE_models import VAE
from feature_analysis.train import train_VAE
from feature_analysis import inference, dimensionality_reduction, plot_lowdim, evaluate_clusters

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if not torch.cuda.is_available():
    print("Using non-cuda device: {}".format(device))



# Global settings
session_name = '6c'#'FoI_1'
pipeline_files_folder = '/mnt/DATA1/anton/pipeline_files'

# hsp_channel, dapi_channel = 1, 0 # lowres
hsp_channel, dapi_channel = 0, 2 # highres
threads = 24


paths = setup.setup(pipeline_files_folder, session_name=session_name)


# settingsd
model_path = '/mnt/DATA1/anton/pipeline_files/segmentation/models/best_model.pth'
# tif_folder = '/mnt/DATA1/anton/data/force_of_infection/tifs'
# tif_folder = '/mnt/DATA1/anton/data/parasite_annotated_dataset/images/lowres/NF175/D3'
tif_folder = '/mnt/DATA3/compounds/11C-organised'
segmentation_batch_size = 10



image_paths = data_utils.get_paths(tif_folder) # list of paths to tif files

# print(len(image_paths))
# seg_stems = [os.path.basename(p) for p in data_utils.get_paths(paths['parasite_masks_folder'])]
# image_paths = [p for p in image_paths if not os.path.basename(p) in seg_stems]
# print(len(image_paths))

channels = [[dapi_channel, hsp_channel] for img_path in image_paths]


# settings
feature_dict = features.default_feature_dict(hsp_channel=hsp_channel, dapi_channel=dapi_channel)

def annotated_metadata(tif_path):
    strain_mapping = {'NF135': 135, 'NF54': 54, 'NF175': 175}
    day_mapping = {'D3': 3, 'D5': 5, 'D7': 7}

    return {'strain': [strain_mapping[s] for s in strain_mapping.keys() if s in tif_path][0],
            'day': [day_mapping[s] for s in day_mapping.keys() if s in tif_path][0]}

def foi_metadata(tif_path):
    return {'strain': int(tif_path.split('/')[7][2:]), 'foi': tif_path.split('/')[8], 'foi_ratio': int(tif_path.split('/')[8].strip('h').split('s')[0]) / int(tif_path.split('/')[8].strip('h').split('s')[1])}

def sixc_metadata(tif_path):
    return {'drug': tif_path.split('/')[5]}

metadata_func = sixc_metadata

print(feature_dict)


# feature extraction
features.collect_features_from_folder(tif_folder=tif_folder, parasite_mask_folder=paths['parasite_masks_folder'], feature_dict=feature_dict, hepatocyte_mask_folder=paths['hepatocyte_masks_folder'], 
                                      csv_path=paths['feature_file'], metadata_func=metadata_func, workers=threads)





# VAE train settings
hidden_dim = 250
latent_dim = 10
learning_rate = 0.001
batch_size = 48
epochs = 150
VAE_mode = 'vanilla'
beta = None
non_feature_columns = ['file', 'label', 'drug']

vae_identifier = '{}_{}LD_{}-VAE_{}epochs'.format(session_name, latent_dim, VAE_mode, epochs)





# Setting some paths
model_path = os.path.join(paths['FA_models_folder'], '{}.pth'.format(vae_identifier))
latent_space_path = os.path.join(paths['FA_latent_spaces_folder'], '{}.csv'.format(vae_identifier))
embeddings_path = os.path.join(paths['FA_embeddings_folder'], '{}.csv'.format(vae_identifier))

#### Creating the dataloaders
df = pd.read_csv(paths['feature_file'])
# df['force_of_infection-strain'] = ['{}_{}'.format(int(strain), foi) for strain, foi in zip(df['strain'], df['force_of_infection'])]
# print(df.groupby(['strain'])['strain'].count())

msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]

train_ds = CellDataset(train_df, non_feature_columns=non_feature_columns)
test_ds = CellDataset(test_df, non_feature_columns=non_feature_columns)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

### Training the VAE
model = VAE(len(train_ds.features.columns), hidden_dim, latent_dim, mode=VAE_mode, beta=beta) # create VAE model
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # define optimizer

train_VAE(model, train_loader, test_loader, epochs, optimizer, model_path, device)





### inference
model.load_state_dict(torch.load(model_path)) # load best performing version of model

complete_ds = CellDataset(df, non_feature_columns=non_feature_columns)
complete_loader = DataLoader(complete_ds, batch_size=batch_size, shuffle=False)

latent_space_df = inference.get_latent_embedding(model, complete_loader, device, latent_space_path=latent_space_path)






# # Dimension reduction settings
# dimensionality_reducer = 'umap'

# features_for_dim_reduction_plots = ['drug']#['strain', 'foi', 'foi_ratio']
# marker_size = 2
# alpha = 0.5
# colormap = 'Dark2'

# # Distance matrix settings
# features_for_distance_matrix = ['drug']#['strain', 'foi', 'foi_ratio']





# # Dimensionality reduction plotting
# for feature in features_for_dim_reduction_plots:
#     figure_path = os.path.join(paths['FA_latent_space_plots_folder'], '{}_{}-of-{}.png'.format(vae_identifier, dimensionality_reducer.upper(), feature))
#     title = '{} of {}'.format(dimensionality_reducer.upper(), feature)
#     labels = df[feature]

#     plot_lowdim.plot(embeddings_path, labels, title=title, save_path=figure_path, show=False, marker_size=marker_size, alpha=alpha, colormap=colormap)





# # Generating distance matrices
# latent_space_df = pd.read_csv(latent_space_path)
# for feature in features_for_distance_matrix:
#     figure_path = os.path.join(paths['FA_latent_space_matrices_folder'], '{}_silhouette_scores_of_{}.png'.format(vae_identifier, feature))
#     labels = df[feature]

#     mean, scores = evaluate_clusters.get_sillhouette_scores(latent_space_df, labels)
#     evaluate_clusters.plot_distance_matrix(scores, show=False, save_path=figure_path)