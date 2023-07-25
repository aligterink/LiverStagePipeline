import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import setup
from feature_analysis.cell_dataset import CellDataset
from feature_analysis.VAE_models import VAE
from feature_analysis.train import train_VAE
from feature_analysis import inference, dimensionality_reduction, plot_lowdim, evaluate_clusters

torch.set_default_dtype(torch.float64)

### Settings
pipeline_files_folder = '/mnt/DATA1/anton/pipeline_files'
# features_path = "/mnt/DATA1/anton/pipeline_files/feature_analysis/features/untreated_GS_validation_features.csv"
features_path = "/mnt/DATA1/anton/pipeline_files/feature_analysis/features/FoI_features2_no_density_features.csv"
non_feature_columns = ['file', 'label', 'strain', 'force_of_infection_ratio', 'force_of_infection', 'force_of_infection-strain'] #'day']
name = 'FoI2_no_density_features'

# VAE train settings
hidden_dim = 250
latent_dim = 10
learning_rate = 0.001
batch_size = 48
epochs = 30
VAE_mode = 'vanilla'
beta = None

# Distance matrix settings
features_for_distance_matrix = ['force_of_infection-strain']
show_distance_matrices = True

# Dimension reduction settings
dimensionality_reducer = 'umap'
features_for_dim_reduction_plots = ['strain', 'force_of_infection', 'force_of_infection-strain']
show_dim_reduction_plots = False
marker_size = 7
alpha = 0.7
colormap = 'Paired'

paths = setup.setup(pipeline_files_folder)

# Setting some paths
identifier = '{}_{}LD_{}-VAE_{}epochs'.format(name, latent_dim, VAE_mode, epochs)
model_path = os.path.join(paths['FA_models_folder'], '{}.pth'.format(identifier))
latent_space_path = os.path.join(paths['FA_latent_spaces_folder'], '{}.csv'.format(identifier))
embeddings_path = os.path.join(paths['FA_embeddings_folder'], '{}.csv'.format(identifier))

#### Creating the dataloaders
df = pd.read_csv(features_path)
df['force_of_infection-strain'] = ['{}_{}'.format(int(strain), foi) for strain, foi in zip(df['strain'], df['force_of_infection'])]

# # print(df.groupby(['strain'])['strain'].count())

# msk = np.random.rand(len(df)) < 0.8
# train_df = df[msk]
# test_df = df[~msk]

# train_ds = CellDataset(train_df, non_feature_columns=non_feature_columns)
# test_ds = CellDataset(test_df, non_feature_columns=non_feature_columns)

# ### Training the VAE

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# if not torch.cuda.is_available():
#     print("Using non-cuda device: {}".format(device))

# # Create VAE model
# model = VAE(len(train_ds.features.columns), hidden_dim, latent_dim, mode=VAE_mode, beta=beta)

# # Define optimizer
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Create data loaders
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

# train_VAE(model, train_loader, test_loader, epochs, optimizer, model_path, device)

# ### inference
# model.load_state_dict(torch.load(model_path))

# complete_ds = CellDataset(df, non_feature_columns=non_feature_columns)
# complete_loader = DataLoader(complete_ds, batch_size=batch_size, shuffle=False)

# latent_space_df = inference.get_latent_embedding(model, complete_loader, device, latent_space_path=latent_space_path)

latent_space_df = pd.read_csv(latent_space_path)

### distance matrix
for feature in features_for_distance_matrix:
    figure_path = os.path.join(paths['FA_latent_space_matrices_folder'], '{}_silhouette_scores_of_{}.png'.format(identifier, feature))
    labels = df[feature]

    scores = evaluate_clusters.get_sillhouette_scores(latent_space_df, labels)
    evaluate_clusters.plot_distance_matrix(scores, show=show_distance_matrices, save_path=figure_path)

# ### Dimensionality reduction
# if dimensionality_reducer == 'tsne':
#     dimensionality_reduction.get_tsne(latent_space_df, save_path=embeddings_path)
# elif dimensionality_reducer == 'umap':
#     dimensionality_reduction.get_umap(latent_space_df, save_path=embeddings_path)

# ### Dimensionality reduction plotting
# for feature in features_for_dim_reduction_plots:
#     figure_path = os.path.join(paths['FA_latent_space_plots_folder'], '{}_{}-of-{}.png'.format(identifier, dimensionality_reducer.upper(), feature))
#     title = '{} of {}'.format(dimensionality_reducer.upper(), feature)
#     labels = df[feature]

#     plot_lowdim.plot(embeddings_path, labels, title=title, save_path=figure_path, show=show_dim_reduction_plots, marker_size=marker_size, alpha=alpha, colormap=colormap)
