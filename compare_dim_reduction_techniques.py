import sys
import os
sys.path.append(os.path.abspath(__file__).split('LiverStagePipeline')[-2] + 'LiverStagePipeline')

import numpy as np
import pandas as pd

from feature_analysis.cell_dataset import CellDataset
from feature_analysis import evaluate_clusters, dimensionality_reduction

features_path = '/mnt/DATA1/anton/pipeline_files/feature_analysis/features/FoI_features2.csv'
vae_path = '/mnt/DATA1/anton/pipeline_files/feature_analysis/latent_spaces/FoI2_10LD_vanilla-VAE_30epochs.csv'

grouping = 'force_of_infection'

features = pd.read_csv(features_path)
labels = features[grouping]
ds = CellDataset(features, non_feature_columns=['file', 'label', 'strain', 'force_of_infection_ratio', 'force_of_infection'])
normalized_features = pd.DataFrame(ds.features)

mean, scores = evaluate_clusters.get_sillhouette_scores(normalized_features, labels)
evaluate_clusters.plot_distance_matrix(scores, save_path='/mnt/DATA1/anton/dim_reduction_comparison/raw_matrix.png')
print('normalized features', mean)

pca_embeddings = dimensionality_reduction.get_pca(normalized_features, n_components=10)
mean, scores = evaluate_clusters.get_sillhouette_scores(pca_embeddings, labels)
evaluate_clusters.plot_distance_matrix(scores, save_path='/mnt/DATA1/anton/dim_reduction_comparison/PCA_matrix.png')
print('PCA', mean)

umap_embeddings = dimensionality_reduction.get_umap(normalized_features, n_components=10)
mean, scores = evaluate_clusters.get_sillhouette_scores(umap_embeddings, labels)
evaluate_clusters.plot_distance_matrix(scores, save_path='/mnt/DATA1/anton/dim_reduction_comparison/UMAP_matrix.png')
print('umap', mean)

vae_embeddings = pd.read_csv(vae_path)
mean, scores = evaluate_clusters.get_sillhouette_scores(vae_embeddings, labels)
evaluate_clusters.plot_distance_matrix(scores, save_path='/mnt/DATA1/anton/dim_reduction_comparison/VAE_matrix.png')
print('VAE', mean)
