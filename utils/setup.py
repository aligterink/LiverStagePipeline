import os
import sys

def setup(pipeline_files_folder):
    paths = {
        'pipeline_files_folder': pipeline_files_folder, 
        'segmentation_folder': os.path.join(pipeline_files_folder, 'segmentation'),
        'feature_analysis_folder': os.path.join(pipeline_files_folder, 'feature_analysis')
    }
    paths.update({
        'segmentation_models_folder': os.path.join(paths['segmentation_folder'], 'models'),
        'segmentation_figures_folder': os.path.join(paths['segmentation_folder'], 'figures'),
        'segmentation_logs_folder': os.path.join(paths['segmentation_folder'], 'logs'),
        'segmentation_crops_folder': os.path.join(paths['segmentation_folder'], 'crops'),

        'FA_features_folder': os.path.join(paths['feature_analysis_folder'], 'features'),
        'FA_models_folder': os.path.join(paths['feature_analysis_folder'], 'models'),
        'FA_latent_spaces_folder': os.path.join(paths['feature_analysis_folder'], 'latent_spaces'),
        'FA_embeddings_folder': os.path.join(paths['feature_analysis_folder'], 'embeddings'),
        'FA_figures_folder': os.path.join(paths['feature_analysis_folder'], 'figures')
    })
    paths.update({
        'FA_latent_space_plots_folder': os.path.join(paths['FA_figures_folder'], 'latent_space_plots'),
        'FA_latent_space_matrices_folder': os.path.join(paths['FA_figures_folder'], 'latent_space_matrices')
    })
    
    for folder in paths.values():
        os.makedirs(folder, exist_ok=True)
    
    return paths
