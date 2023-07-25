from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import seaborn as sns
import matplotlib.pyplot as plt


# Compute the silhouette score between each pair of clusters
def get_sillhouette_scores(X, labels):
    unique_labels = list(set(labels))
    scores = pd.DataFrame(np.zeros((len(unique_labels), len(unique_labels))), columns=unique_labels, index=unique_labels)

    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                # mask = labels.isin([label_i, label_j]).to_numpy()
                mask = np.array([label in [label_i, label_j] for label in labels])#.to_numpy()
                sub_labels = np.array(labels)[mask]
                sub_X = X[mask].to_numpy()

                # sub_labels = sub_labels[:800]
                # sub_X = sub_X[:800, :]

                score = silhouette_score(X=sub_X, labels=sub_labels, metric='euclidean')
                scores.at[label_i, label_j] = score
                scores.at[label_j, label_i] = score
                print('did ', label_i, label_j)
    
    return scores

def plot_distance_matrix(scores, show=False, save_path=None):
    plt.figure(figsize=(32,16))


    clustermap = sns.clustermap(scores, cmap=sns.light_palette("seagreen", as_cmap=True), method='average', metric='euclidean', annot=True)
    clustermap.cax.set_visible(False)
    clustermap.ax_row_dendrogram.set_visible(False)
    clustermap.ax_col_dendrogram.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


if __name__ == '__main__':
    # features_path = '/mnt/DATA1/anton/pipeline_files/feature_analysis/features/lowres_dataset_selection_features.csv'
    # latent_space_path = '/mnt/DATA1/anton/pipeline_files/feature_analysis/latent_spaces/testin_10LD_vanilla-VAE_1epochs.csv'
    # grouping = 'day'

    features_path = '/mnt/DATA1/anton/pipeline_files/feature_analysis/features/FoI_features2.csv'
    latent_space_path = '/mnt/DATA1/anton/pipeline_files/feature_analysis/latent_spaces/FoI2_10LD_vanilla-VAE_30epochs.csv'
    grouping = 'force_of_infection'

    features = pd.read_csv(features_path)
    latent_space = pd.read_csv(latent_space_path)

    labels = features[grouping]
    labels = ['{}_{}'.format(int(strain), foi) for strain, foi in zip(features['strain'], features['force_of_infection'])]
    assert all(s.startswith('Latent dimension') for s in latent_space.columns), 'all columns in {} should start with \'Latent dimension\''
    assert len(labels) == len(latent_space), '{} contains {} samples while {} contains {}. These files likely do not match.'.format(features_path, len(features), latent_space_path, len(latent_space))

    scores = get_sillhouette_scores(latent_space, labels)
    print(scores)

