from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import numpy as np
import pandas as pd

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_umap(df, n_components=2):
    embedding = umap.UMAP(n_neighbors=30, n_components=n_components, random_state=43, min_dist=0.01).fit(df).embedding_
    embedding_df = pd.DataFrame(embedding, columns=['Dimension {}'.format(i) for i in range(embedding.shape[1])])
    return embedding_df

def get_tsne(df, n_components=2):
    tsne = TSNE(n_components=n_components)
    embedding = tsne.fit_transform(df)
    embedding_df = pd.DataFrame(embedding, columns=['Dimension {}'.format(i) for i in range(embedding.shape[1])])
    return embedding_df

def get_pca(df, n_components=2):
    pca = PCA(n_components=n_components).fit(df)
    embedding = pca.transform(df)
    print('PCA: {} components explained {} of variance'.format(pca.n_components_, round(sum(pca.explained_variance_ratio_), 2)))
    embedding_df = pd.DataFrame(embedding, columns=['Dimension {}'.format(i) for i in range(embedding.shape[1])])

    return embedding_df

def reduce(name, df, n_components=2, save_path=None):
    if name == 'umap':
        embedding_df = get_umap(df, n_components=n_components)
    elif name == 'tsne':
        embedding_df = get_tsne(df, n_components=n_components)
    elif name == 'pca':
        embedding_df = get_pca(df, n_components=n_components)

    if save_path:
        embedding_df.to_csv(save_path, index=False)
    
    return embedding_df


if __name__ == '__main__':
    feature = 'strain'
    latent_space_path = '/mnt/DATA1/anton/pipeline_files/results/latent_spaces/FoI.csv'

    from pathlib import Path
    import os

    save_path = os.path.join("/mnt/DATA1/anton/pipeline_files/results/figures/VAE", '{}_{}.png'.format(Path(latent_space_path).stem, feature)) 

    df = read_features(latent_space_path)

    plot(df, feature, 'umap')