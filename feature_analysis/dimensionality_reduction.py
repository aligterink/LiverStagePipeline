from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import pandas as pd

import umap
from sklearn.manifold import TSNE

def get_umap(df, save_path=None):
    embedding = umap.UMAP(n_neighbors=5, n_components=2, random_state=42).fit(df).embedding_
    embedding_df = pd.DataFrame(embedding, columns=['Dimension {}'.format(i) for i in range(embedding.shape[1])])
    if save_path:
        embedding_df.to_csv(save_path, index=False)

    return embedding_df

def get_tsne(df, save_path=None):
    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(df)
    embedding_df = pd.DataFrame(embedding, columns=['Dimension {}'.format(i) for i in range(embedding.shape[1])])
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