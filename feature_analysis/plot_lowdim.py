import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pandas as pd

def plot(embeddings_path, labels, title=None, save_path=None, show=False, marker_size=1, alpha=1, colormap='Paired', label_settings=None, gradual=False, figsize=(30,20)):
    labels = [str(l) for l in labels]
    embedding = pd.read_csv(embeddings_path).to_numpy()
    plt.figure(figsize=figsize)

    if label_settings:
        if isinstance(label_settings[0], list):
            label_order = [l[0] for l in label_settings if l[0] in labels]
            label_names = [l[1] for l in label_settings if l[0] in labels]
        else:
            label_order, label_names = label_settings, label_settings
    else:
        label_order, label_names = list(set(labels)), list(set(labels))

    if gradual:
        s = np.arange(0.2, 1.2, 1/len(label_order))
        colors = mpl.colormaps[colormap](s)
    else:
        colors = mpl.colormaps[colormap]([i for i in range(len(label_order))])

    # Create a scatter plot
    for label, name, color in zip(label_order, label_names, colors):
        indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(np.array(embedding[:, 0])[indices], np.array(embedding[:, 1])[indices], color=color, label=label, s=marker_size, alpha=alpha)

    # Add a legend with unique colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=name) for i, name in enumerate(label_names)]
    plt.legend(handles=legend_elements)

    # Set the axis labels
    plt.xlabel('dimension 1')
    plt.ylabel('dimension 2')

    # Set the plot title
    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
