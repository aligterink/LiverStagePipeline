import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pandas as pd

def plot(embeddings_path, labels, title=None, save_path=None, show=False, marker_size=1, alpha=1, colormap='Paired'):
    embedding = pd.read_csv(embeddings_path).to_numpy()
    unique_labels = np.array(list(set(labels)))

    plt.figure(figsize=(14,6))

    try:
        try:
            # the couple lines below are for plotting the force of infection with gradual colors
            ratios = [int(x.translate({ord(i): None for i in 'sh'}).split('-')[0]) / int(x.translate({ord(i): None for i in 'sh'}).split('-')[1]) for x in unique_labels]
            ratios, unique_labels = zip(*sorted(zip(ratios, unique_labels), reverse=True))
            ratios = np.array(ratios)
            unique_labels = np.array(unique_labels)
            s = np.arange(0.2, 1.2, 1/len(ratios))
            colors = mpl.colormaps[colormap](s)

        except Exception as e:
            s = (unique_labels - unique_labels.min()) / unique_labels.ptp()
            colors = mpl.colormaps[colormap](s)
    except:
        colors = mpl.colormaps[colormap]([i for i in range(len(unique_labels))])

    colors = mpl.colormaps[colormap]([i for i in range(len(unique_labels))])


    # Create a scatter plot and assign colors based on labels
    for i, label in enumerate(set(labels)):
        indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(np.array(embedding[:, 0])[indices], np.array(embedding[:, 1])[indices], color=colors[np.where(unique_labels==label)], label=label, s=marker_size, alpha=alpha)

    # Add a legend with unique colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=label)
                    for i, label in enumerate(unique_labels)]
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
