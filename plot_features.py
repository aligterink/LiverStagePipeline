import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_features(path, features):
    values_dict = {}
    df = pd.read_csv(path)
    for feature in features:
        values_dict[feature] = df[feature].tolist()
    return values_dict

def zonal_plot():
    strain1 = read_features(R"C:\Users\anton\Documents\microscopy_data\results\features.csv", ['orientation', 'area_convex'])
    strain2 = read_features(R"C:\Users\anton\Documents\microscopy_data\results\features.csv", ['axis_major_length', 'axis_minor_length'])
    strain3 = read_features(R"C:\Users\anton\Documents\microscopy_data\results\features.csv", ['eccentricity', 'equivalent_diameter_area'])

    fig, axs = plt.subplots(ncols=3, sharex=False, sharey=False, figsize=(12, 3))

    axs[0].scatter(strain1['area_convex'], strain1['orientation'])
    axs[1].scatter(strain2['axis_major_length'], strain2['axis_minor_length'])
    axs[2].scatter(strain3['eccentricity'], strain3['equivalent_diameter_area'])

    axs[0].set_ylabel('dsf')
    axs[1].set_xlabel('fsd')

    plt.show()


if __name__ == '__main__':
    zonal_plot()

