import pandas as pd
import numpy as np
from scipy.stats import f_oneway

def anova(features, groups):
    feature_names, group_names = features.columns, groups.columns
    f_statistics = pd.DataFrame(columns=group_names, index=feature_names)
    p_values = pd.DataFrame(columns=group_names, index=feature_names, dtype=np.float128)
    
    for feature in features.columns:
        for group in groups.columns:
            vals = [features.loc[groups[group] == u, feature].tolist() for u in groups[group].unique()]
            anova_res = f_oneway(*vals, axis=0)
            f_statistics.at[feature, group] = anova_res[0]
            p_values.at[feature, group] = anova_res[1]

    print('& ' + ' & '.join(groups.columns))
    for feature in features.columns:
        feature_str = feature.replace('_', '\\_')
        for group in groups.columns:
            feature_str += ' & {} & {}'.format(round(f_statistics.at[feature, group], 2), '{:0.3e}'.format(p_values.at[feature, group]) if p_values.at[feature, group] > 1e-99999999999 else 0)
        feature_str += ' \\\\'
        print(feature_str)

    return f_statistics, p_values

if __name__ == '__main__':
    df = pd.read_csv('/mnt/DATA1/anton/pipeline_files/feature_analysis/features/lowres_dataset_selection_features.csv')

    df = df[df.strain != 54].reset_index().drop(['index'], axis=1)

    # group_names = ['day', 'strain']
    # groups = df[group_names]
    # features = df.drop(['file', 'label'] + group_names, axis=1)

    x = ['NF{}-D{}'.format(strain, day) for strain, day in zip(df['strain'], df['day'])]
    groups = pd.DataFrame({'strain-day': x})
    features = df.drop(['file', 'label', 'strain', 'day'], axis=1)

    features = features.interpolate()
    anova(features, groups)