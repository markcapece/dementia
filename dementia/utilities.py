# Imports
# --Python
import numpy as np
import pandas as pd

# --Scikit Learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def transform_data(cross_path='data/oasis_cross-sectional.csv', long_path='data/oasis_longitudinal.csv'):
    """Loads OASIS1 and OASIS2 datasets into pandas DataFrames

    Renames M/F and EDUC columns
    Adds num_sex, Group, CDR_Group, and num_group columns
    Transforms OASIS2 Edu column into OASIS1 format
    Merges OASIS1 and OASIS2 datasets

    Keyword arguments:
        cross_path -- path to OASIS1 CSV
        long_path -- path to OASIS2 CSV

    Returns:
         tuple of (OASIS1, OASIS2, merged DataFrames)
    """
    cross = pd.read_csv(cross_path, index_col='ID')
    long = pd.read_csv(long_path, index_col='MRI ID')

    group_map = {0: 'Nondemented', 0.5: 'Demented', 1: 'Demented', 2: 'Demented'}
    cdr_map = {np.nan: np.nan, 0: 0, 0.5: 0.5, 1: 1, 2: 1}
    sex_map = {'F': 0, 'M': 1}

    for d in [cross, long]:
        d_columns = list(d.columns)
        for n, col in enumerate(d_columns):
            if col.upper() == 'M/F':
                d_columns[n] = 'Sex'
            elif col.upper() == 'EDUC':
                d_columns[n] = 'Edu'
        d.columns = d_columns
        d.dropna(subset=['CDR'], axis=0, inplace=True)

        d['num_sex'] = d['Sex'].map(sex_map)
        d['Group'] = d['CDR'].map(group_map)
        d['CDR_Group'] = d['CDR'].map(cdr_map)
        d['num_group'] = d['Group'].apply(lambda x: 0 if x == 'Nondemented' else 1 if x == 'Demented' else np.nan)

    long['Edu'] = long['Edu'].apply(lambda x: 1 if x < 12 else 2 if x == 12 else 3 if x < 16 else 4 if x == 16 else 5)
    cross = pd.concat([cross, pd.get_dummies(cross['Group'])], axis=1)
    long = pd.concat([long, pd.get_dummies(long['Group'])], axis=1)

    return cross, long, pd.concat([cross, long], axis=0, sort=True)


def principal_component_transformation(df, features, targets, var=0.95):
    """Fits PCA and transforms features into PC frame

    Positional arguments:
        df -- the pandas DataFrame to be decomposed
        features -- list of column names that will be treated as machine learning inputs
        targets -- list of column names that will be treated as machine learning outputs

    Keyword arguments:
        var -- target variance to define minimum number of principal components

    Returns:
         tuple of (PCA model object, transformed feature DataFrame)
    """
    pca = PCA(n_components=var)
    scaler = StandardScaler()

    df_nona = df[features + targets].dropna()
    df_features = df_nona[features]
    df_targets = df_nona[targets]
    df_index = df_nona.index

    df_features_scaled = scaler.fit_transform(df_features)
    pca.fit(df_features_scaled, df_targets)
    df_features_pc = pca.transform(df_features_scaled)

    output = pd.DataFrame(df_features_pc, columns=['PC{}'.format(i + 1) for i in range(df_features_pc.shape[1])],
                          index=df_index)
    output[targets] = df_targets

    return pca, output
