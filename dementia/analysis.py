# Import
# --Python
from itertools import accumulate, combinations
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --Scikit Learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

# --Local
try:
    from dementia.param_map import PARAM_MAP
except ImportError:
    PARAM_MAP = {}


class Analysis(object):
    """Contains methods for statistical analysis and machine learning of a DataFrame"""
    def __init__(self, df, features=None, targets=None, scale=False):
        """Records feature and target names and creates separate feature and target DataFrames

        Positional argument:
            df -- the pandas DataFrame to be analyzed

        Keyword arguments:
            features -- list of column names that will be treated as machine learning inputs
            targets -- list of column names that will be treated as machine learning outputs
            scale -- bool; if True, uses StandardScaler on all feature columns
        """
        if not isinstance(features, list) or not isinstance(targets, list):
            raise ValueError('Must give list of features and list of targets')

        self.df = df
        self.features = features
        self.feature_names_string = ', '.join(features)
        self.targets = targets
        self.columns = self.df.columns
        self.scale = scale
        self.x, self.y = self._make_features_targets()

    def _make_features_targets(self):
        df = self.df[self.features + self.targets].copy()
        df.dropna(how='any', axis=0, inplace=True)
        df_features = df[self.features]
        df_targets = df[self.targets]
        df_index = df.index

        if self.scale:
            scaler = StandardScaler()
            df_features = scaler.fit_transform(df_features)
            df_features = pd.DataFrame(df_features, columns=self.features, index=df_index)

        return df_features, df_targets

    def machine_learning(self, models=None, param_map=PARAM_MAP):
        """Finds the best parameters and determines the cross val score for a list of models

        Keyword arguments:
            models -- list of sklearn model objects
            param_map -- dictionary of key=sklearn model name, value=dictionary of parameter values to be optimized

        Returns:
            pandas DataFrame of the mean, std, and parameters of the cross-validated best model for each algorithm
        """
        if not models:
            raise ValueError('Must give a list of sklearn models')

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.20, random_state=42)
        result = {}
        kfolds = 5

        for m in models:
            m_name = m.__class__.__name__
            try:
                search = GridSearchCV(estimator=m, param_grid=param_map[m_name])
                search.fit(x_train, y_train)
                best_search = search.best_estimator_
            except KeyError:
                best_search = m
            finally:
                score = cross_val_score(best_search, self.x, self.y, cv=kfolds)
                params = best_search.get_params()

            result[m_name] = {
                'mean': score.mean(),
                'std': score.std(),
                'params': params
            }

            result[m_name] = {(self.feature_names_string, 'mean'): score.mean(),
                              (self.feature_names_string, 'std'): score.std(),
                              (self.feature_names_string, 'params'): params}

        return pd.DataFrame(result)

    def ml_table(self, models=None, param_map=PARAM_MAP, array=None):
        """Performs combinatoric arrangement of features and executes machine_learning() with each combination

        Keyword arguments:
            models -- list of sklearn model objects
            param_map -- dictionary of key=sklearn model name, value=dictionary of parameter values to be optimized
            array -- 'combination' or 'accumulation' feature aggregation techniques

        Returns:
            pandas DataFrame of the mean, std, and parameters of the cross-validated best model for each algorithm
        """
        if not array:
            raise ValueError('No feature aggregation type given')

        array_comb = {'combination', 'c', 'combo', 'com'}
        array_accum = {'accumulation', 'a', 'accum', 'acc'}

        num_inputs = len(self.features)
        out_dict = {}

        print('Now analyzing: ', end='', flush=True)

        if array in array_comb:
            for n in range(num_inputs):
                n += 1
                list_of_features = list(combinations(self.features, n))

                for f in list_of_features:
                    feat = list(f)
                    print(str(feat), end=' ... ', flush=True)
                    temp_analysis = Analysis(self.df, feat, self.targets)
                    out_dict[str(feat)] = temp_analysis.machine_learning(models=models, param_map=param_map)

        elif array in array_accum:
            list_of_features = list(accumulate([[f] for f in self.features]))

            for feat in list_of_features:
                print(str(feat), end=' ... ', flush=True)
                temp_analysis = Analysis(self.df, feat, self.targets)
                out_dict[str(feat)] = temp_analysis.machine_learning(models=models, param_map=param_map)

        else:
            raise ValueError('Invalid array value')

        print('Done')

        out_df = pd.concat([out_dict[x] for x in out_dict])
        out_df.rename_axis(['Features', 'Statistics'])

        return out_df

    def violin_plots(self, x=None, hue='Sex'):
        """Plots a violin plot of each feature column

        Keyword arguments:
            x -- column name to group the data along the x-axis (recommend 'Group')
            hue -- column name to split the violin plots (default 'Sex')
        """
        number_of_columns = len(self.features)
        df_hue = self.df[hue] if hue else None
        split = True if any(df_hue) else False
        df_x = self.df[x] if x else None
        n = 1

        for c in self.features:
            if c != hue:
                try:
                    plt.subplot(np.ceil(number_of_columns / 2), 2, n)
                    sns.violinplot(x=df_x, y=self.df[c], hue=df_hue, split=split)
                    n += 1
                except KeyError:
                    continue
        plt.show()

    def three_d_plot(self, axes):
        """Plots a 3D scatter plot of three feature columns

        Positional argument:
            axes -- list or tuple of at least three column names; names after the first three are ignored
        """
        def color_map(c):
            if c == 0:
                return 'green'
            elif c == 0.5:
                return 'orange'
            elif c == 1:
                return 'red'
            elif c == 2:
                return 'purple'
            else:
                return 'white'

        color = list(map(color_map, self.df['CDR'].values))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.df[axes[0]], self.df[axes[1]], self.df[axes[2]], c=color)
        ax.view_init(30, 250)
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.set_zlabel(axes[2])
        plt.show()

    def stats_table(self, sort_by=None, stats=('mean', 'std')):
        """Gives a table of simple statistics for each column

        Keyword arguments:
            sort_by -- list of column names by which to group the data
            stats -- list or tuple of statistic names from the pandas describe() function (default ('mean', 'std'))

        Returns:
            pandas DataFrame of statistics
        """
        if not sort_by:
            sort_by = self.targets

        describe = self.df[self.features + sort_by].groupby(sort_by).describe()

        table = pd.DataFrame()
        for c in self.features:
            for s in stats:
                table = pd.concat([table, describe[c][s]], axis=1)
        table.columns = pd.MultiIndex.from_product([self.features, stats], names=['Features', 'Statistics'])

        return table

    def skew_and_kurtosis(self):
        """Gives a table of skew and kurtosis indexed by column"""
        skew_kurt = {'Skew': {}, 'Kurtosis': {}}
        for c in self.features:
            skew_kurt['Skew'][c] = self.df[c].skew(skipna=True)
            skew_kurt['Kurtosis'][c] = self.df[c].kurtosis(skipna=True)

        return pd.DataFrame(skew_kurt)

    def correlations(self):
        """Plots a scatter matrix and gives a table of r values between each permutation of two columns"""
        plt.figure(figsize=(16, 12))
        pd.plotting.scatter_matrix(self.df[self.features], diagonal='hist')
        plt.show()

        return self.df[self.features].corr()

    def __str__(self):
        return 'Analysis object for DataFrame with features {} and targets {}\n{}'.format(
            self.features, self.targets, self.df.head()
        )

    def __repr__(self):
        return str(self)
