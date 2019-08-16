from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


PARAM_MAP = {
    'SVC':
    {
        'C': [0.1, 0.5, 1, 2, 4],
        'gamma': ['scale', 0, 0.1, 1, 10, 100]
    },
    'RandomForestClassifier':
    {
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [2, 5, 10, 20],
        'max_leaf_nodes': [2, 5, 10, 20],
        'min_impurity_decrease': [1e-3, 1e-2, 1e-1, 0]
    },
    'GradientBoostingClassifier':
    {
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [2, 5, 10, 20],
        'max_leaf_nodes': [2, 5, 10, 20],
        'min_impurity_decrease': [1e-3, 1e-2, 1e-1, 0]
    },
    'AdaBoostClassifier':
    {
        'n_estimators': [1, 10, 100, 1000],
        'base_estimator': [DecisionTreeClassifier(), RandomForestClassifier()]
    },
    'MLPClassifier':
    {
        'hidden_layer_sizes': [(16,), (64,), (64, 64), (128, 128), (128, 128, 128)],
        'max_iter': [10, 100, 200, 500, 1000]
    },
    'DecisionTreeClassifier':
    {
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [2, 5, 10, 20],
        'max_leaf_nodes': [2, 5, 10, 20],
        'min_impurity_decrease': [1e-3, 1e-2, 1e-1, 0]
    },
    'KNeighborsClassifier':
    {
        'n_neighbors': [2, 5, 10, 20],
        'leaf_size': [5, 10, 20, 30, 40],
        'p': [1, 2]
    }
}