import pandas as pd 
import numpy as np
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

import sys
import time

from sklearn.ensemble import ExtraTreesRegressor

folds = 5
seed = 7

model='et'

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_data(version = 'v2'):

    train = pd.read_csv("../../data/X_train_" + version + ".csv")

    y_train = train['SalePrice']
    X_train = train.loc[:,'MSSubClass':'SaleCondition_Partial']

    test = pd.read_csv("../../data/X_test_" + version + ".csv")
    X_test = test.loc[:,'MSSubClass':'SaleCondition_Partial']

    return X_train, y_train, X_test

def main():
    
    X_train, y_train, X_test = load_data('v2')

    # tunning
    model = ExtraTreesRegressor()

    # dict with tunning parameters
    param_grid = {
        'bootstrap': [True, False],
        'max_depth': range(2, 11, 2), 
        'min_samples_split': [2, 3, 4],
        #'min_samples_leaf': range(1, 20, 5),
        #'max_leaf_nodes': [100],
        'n_estimators': [5, 10, 15, 20, 25]
    }

    kfold = KFold(n_splits=folds, random_state=seed)

    scorer = make_scorer(rmse, greater_is_better=False)
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, verbose=1, scoring=scorer)
    grid_result = grid_search.fit(X_train, y_train)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        #print("%f (%f) with: %r" % (mean, stdev, param))
        print("{:06.5f} ({:06.5f}) with {}".format(mean, stdev, param))

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


if __name__ == "__main__":
    main()
