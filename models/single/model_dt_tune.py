import pandas as pd 
import numpy as np
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import sys
import time

from sklearn.tree import DecisionTreeRegressor

folds = 7
seed = 7

model='dt'

def rmse_cv(model, X_train, y_train):
    kfold = KFold(n_splits=folds, random_state=seed)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kfold)).mean()
    return(rmse)



# # load clean dataset
# data = pd.read_csv("./data/clean_dataset.csv")

# # split to train/test
# train = data.loc[data.set == 'train', data.columns.values[1:]]
# test = data.loc[data.set == 'test', data.columns.values[1:]]

# X_train = train.drop(['Id', 'SalePrice', 'logSalePrice', 'set'], axis=1)
# y_train = np.log1p(train['SalePrice'])

# X_test = test.drop(['Id', 'SalePrice', 'logSalePrice', 'set'], axis=1)

X_train = pd.read_csv("./data/X_train_v1.csv")

y_train = X_train['SalePrice']
X_train = X_train.loc[:,'MSSubClass':'SaleCondition_Partial']

X_test = pd.read_csv("./data/X_test_v1.csv")
X_test = X_test.loc[:,'MSSubClass':'SaleCondition_Partial']

# dt tunning
model = DecisionTreeRegressor()

# dict with tunning parameters
param_grid = {
    'max_depth': range(2, 11, 2), 
    'min_samples_split': [2, 3, 4]
}

seed = 7
kfold = KFold(n_splits=7, random_state=seed)

#grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=1, cv=kfold, verbose=1)
grid_search = GridSearchCV(model, param_grid, scoring=rmse_cv, n_jobs=1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
	#print("%f (%f) with: %r" % (mean, stdev, param))
    print("{:06.5f} ({:06.5f}) with {}".format(mean, stdev, param))





