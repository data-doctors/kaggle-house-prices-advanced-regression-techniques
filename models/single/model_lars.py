import pandas as pd 
import numpy as np
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import sys
import time
import warnings

from sklearn import linear_model

warnings.filterwarnings('ignore')

folds = 5
seed = 7

model_label='lars'

def rmse_cv(model, X_train, y_train):
    kfold = KFold(n_splits=folds, random_state=seed)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kfold))
    return(rmse)

def main(predictions = False):

    train = pd.read_csv("./data/X_train_v1.csv")

    y_train = train['SalePrice']
    X_train = train.loc[:,'MSSubClass':'SaleCondition_Partial']

    test = pd.read_csv("./data/X_test_v1.csv")
    X_test = test.loc[:,'MSSubClass':'SaleCondition_Partial']

    # build model
    model = linear_model.Lars(n_nonzero_coefs=64)

    # fit model
    model.fit(X_train, y_train)

    # evaluate model
    results = rmse_cv(model, X_train, y_train)
    print("RMSE-{}-CV({})={:06.5f}+-{:06.5f}".format(model_label, folds, results.mean(), results.std()))

    # # predict
    if predictions:
        y_test_pred_log = model.predict(X_test)
        y_test_pred = np.expm1(y_test_pred_log)
        submission = pd.DataFrame({'Id':test['Id'], 'SalePrice':y_test_pred})

        subFileName = "./submissions/sub-" + model_label + "-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
        print("saving to file: " + subFileName)
        submission.to_csv(subFileName, index=False)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    savePredict = False
    if len(sys.argv) > 1:
        if sys.argv[1] == 'save':
            savePredict = True
    main(predictions = savePredict)
