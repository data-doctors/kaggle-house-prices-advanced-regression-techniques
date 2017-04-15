import pandas as pd 
import numpy as np
import xgboost as xgb
from matplotlib import pyplot
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import sys

folds = 7
seed = 7

model='xgb'

def rmse_cv(model, X_train, y_train):
    kfold = KFold(n_splits=folds, random_state=seed)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kfold))
    return(rmse)

def main(predictions = False):

    # ----------------------------------------------------------------------------

    data = pd.read_csv("./data/clean_dataset.csv")

    train = data.loc[data.set == 'train', data.columns.values[1:]]
    test = data.loc[data.set == 'test', data.columns.values[1:]]

    X_train = train.drop(['Id', 'SalePrice', 'logSalePrice', 'set'], axis=1)
    y_train = np.log1p(train['SalePrice'])

    X_test = test.drop(['Id', 'SalePrice', 'logSalePrice', 'set'], axis=1)

    # ----------------------------------------------------------------------------

    # build model
    model_xgb = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.2,
        max_depth=2
    )

    # fit model
    model_xgb.fit(X_train, y_train, verbose=True)

    # evaluate model
    results = rmse_cv(model_xgb, X_train, y_train)
    print("RMSE-{}-CV({})={}+-{}".format(model, folds, results.mean(), results.std()))

    # # predict
    if predictions:
        y_test_pred_log = model_xgb.predict(X_test)
        y_test_pred = np.expm1(y_test_pred_log)
        submission = pd.DataFrame({'Id':test['Id'], 'SalePrice':y_test_pred})

        print(submission)
        submission.to_csv("./output/sub-xgb-11_04_2017.csv", index=False)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    savePredict = False
    if len(sys.argv) > 1:
        if sys.argv[1] == 'save':
            savePredict = True
    main(predictions = savePredict)