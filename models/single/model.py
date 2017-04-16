import pandas as pd 
import numpy as np
import xgboost as xgb
from matplotlib import pyplot
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("./data/clean_dataset.csv")

train = data.loc[data.set == 'train', data.columns.values[1:]]
test = data.loc[data.set == 'test', data.columns.values[1:]]

X_train = train.drop(['Id', 'SalePrice', 'logSalePrice', 'set'], axis=1)
y_train = np.log1p(train['SalePrice'])

X_test = train.drop(['Id', 'SalePrice', 'logSalePrice', 'set'], axis=1)

# evaluate with train/valid split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.33, random_state=7)

# train
start_time = time.time()
print('training on defaults (-1)')
model_xgb = xgb.XGBRegressor()
model_xgb.fit(X_train, y_train)
end_time = time.time()
elapsed = end_time - start_time
print('time elapsed: ' + str(elapsed))

# predict
y_valid_pred_log = model_xgb.predict(X_valid)
y_valid_pred = np.expm1(y_valid_pred_log)

#compare_df = pd.DataFrame({'SalePrice_predic':xgb_pred, 'SalePrice_real':y_valid})

print("RMSE: {}".format(np.sqrt(mean_squared_error(y_valid, y_valid_pred_log))))

# print(compare_df)
#submission.to_csv("./output/sub-gbm-30_03_2017.csv", index=False)
#submission.head()





# results = []
# num_threads = [1, 2, 3, 4, 5, 6, 7, 8]

# for threads in num_threads:
#     start_time = time.time()
#     print('training on: ' + str(threads) + " threads")
#     model_xgb = xgb.XGBRegressor(nthread=threads)
#     model_xgb.fit(X_train, y_train)

#     end_time = time.time()
#     elapsed = end_time - start_time
#     print('time elapsed: ' + str(elapsed))

#     results.append(elapsed)

#pyplot.plot(num_threads, results)
#pyplot.show()







