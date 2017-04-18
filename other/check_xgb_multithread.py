import pandas as pd 
import numpy as np
import xgboost as xgb
from matplotlib import pyplot
import time

data = pd.read_csv("../data/clean_dataset.csv")

train = data.loc[data.set == 'train', data.columns.values[1:]]
test = data.loc[data.set == 'test', data.columns.values[1:]]

X_train = train.drop(['Id', 'SalePrice', 'logSalePrice', 'set'], axis=1)
y_train = np.log1p(train['SalePrice'])

X_test = train.drop(['Id', 'SalePrice', 'logSalePrice', 'set'], axis=1)


# train
start_time = time.time()
print('training on defaults (-1)')
model_xgb = xgb.XGBRegressor()
model_xgb.fit(X_train, y_train)
end_time = time.time()
elapsed = end_time - start_time
print('time elapsed: ' + str(elapsed))

print('-----------')

results = []
num_threads = [1, 2, 3, 4, 5, 6, 7, 8]

for threads in num_threads:
    start_time = time.time()
    print('training on: ' + str(threads) + " threads")
    model_xgb = xgb.XGBRegressor(nthread=threads)
    model_xgb.fit(X_train, y_train)

    end_time = time.time()
    elapsed = end_time - start_time
    print('time elapsed: ' + str(elapsed))

    results.append(elapsed)

#pyplot.plot(num_threads, results)
#pyplot.show()







