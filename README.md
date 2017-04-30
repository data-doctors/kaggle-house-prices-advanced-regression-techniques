# kaggle-house-prices-advanced-regression-techniques
Repository for source code of kaggle competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Overview

## Data

`data` folder contains original data

## Repository structure

## Training

### Training all models (bulk training)

In order to train all models
```
$ ./run_all.sh
RMSE-xgb-CV(7)=0.15017262592+-0.0403780999289
RMSE-lgb-CV(7)=0.230416102431+-0.0982360336472
RMSE-rf-CV(7)=0.178752572944+-0.0495588133233
RMSE-et-CV(7)=0.177138296419+-0.0523244324721
RMSE-lasso-CV(7)=0.167043833535+-0.0590946122368
RMSE-ridge-CV(7)=0.16305872566+-0.0592719750453
RMSE-elasticnet-CV(7)=0.166431639245+-0.0591651827043
```

### Saving submision of single model

```
$ python models/single/model_xgb.py save
```

### Scores

Best single models:

| Model                      | CV               | LB      |
| :------------------------- |:----------------:| :-------|
| DecisionTreeRegressor      | 0.19013+-0.01304 | 0.18804 |
| RandomForestRegressor      | 0.14744+-0.00871 | 0.14623 |
| ExtraTreesRegressor        | 0.13888+-0.01208 | 0.15194 |
| XGBoost                    | 0.12137+-0.01128 | 0.12317 |
| LightGBM                   | 0.20030+-0.01182 | 0.21416 |
| Lasso                      | 0.11525+-0.01191 | 0.12091 |
| Ridge                      | 0.11748+-0.01170 | 0.12263 |
| ElasticNet                 | 0.11364+-0.01677 | 0.11976 |
| SVM                        | 0.19752+-0.01386 | 0.20416 |

Ensembles:


## Team
- [Obaidur Rahaman](https://github.com/obaidur-rahaman)
- [Marco Di Vivo](https://github.com/divivoma)
- [Benjamin Melloul]()
- [Ayush Kumar](https://github.com/swifty1)
- [Robert Jonczy](https://github.com/rjonczy)

