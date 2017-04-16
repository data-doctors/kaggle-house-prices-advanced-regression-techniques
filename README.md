# kaggle-house-prices-advanced-regression-techniques
Repository for source code of kaggle competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Overview

## Data

`data` folder contains original data

## Repository structure

## Training

### training all models (bulk training)

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

### saving submision of single model

```
$ python models/single/model_xgb.py save
```

### Scores

Single models:
| Model                      | CV            | LB    |
| -------------              |:-------------:| -----:|
| XGBoost                    | right-aligned | $1600 |
| LightGBM                   | right-aligned | $1600 |
| Random Forest              | centered      |   $12 |
| Extra Trees                | centered      |   $12 |
| Lasso Regression           | are neat      |    $1 |
| Ridge Regression           | are neat      |    $1 |
| ElasticNet Regression      | are neat      |    $1 |

Ensembles:


## Team
- [Obaidur Rahaman](https://github.com/obaidur-rahaman)
- [Marco Di Vivo](https://github.com/divivoma)
- [Benjamin Melloul]()
- [Ayush Kumar](https://github.com/swifty1)
- [Robert Jonczy](https://github.com/rjonczy)

