#!/bin/sh

# base/single models

# trees algorithms
python models/single/model_dt.py $1
python models/single/model_rf.py $1
python models/single/model_et.py $1
python models/single/model_xgb.py $1
python models/single/model_lgb.py $1

# linear algorithms
python models/single/model_lasso.py $1
python models/single/model_ridge.py $1
python models/single/model_elasticnet.py $1

# non-linear algorithm
python models/single/model_svm.py $1

# stacking
# python models/ensembles/model_stack_l1.py
# python model_stack_l1.py# python models/ensembles/model_stack_l1.py