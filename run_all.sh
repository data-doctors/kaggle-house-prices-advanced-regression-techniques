#!/bin/sh

# base/single models

# trees algorithms
python models/single/model_dt.py
python models/single/model_rf.py
python models/single/model_et.py
python models/single/model_xgb.py
python models/single/model_lgb.py


# linear algorithms
python models/single/model_lasso.py
python models/single/model_ridge.py
python models/single/model_elasticnet.py

# stacking
# python models/ensembles/model_stack_l1.py
# python model_stack_l1.py# python models/ensembles/model_stack_l1.py