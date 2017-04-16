#!/bin/sh

# base/single models
python models/single/model_xgb.py
python models/single/model_lgb.py
python models/single/model_rf.py
python models/single/model_et.py
python models/single/model_lasso.py
python models/single/model_ridge.py
python models/single/model_elasticnet.py

# stacking
# python models/ensembles/model_stack_l1.py
# python model_stack_l1.py# python models/ensembles/model_stack_l1.py