import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import gridspec
from scipy.stats import skew
#from scipy.stats.stats import pearson
#from scipy import stats
import xgboost as xgb
from scipy.stats import norm
#from pyglmnet import GLM # Marco: need to understand how to install this 
#from sklearn.preprocessing import StandardScaler

#from pandas import ExcelWriter  #Marco: this is useful to write in excel

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

#Distribution and skewness
def distskew(dataset,feature):
    fig = plt.figure()
    sns.distplot(dataset[feature], fit=norm);
    return("Skewness = ",skew(dataset[feature].dropna()))

# Scatter plot
def scatplot(a,b):
    scatplotdata = pd.DataFrame({"x":a, "y":b})
    scatplotdata.plot(x = "x", y = "y", kind = "scatter")
    return()

#Remove outliers from the training set only
#in this way we will not alter the number of the test example !!

train = train.drop(train[train['GrLivArea'] > 4000].index)
train = train.drop(train[train['TotalBsmtSF']> 6000].index)

# remove outlier for LotFrontage and mean imputation
train = train.drop(train[train['LotFrontage']>150].index)
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())

# Removing outlier for LotArea and transforming
train = train.drop(train[train['LotArea']>50001].index)


#ALL_DATA: let's build an unique dataframe by contatenation of train and test


all_data = pd.concat((train.loc[:,'Id':'SaleCondition'],
                      test.loc[:,'Id':'SaleCondition']))
print("Data shapes:")
print("train:",train.shape)
print("test:",test.shape)
print("all_data:",all_data.shape)

#get the index of the numerical features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#Get Categorical data in a fast way:
categorical_features = pd.DataFrame(all_data.describe(include = ['O'])).columns


#prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
#prices.hist()
                                   
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])



#DATA CLEANING:

REMOVING_THRESH = 0.8

total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(10))
#MDV: remove the feature that has more than 10% of missing values
print("We are going to remove the below features : ")
print(missing_data[missing_data['Percent'] > REMOVING_THRESH])

all_data = all_data.drop((missing_data[missing_data['Percent'] > REMOVING_THRESH]).index,1)

print("After removed data the new dim of all_data are")
print(all_data.shape)


#FEATURE ENGINEERING:
#Marco: replace the TotalBsmtSF = 0 with the relative GrLivArea value of that house.
#improvment from skewness of -5.1 to 2.1

no_basement_houses_index = train[train['TotalBsmtSF']== 0].index
train['TotalBsmtSF'].loc[no_basement_houses_index] = train['GrLivArea'].loc[no_basement_houses_index]
#print(train['TotalBsmtSF'].loc[no_basement_houses_index]) #Debug

#scatplot(np.log1p(train["TotalBsmtSF"]),train["SalePrice"])#debug
#distskew(train,"TotalBsmtSF")#debug


# Add the living areas and basement aread to create a new feature TotArea
all_data["TotArea"] = all_data["GrLivArea"] + all_data["TotalBsmtSF"]
train["TotArea"] = train["GrLivArea"] + train["TotalBsmtSF"]


#Marco 08052017 some new features:
 

all_data["HasFirePlace"] =all_data["Fireplaces"]>0 
all_data["HasWoodDeck "] =all_data["WoodDeckSF"]>0


#***** Ayouk Feature Engineering -Log transformation :********
all_data['LotArea']=np.log1p(all_data['LotArea'])


#***** Obaidur Feature Engineering:*************************
    
all_data["TotBsmtFin"] = all_data["BsmtFinSF1"] + all_data["BsmtFinSF2"]
train["TotBsmtFin"] = train["BsmtFinSF1"] + train["BsmtFinSF2"]
                 
all_data = all_data.drop("BsmtFinSF1",1)
all_data = all_data.drop("BsmtFinSF2",1)

all_data["TotBath"] = all_data["FullBath"] + 0.5*all_data["HalfBath"] + all_data["BsmtFullBath"] + 0.5*all_data["BsmtHalfBath"]
train["TotBath"] = train["FullBath"] + 0.5*train["HalfBath"] + train["BsmtFullBath"] + 0.5*train["BsmtHalfBath"]

all_data = all_data.drop("FullBath",1)
all_data = all_data.drop("HalfBath",1)
all_data = all_data.drop("BsmtFullBath",1)
all_data = all_data.drop("BsmtHalfBath",1)
#print("all_data dim:",all_data.shape)#debug



#SKEWENESS ANALYSIS:
#Refresh the index of the numerical features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#Uncomment to calculate the skewness for the train data 

#skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

#Marco: here i've changed the train[numeric_feats] with all_data[numeric_feats]
#because we are working on all_data and train is just a part of it.

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

#exctract the features with skewness higher than 75%
skewed_feats = skewed_feats[skewed_feats > 0.1]
skewed_feats_idx = skewed_feats.index

#log transform skewed numeric features with skewness > 10%:
#trying to have more "normal" feats (less skewed)

all_data[skewed_feats_idx] = np.log1p(all_data[skewed_feats_idx])

train[skewed_feats_idx] = np.log1p(train[skewed_feats_idx])

#Marco: check skewness improvement:
    
log_skewed_feats = all_data[skewed_feats_idx].apply(lambda x: skew(x.dropna())) #compute skewness
df_skew = pd.DataFrame({"log":np.abs(log_skewed_feats),"origin":np.abs(skewed_feats)})

df_skew["diff"] = (df_skew.origin) - (df_skew.log) #if negative then there is no improvment

print("Check Skewness after log:")
 #more improvment can be done here by checking different transoframation 
 #for negative resulte of diff
print(df_skew)  


#DROP SOME FEATURES:
   
all_data = all_data.drop("BsmtFinType1",1)
all_data = all_data.drop("2ndFlrSF",1)
all_data = all_data.drop("BedroomAbvGr",1)

all_data = all_data.drop("LowQualFinSF",1)
all_data = all_data.drop("3SsnPorch",1)
all_data = all_data.drop("PoolArea",1)

all_data = all_data.drop('Condition2',1)

#print("all_data dim: ",all_data.shape')#debug
#print("\nX_test dim: ",X_test.shape)#debug

#One Hot Encoding: Get dummies:
dummies = pd.get_dummies(all_data)
all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
    #median is less prone to outliers than the mean
all_data = all_data.fillna(all_data.median())


#Let's create the X_train and X_test matrix:
    # Normalize the features (this does not seem to help: increases error)
#all_data = all_data.apply(lambda x: x/np.sqrt(sum(x**2)))

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
Y_train = train.SalePrice #  remember that it is already log transformed
X_train['SalePrice'] = train.SalePrice

# save 
X_train.to_csv("../data/X_train_v2.csv")
X_test.to_csv("../data/X_test_v2.csv")

X_train = X_train.drop("SalePrice",1)
#print("\nShape check\n train.shape: ",train.shape)
#print(" X_Train dim: ",X_train.shape)
#print("\n X_test dim: ",X_test.shape)

#Correlation matrix analysis:
    
#X_check=X_train.assign(SalePrice=train.SalePrice, index=X_train.index)
#corrmat = X_check.corr()
##Correlation matrix with Log of Sale Price
#k = 10 #number of variables for heatmap
#cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
###cols = corrmat.nsmallest(k, 'SalePrice')['SalePrice'].index
#
#cm = np.corrcoef(X_check[cols].values.T)
#sns.set(font_scale=1.25)
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


#EVALUATE SOME MODEL PERFORMANCES:


#first evaluation of the regular linear regression method:

#RIDGE (L2 NORM):
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
           for alpha in alphas]
    
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
#Marco: added log scale to have a better view of the minimum
plt.xscale("log")
plt.show()
print("The min value of Ridge is ",cv_ridge.min())



#Choose the best alpha by taking the alpha that give the lowest rmse
best_alpha = 8
# Now fit Ridge model
model_ridge = Ridge(alpha = best_alpha).fit(X_train, Y_train)

#LASSO (L1 NORM):
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, Y_train)
print("The min value of Lasso is ",rmse_cv(model_lasso).mean())


#Lasso choose performs also feature selection 

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

#Let's look to the most important coefficients:
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")  
plt.show()  


#LASSO RESIDUAL ANALYSIS: 
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds_log = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":Y_train})
#let's return to "original" value from the log:
preds_val = np.expm1(preds_log)

#solution = pd.DataFrame({"id":test.Id, "SalePrice":preds_val})

#preds_log["residuals"] = preds_log["true"] - preds_log["preds"]
#
#preds_log.plot(x = "preds", y = "true",kind = "scatter")
#plt.title('Diff between true and predicted values - Log Value ')
#plt.show()

preds_val["residuals"] = preds_val["true"] - preds_val["preds"]
preds_val["abs_residuals"]=np.abs(preds_val["residuals"])
preds_val.plot(x = "preds", y = "residuals",kind = "scatter")
plt.title('Diff between true and predicted values -  Value in $')
plt.show()

preds_val.sort_values(ascending=False,inplace=True,by="abs_residuals")
print("This are the houses ordered by absolute error on sale price prediction with lasso model:")
print(preds_val.head(10))


#XGBOOST MODEL:
    
#Let's add an xgboost model to our linear model to see 
#if we can improve our score:

#y = preds_log["residuals"]
    
dtrain = xgb.DMatrix(X_train, label = Y_train)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
plt.show()

#the params were tuned using xgb.cv
model_xgb = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.1) 
model_xgb.fit(X_train, Y_train)






xgb_preds = np.expm1(model_xgb.predict(X_test))
ridge_preds = np.expm1(model_ridge.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))


predictions_lasso = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
#predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
#plt.show()

predictions_ridge = pd.DataFrame({"xgb":xgb_preds, "ridge":ridge_preds})

lasso_vs_ridge = pd.DataFrame({"lasso":lasso_preds, "ridge":ridge_preds})

fig = plt.figure()


fig.suptitle("Prediction check", fontsize=16)
ax = plt.subplot("131")
ax.set_title("XGB VS Lasso")
ax.scatter(xgb_preds,lasso_preds)

#axes = plt.gca()
#ax.set_xlim([xmin,xmax])
#axes.set_ylim([ymin,ymax])

ax = plt.subplot("132")
ax.set_title("XGB VS Ridge")
ax.scatter(xgb_preds,ridge_preds)
ax.autoscale(tight=True)

ax = plt.subplot("133")
ax.set_title("Lasso VS Ridge")
ax.scatter(lasso_preds,ridge_preds,alpha=.1)

plt.show()


#SMALL ENSEMBLING OF THE MODELS:

#preds = 0.30*lasso_preds + 0.30*ridge_preds + 0.40*xgb_preds
preds = 0.60*lasso_preds + 0.40*xgb_preds
#preds = lasso_preds
#preds = ridge_preds
#preds= xgb_preds

solution = pd.DataFrame({"SalePrice":preds,"id":test.Id})

#N.B. USE YOUR RELATIVE PATH HERE FOR THE CORRECT OUTPUT:
#Obaidur:
#solution.to_csv("D:/acads/ds/kaggle/house-price/my_solution_eight.csv", index = False)
#Marco
solution.to_csv("..//..//LocalSubmission//XGB_and_Lasso_130517.csv", index = False)
#Ayush:
#solution.to_csv("C:/Users/Aoos/Documents/Projects/HousePrice/Solution/solution1.csv", index = False)


#SCORE HISTORY:
#Marco: 13/05/2017 with preds = 0.60*lasso_preds + 0.40*xgb_preds --> LB 0.12206


