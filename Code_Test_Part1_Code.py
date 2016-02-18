
# coding: utf-8

# #Importing Required Packages 

# In[127]:

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,LassoCV
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.decomposition import PCA,KernelPCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from __future__ import division
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')
import xgboost
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.grid_search import GridSearchCV


# # Loading Data

# In[10]:

data_train = pd.read_table(r'E:\AV\c1_data_science_challenge\codetest_train.txt')

# In[5]:

data_train.head()

# List Percentage of missing values 
percent_of_missing_train  = data_train.apply(lambda x :  (float(sum(x.isnull()))/len(x))*100 )
print "Train data has "+str(len(percent_of_missing_train[percent_of_missing_train > 5]))+" columsn with more than 5% missing"


# In[9]:

#Target in train has no missing values
sum(data_train['target'].isnull())

#Storing Train Categorical column names
categorical_col_train = []
for x in data_train.columns :
    if data_train[x].dtype == 'object' :
        categorical_col_train.append(x)


train_numerical = data_train.drop(categorical_col_train+['target'],axis=1)
train_categorical = data_train[categorical_col_train]
train_target = data_train['target']
train_panel = pd.concat([train_numerical,train_categorical],axis=1)

#Imputing missing values with mean for numerical columns and mode for categorical columns
for col in train_panel.columns :
    if train_panel[col].dtype == 'object' :
        most_occured =  train_panel[col].value_counts().index[0]
        train_panel[col].fillna(most_occured,inplace=True)
    else:        
        train_panel[col].fillna(train_panel[col].mean(),inplace=True)


# removing constant columns
for colname in train_panel.columns:
    if len(np.unique(train_panel[colname].values.astype("str"))) == 1:
        del train_panel[colname]
        print("Column %s has zero variance and is removed from data" % (colname))


# In[20]:

#Creating dummy variables
train_panel = pd.get_dummies(train_panel, columns=categorical_col_train)

#Multiplying the CV score with -1 to make it positive
#Github comments for cross_val_score suggests this is know bug 

# In[44]:

#fitting Linear Model
LR = LinearRegression(n_jobs=-1)
LR_Cross_val = cross_val_score(LR,train_panel,train_target,cv=10,scoring = 'mean_squared_error').mean()
print "CV Score for Linear Regression : "+str(-1*LR_Cross_val)
#MSE = 12.27

#Fitting RandomeForest
rf = RandomForestRegressor(n_estimators = 200,n_jobs=-1)
RF_cross_val = cross_val_score(rf,train_panel,train_target,cv=10,scoring = 'mean_squared_error').mean()
print "CV Score for RandomForest  : "+str(-1*RF_cross_val)
#MSE = 11.81


#Tunning Alpha for Lasso 
for i in np.arange(0.01,0.5,0.05) :
    las = Lasso(alpha=i)
    print "Aplha ="+str(i)+"    CV: "+str(-1*cross_val_score(las,train_panel,train_target,cv=10,scoring = 'mean_squared_error').mean())


#Fitting Lasso
las = Lasso(alpha=0.06)
Las_CV = cross_val_score(las,train_panel,train_target,cv=10,scoring = 'mean_squared_error').mean()
print "CV Score for Lasso  : "+str(-1*Las_CV)
#MSE = 11.63


#Performing PCA
pca = PCA(n_components='mle')
pca.fit(train_panel)

#Explained Varinace ratio
NO_cols = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_ > 0.05])
print str(NO_cols)+" columsn have variance ration greater than 0.05"


#Feature selectiong using Lasso
Lass = Lasso(alpha = 0.1)
Lass = Lass.fit(train_panel,train_target)
model_selecting = SelectFromModel(Lass, prefit=True)
features_selected = train_panel.columns[model_selecting.get_support()]
train_features_subset = model_selecting.transform(train_panel)

print str(train_features_subset.shape[1])+" columns selected"

#After Feature Selection

LR_Cross_val = cross_val_score(LR,train_features_subset,train_target,cv=10,scoring = 'mean_squared_error').mean()
print "CV Score for Linear Regression : "+str(-1*LR_Cross_val)




#Tunning Alpha for Lasso 
for i in np.arange(0.01,0.5,0.05) :
    las = Lasso(alpha=i)
    print "Aplha ="+str(i)+"    CV: "+str(-1*cross_val_score(las,train_features_subset,train_target,cv=10,scoring = 'mean_squared_error').mean())

las = Lasso(alpha=0.06)
Las_CV = cross_val_score(las,train_features_subset,train_target,cv=10,scoring = 'mean_squared_error').mean()

print "CV Score for Lasso  : "+str(-1*Las_CV)
#MSE = 11.5


#RandomeForest
rf = RandomForestRegressor(n_estimators = 500,n_jobs=-1,min_samples_split=2,max_features=0.99)
rf_cv = cross_val_score(rf,train_features_subset,train_target,cv=10,verbose=0,scoring = 'mean_squared_error').mean() #-9.0946247212837523
print "CV Score for RF  : "+str(-1*rf_cv)
#MSE = 9.08


#Fitting ExtraTrees
ET = ExtraTreesRegressor(n_estimators = 500,n_jobs=-1,min_samples_split=2,max_features=0.99)
ET_cross_val = cross_val_score(rf,train_features_subset,train_target,cv=10,scoring = 'mean_squared_error').mean()
print "CV Score for ExtraTrees : "+str(-1*ET_cross_val)
#MSE = 9.07


#Fitting XgBoost
parameters = {
        'n_estimators': [100, 250, 500],
        'learning_rate': [0.05, 0.1, 0.3],
        'max_depth': [6, 9],
        'subsample': [0.7,0.8,0.9],
        'colsample_bytree': [0.7,0.8,0.9],
        'gamma':[0.1,0.4,0.6]
    }


xgb_model = xgboost.XGBRegressor(param)
clf = GridSearchCV(xgb_model, parameters, n_jobs= 3, cv=10,scoring ="mean_squared_error")

clf.fit(train_features_subset,train_target)

score = []
par = {'n_estimators':[],'learning_rate':[],
       'max_depth':[],'subsample':[],
       'colsample_bytree':[],'gamma':[],'scores':[],'min_child_weight':[]
      }

for i in clf.grid_scores_:
    par['scores'].append(i[1])
    for k in i[0].keys() :
        par[k].append( i[0][k])

pd.DataFrame(par).to_csv(r'E:\AV\c1_data_science_challenge\xgboost_tune_parameters.csv')


# In[129]:

param_tune = {'max_depth':7, 'learning_rate':0.05 ,'colsample_bytree':0.8,'min_child_weight' : 5 ,'subsample' : 0.8}
param_tune['objective'] = 'reg:linear'


X_train,X_test,y_train,y_test = train_test_split(train_features_subset,train_target,test_size = 0.4)

num_rounds = 500
model = xgboost.train(param_tune, xgboost.DMatrix(X_train,y_train), num_rounds)

xgb_pred = model.predict(xgboost.DMatrix(X_test),ntree_limit=model.best_iteration)

print mean_squared_error(xgb_pred,y_test)

#MSE = 4.9

# In[103]:

#Performing same operation for Test Dataset
data_test = pd.read_table(r'E:\AV\c1_data_science_challenge\codetest_test.txt')

percent_of_missing_test  = data_test.apply(lambda x :  (float(sum(x.isnull()))/len(x))*100 )
print "Test data has "+str(len(percent_of_missing_test[percent_of_missing_test > 5]))+" columsn with more than 5% missing"

#Storing test Categorical column names
categorical_col_test = []
for x in data_test.columns :
    if data_test[x].dtype == 'object' :
        categorical_col_test.append(x)


test_numerical = data_test.drop(categorical_col_test,axis=1)
test_categorical = data_test[categorical_col_test]

test_panel = pd.concat([test_numerical,test_categorical],axis=1)


#Imputing missing values with mean for numerical columns and mode for categorical columns
for col in test_panel.columns :
    if test_panel[col].dtype == 'object' :
        most_occured =  test_panel[col].value_counts().index[0]
        test_panel[col].fillna(most_occured,inplace=True)
    else:        
        test_panel[col].fillna(test_panel[col].mean(),inplace=True)


# removing constant columns
for colname in test_panel.columns:
    if len(np.unique(test_panel[colname].values.astype("str"))) == 1:
        del test_panel[colname]
        print("Column %s has zero variance and is removed from data" % (colname))

#Creating dummy variables
test_panel = pd.get_dummies(test_panel, columns=categorical_col_test)

#Subsetting the features
test_features_subset = test_panel[features_selected]


# ##Prediction of Model

# In[123]:

#training XGBoost with tuned paramters
param_tune = {'max_depth':7, 'learning_rate':0.05 ,'colsample_bytree':0.8,'min_child_weight' : 5 ,'subsample' : 0.8}
param_tune['objective'] = 'reg:linear'

num_rounds = 1000
model = xgboost.train(param_tune, xgboost.DMatrix(train_features_subset,train_target), num_rounds)

xgb_pred = model.predict(xgboost.DMatrix(test_features_subset),ntree_limit=model.best_iteration)

pd.DataFrame(xgb_pred).to_csv(r'E:\AV\c1_data_science_challenge\codetest_prediction.txt',header=False,index=False)

