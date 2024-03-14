#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df = pd.read_csv("D:/College/DMM/Project/Data Set 1/kc_house_data.csv")
df


# In[9]:


df.describe()


# In[10]:


df.isna().sum()


# In[11]:


df.info()


# In[12]:


df1 = df.drop(['id','date'], axis=1)
df1.head()


# In[13]:


# Correlation:
# We can demonstrate that all variables are in good correlation with ‘price’. Only ‘zipcode’ has a negative correlation of -0.05 
# but are very near to 0 with the target variable.
# ‘sqft_living’, ‘grades’ and ‘bathrooms’ are having a positive strong correlation with the target variable ‘price’.

corr = df1.corr()
plt.figure(figsize=(25,15))
sns.heatmap(corr, annot=True)


# In[14]:


# Splitting data into train and test
# We used train_test_split from sklearn library to split our data into 75% and 25% for train and test sets respectively. 
# We created x_train, x_test, y_train and y_test. The Random state for train and test is 3.

from sklearn.model_selection import train_test_split


# In[15]:


x = df1.drop(['price'], axis=1)
y = df1['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)


# In[17]:


# Visualization:
# 1st Plot: Shows the bedrooms count, and it can be observed that most of the properties are having 3 bedrooms and 4 bedrooms.

plt.subplots(figsize=(7, 5))
sns.countplot(df1["bedrooms"])
plt.show()


# In[18]:


# 2nd Plot: Shows the bathroom count, and it can be observed that most of the houses are having 2.5, 1, and 1.75 bathrooms.

plt.subplots(figsize=(15, 5))
sns.countplot(df1["bathrooms"])
plt.show()


# In[19]:


# 3rd Plot: Shows property with waterfront and 
# we can observe that the maximum of the houses is not having a waterfront and only a few have a waterfront feature.

sns.countplot(df1["waterfront"])
plt.show()


# In[20]:


# 4th Plot: 
# Shows how many floors maximum properties have, and we can observe that most of the properties are having 1 and 2 floors.

sns.countplot(df1["floors"])
plt.show()


# In[25]:


# Machine Learning models:
#     Decision Tree using sklearn:

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV


# In[26]:


dtree_up = DecisionTreeRegressor()
dtree_up.fit(x_train, y_train)               # Fitting model with x_train and y_train
dtree_pred_up = dtree_up.predict(x_test)     # Predicting the results
print('RMSE:', np.sqrt(mean_squared_error(y_test, dtree_pred_up, squared=False)))
print('r2 score: %.2f' % r2_score(y_test, dtree_pred_up))
print("Accuracy :",dtree_up.score(x_test, y_test))


# In[28]:


# HyperParameter Tuned Decision Tree Regressor:

d = np.arange(1, 21, 1)

dtree = DecisionTreeRegressor(random_state=5)
hyperParam = [{'max_depth':d}]

gsv = GridSearchCV(dtree,hyperParam,cv=5,verbose=1)
best_model = gsv.fit(x_train, y_train)                          # Fitting model with xtrain_scaler and y_train
dtree_pred_mms = best_model.best_estimator_.predict(x_test)     # Predicting the results

print("Best HyperParameter: ",gsv.best_params_)

print('RMSE:', np.sqrt(mean_squared_error(y_test, dtree_pred_mms, squared=False)))
print('r2 score: %.2f' % r2_score(y_test, dtree_pred_mms))
print("Accuracy :",best_model.score(x_test, y_test))


# In[29]:


labels = {'True Labels': y_test, 'Predicted Labels': dtree_pred_mms}
df_lm = pd.DataFrame(data = labels)
sns.lmplot(x='True Labels', y= 'Predicted Labels', data = df_lm)


# In[30]:


# Random Forest using sklearn:

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# In[31]:


# Simple Random Forest

rf = RandomForestRegressor()
rf.fit(x_train, y_train)             # Fitting model with x_train and y_train
rf_pred = rf.predict(x_test)         # Predicting the results
print('RMSE:', np.sqrt(mean_squared_error(y_test, rf_pred, squared=False)))
print('r2 score: %.2f' % r2_score(y_test, rf_pred))
print("Accuracy :",rf.score(x_test, y_test))


# In[32]:


# HyperParameter Tuned Random Forest Regressor:

nEstimator = [140,160,180,200,220]
depth = [10,15,20,25,30]

RF = RandomForestRegressor()
hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]

gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
gsv.fit(x_train, y_train)

print("Best HyperParameter: ",gsv.best_params_)
scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))
maxDepth=gsv.best_params_['max_depth']
nEstimators=gsv.best_params_['n_estimators']

model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)
model.fit(x_train, y_train)        # Fitting model with x_train and y_train

# Predicting the results:
rf_pred_tune = model.predict(x_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, rf_pred_tune, squared=False)))
print('r2 score: %.2f' % r2_score(y_test, rf_pred_tune))
print("Accuracy :",model.score(x_test, y_test))


# In[33]:


labels = {'True Labels': y_test, 'Predicted Labels': rf_pred_tune}
df_lm = pd.DataFrame(data = labels)
sns.lmplot(x='True Labels', y= 'Predicted Labels', data = df_lm)


# In[35]:


#Decision Tree by Lightgbm

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Creating LGBM dataset for training and testing
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)

# Setting parameters for LightGBM model
params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1}

# Training LightGBM model
num_rounds = 100
lgb_model = lgb.train(params, train_data, num_rounds, valid_sets=[test_data], early_stopping_rounds=10)

# Making predictions on test data
lgb_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration)


# In[36]:


# Evaluating model performance
print('RMSE:', mean_squared_error(y_test, lgb_pred, squared=False))
print('Accuracy:', r2_score(y_test, lgb_pred))


# In[37]:


#Random Forrest by Lightgbm

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Create lightgbm dataset
train_data = lgb.Dataset(x_train, label=y_train)

# Set parameters for the model
params = {
    'boosting_type': 'rf',
    'objective': 'regression',
    'num_leaves': 31,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'bagging_freq': 1,
    'num_threads': 4,
    'metric': 'rmse',
    'verbose': -1
}

# Train the model
model = lgb.train(params, train_data, num_boost_round=100)

# Predict on test data
lgb_pred = model.predict(x_test)


# In[39]:


# Calculate accuracy (RMSE and R^2 score)
rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
r2 = r2_score(y_test, lgb_pred)
print('RMSE:', rmse)
print('Accuracy:', r2)


# In[ ]:





# In[ ]:




