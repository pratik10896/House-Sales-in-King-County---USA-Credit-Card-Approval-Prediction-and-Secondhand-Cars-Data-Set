#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


data =  pd.read_csv("D:/College/DMM/Project/Data Set 3/train.csv")


# In[5]:


data


# In[6]:


data =  data.replace(to_replace=r'[A-Za-z0-9._%+-]+@]', value='new1@dummy.com', regex=True)


# In[8]:


# regex - @ - replace 1
data


# In[11]:


# check for presense of null values
data.isnull().any()


# In[10]:


## for safety drop NaN values
data = data.dropna()


# In[12]:


## find the correlation between numeric features
## example of high correlation - pressure and temperature
# increase in pressure means decrease in temperature and vice-versa, this is negative correlation
# height and weight are cases of positive correlation
## correlation values can range from -1 to +1
# strong correlations are above 0.7 and below -0.7
data.corr()


# In[13]:


data.columns


# In[14]:


data.drop('v.id',axis =1 , inplace = True)


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[16]:


y =  data['current price']
X =  data.drop("current price",axis =1)


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[18]:


model  = RandomForestRegressor()


# In[19]:


model.fit(X_train,y_train)


# In[20]:


predictions =  model.predict(X_test)


# In[21]:


predictions


# In[22]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,predictions)


# In[23]:


X.columns


# In[24]:


mape = np.mean(np.abs(y_test - predictions)/y_test)


# In[25]:


print(mape)


# In[26]:


accuracy =  1-  mape
print(accuracy*100)


# In[29]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Create decision tree regressor object and fit to training data
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predict on test data
y_pred = regressor.predict(X_test)

# Calculate accuracy (R^2 score)
accuracy = r2_score(y_test, y_pred)
print('Accuracy of Decision Tree Regressor: {:.2f}'.format(accuracy))


# In[31]:


#Random Forest regressor using lightgbm
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import numpy as np

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'regression',
    'metric': 'l1',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data], early_stopping_rounds=10)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs(y_test - predictions) / y_test)
accuracy = 1 - mape



# In[32]:


print("Mean Absolute Error: {:.5}".format(mae))
print("Accuracy: {:.2%}".format(accuracy))


# In[33]:


#Decision Tree regressor using lightgbm
from sklearn.metrics import r2_score

# Create dataset objects
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Set hyperparameters for the model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 12,
    'num_leaves': 2 ** 12,
    'min_data_in_leaf': 8,
    'learning_rate': 0.01
}

# Train the model
model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data], early_stopping_rounds=10)

# Make predictions on the test set
y_pred = model.predict(X_test)


# In[34]:


# Calculate accuracy (R^2 score)
accuracy = r2_score(y_test, y_pred)
print('Accuracy of LightGBM Decision Tree Regressor: {:.2f}'.format(accuracy))


# In[ ]:




