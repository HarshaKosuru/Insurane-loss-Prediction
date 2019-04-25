#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


# In[6]:


#reading csv file
insurance= pd.read_csv('insurance.csv')
insurance.replace('-999.25','0.0',inplace=True)
columns=('age','bmi','children','smoker','charges')
insurance1=pd.DataFrame(insurance,columns=columns)


# In[7]:


#generate data
X=np.array(insurance1.drop(['age'],1))
y=np.array(insurance1['age'])

X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:800], y[600:800]
X_train_valid, y_train_valid = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]


# In[10]:


# Train random forest classifier on all train and validation
# test data: evaluate

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)
print("loss= ",score,"%")

