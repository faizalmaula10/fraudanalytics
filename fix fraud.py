#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('fraud_oracle.csv')


# In[3]:


df.info()


# In[4]:


df=df.drop(['Month','WeekOfMonth','DayOfWeek','PolicyNumber','RepNumber','Deductible'],axis=1)


# In[5]:


nominal=df[['Make','DayOfWeekClaimed','MonthClaimed','MaritalStatus','PolicyType','VehicleCategory','BasePolicy']]


# In[6]:


# Get dummy variables
dummy = pd.get_dummies(nominal)

print(dummy)


# In[7]:


df=df.drop(nominal,axis=1)


# In[8]:


df.isnull().sum()


# In[9]:


df=pd.concat([df,dummy],axis=1)


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to object columns
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Convert boolean columns to int (True: 1, False: 0)
bool_cols = df.select_dtypes(include=['bool']).columns
for col in bool_cols:
    df[col] = df[col].astype(int)

print(df)


# In[12]:


X=df.drop(['FraudFound_P'],axis=1)


# In[13]:


y=df['FraudFound_P']


# In[15]:


import matplotlib.pyplot as plt


# In[44]:


from sklearn.preprocessing import StandardScaler

# Create an instance of StandardScaler
scaler = StandardScaler()

# Fit the scaler to your training data
scaler.fit(X)
X=scaler.transform(X)


# In[46]:


X


# In[47]:


fraud_by_year = df.groupby(['Year', 'FraudFound_P']).size().unstack(fill_value=0)

# Plotting
fraud_by_year.plot(kind='bar', stacked=True)
plt.xlabel('Year')
plt.ylabel('Total Fraud Cases')
plt.title('Total Fraud Cases by Year and Fraud Status')
plt.legend(title='Fraud Status', labels=['No Fraud', 'Fraud'])
plt.show()


# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[49]:


from sklearn.metrics import classification_report,confusion_matrix


# In[50]:


print(classification_report(y_pred,y_test))


# In[51]:


print(confusion_matrix(y_pred,y_test))


# In[52]:


from imblearn.under_sampling import RandomUnderSampler


# In[53]:


rus=RandomUnderSampler(sampling_strategy=1)
X_res,y_res=rus.fit_resample(X,y)


# In[54]:


y_res.value_counts()


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)


# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Initialize Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[57]:


print(classification_report(y_pred,y_test))


# In[58]:


print(confusion_matrix(y_pred,y_test))


# In[59]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Define your decision tree classifier
dt = DecisionTreeClassifier(random_state=0)

# Define the parameter grid for hyperparameter tuning
params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 7, 9],
    "min_samples_split": [5, 10, 15, 20, 50, 100],
    "min_samples_leaf": [5, 10, 15, 20, 50, 80, 100]
}

# Perform grid search using GridSearchCV
grid_search = RandomizedSearchCV(estimator=dt,param_distributions=params,cv=5,n_jobs=2)

# Fit the grid search object to your training data
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_dt = grid_search.best_estimator_

# Use the best estimator to make predictions on your test data
y_pred = best_dt.predict(X_test)


# In[60]:


print(classification_report(y_test,y_pred))


# In[61]:


print(confusion_matrix(y_test, y_pred))


# In[62]:


# using RandomizedsearchCV for hyperparameter tuning
params_rf={"criterion":["gini","entropy"],
          "max_depth":[9,11,13,15,17,20],
          "min_samples_split":[20,50,100,200],
          "min_samples_leaf":[2,5,20,10],
          "n_estimators":[50,100,150,200],
          "bootstrap":[True],
          "max_features":["sqrt","log2"],
          "max_samples":[.7,.75,.8,.9]}

# Perform grid search using GridSearchCV
grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(),param_distributions=params_rf,cv=5,n_jobs=2)

# Fit the grid search object to your training data
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_rf = grid_search.best_estimator_

# Use the best estimator to make predictions on your test data
y_pred = best_rf.predict(X_test)


# In[63]:


print(classification_report(y_test,y_pred))


# In[64]:


print(confusion_matrix(y_test, y_pred))


# In[65]:


pip install xgboost


# In[66]:


from xgboost import XGBClassifier


# In[67]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define the parameter grid for hyperparameter tuning
params = {
    'learning_rate': uniform(0.01, 0.3),  # Learning rate
    'max_depth': randint(3, 10),           # Maximum tree depth
    'min_child_weight': randint(1, 10),    # Minimum sum of instance weight (hessian) needed in a child
    'subsample': uniform(0.6, 0.4),        # Subsample ratio of the training instances
    'colsample_bytree': uniform(0.6, 0.4), # Subsample ratio of columns when constructing each tree
    'gamma': uniform(0, 0.5),              # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'reg_alpha': uniform(0, 0.5),          # L1 regularization term on weights
    'reg_lambda': uniform(0, 0.5)          # L2 regularization term on weights
}

# Define XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Perform randomized search using RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=params,
                                   n_iter=100, cv=5, random_state=0, n_jobs=-1)

# Fit the randomized search object to your training data
random_search.fit(X_train, y_train)

# Get the best estimator from the randomized search
best_xgb = random_search.best_estimator_

# Use the best estimator to make predictions on your test data
y_pred = best_xgb.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)


# In[68]:


print(classification_report(y_test,y_pred))


# In[ ]:





# In[69]:


from imblearn.over_sampling import SMOTE


# In[70]:


smote = SMOTE(random_state=42)

# Fit and resample the dataset
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)


# In[73]:


y_resampled.value_counts()


# In[74]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Initialize Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[75]:


conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)


# In[76]:


print(classification_report(y_test,y_pred))


# In[77]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Define your decision tree classifier
dt = DecisionTreeClassifier(random_state=0)

# Define the parameter grid for hyperparameter tuning
params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 7, 9],
    "min_samples_split": [5, 10, 15, 20, 50, 100],
    "min_samples_leaf": [5, 10, 15, 20, 50, 80, 100]
}

# Perform grid search using GridSearchCV
grid_search = RandomizedSearchCV(estimator=dt,param_distributions=params,cv=5,n_jobs=2)

# Fit the grid search object to your training data
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_dt = grid_search.best_estimator_

# Use the best estimator to make predictions on your test data
y_pred = best_dt.predict(X_test)


# In[78]:


conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)


# In[79]:


print(classification_report(y_test,y_pred))


# In[80]:


# using RandomizedsearchCV for hyperparameter tuning
params_rf={"criterion":["gini","entropy"],
          "max_depth":[9,11,13,15,17,20],
          "min_samples_split":[20,50,100,200],
          "min_samples_leaf":[2,5,20,10],
          "n_estimators":[50,100,150,200],
          "bootstrap":[True],
          "max_features":["sqrt","log2"],
          "max_samples":[.7,.75,.8,.9]}

# Perform grid search using GridSearchCV
grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(),param_distributions=params_rf,cv=5,n_jobs=2)

# Fit the grid search object to your training data
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_rf = grid_search.best_estimator_

# Use the best estimator to make predictions on your test data
y_pred = best_rf.predict(X_test)


# In[81]:


conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)


# In[82]:


print(classification_report(y_test,y_pred))


# In[83]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define the parameter grid for hyperparameter tuning
params = {
    'learning_rate': uniform(0.01, 0.3),  # Learning rate
    'max_depth': randint(3, 10),           # Maximum tree depth
    'min_child_weight': randint(1, 10),    # Minimum sum of instance weight (hessian) needed in a child
    'subsample': uniform(0.6, 0.4),        # Subsample ratio of the training instances
    'colsample_bytree': uniform(0.6, 0.4), # Subsample ratio of columns when constructing each tree
    'gamma': uniform(0, 0.5),              # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'reg_alpha': uniform(0, 0.5),          # L1 regularization term on weights
    'reg_lambda': uniform(0, 0.5)          # L2 regularization term on weights
}

# Define XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Perform randomized search using RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=params,
                                   n_iter=100, cv=5, random_state=0, n_jobs=-1)

# Fit the randomized search object to your training data
random_search.fit(X_train, y_train)

# Get the best estimator from the randomized search
best_xgb = random_search.best_estimator_

# Use the best estimator to make predictions on your test data
y_pred = best_xgb.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)


# In[84]:


print(classification_report(y_test,y_pred))


# In[ ]:




