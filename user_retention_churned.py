#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

equity = pd.read_csv('/datasets/equity_value_data.csv')
features = pd.read_csv('/datasets/features_data.csv')


# ## Problem (a)

# Group by user_id and firstly count the number of date for each.
equity_gb = equity.groupby('user_id', as_index=False)
eq_sat = equity_gb.count()


# Choose the user_id which its date series is more than 28 days.
selected = eq_sat[eq_sat['timestamp'] >=28].reset_index()
selected_id = selected['user_id']


# Check whether a user is churned or not.
churned_user = []
for index in range(len(selected_id)):
    id = selected_id[index]
    match = equity[equity['user_id']==id]
    count = 0;
    for m in match['close_equity']:
        if m < 100:
            count += 1
        else:
            count = 0
    if count >= 28:
        churned_user.append(id)


# Compute the percentage of churned user.
percentage_of_churned = len(churned_user) * 100 / eq_sat.shape[0]

# Conclusion: About 20.827% of users have churned in the dataset provided.


# ## Problem (b)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as sm
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Classify the churned user as 1 and un-churned user as 0 and edit into `features` dataframe.
churned_col = []
for index in range(len(features['user_id'])):
    if features['user_id'][index] in churned_user:
        churned_col.append(1)
    else:
        churned_col.append(0)
        
features['churned'] = churned_col


# `cat_features` are categorical values in the dataframe and need to be encoded.
cat_features = ['risk_tolerance', 'investment_experience', 'liquidity_needs', 'platform',
                'instrument_type_first_traded', 'time_horizon']
X = pd.get_dummies(features, columns=cat_features, drop_first=True)


# `time_spent` and `first_deposit_amount` are numerical values and need to be scaled.
sc = MinMaxScaler()
a = sc.fit_transform(features[['time_spent']])
b = sc.fit_transform(features[['first_deposit_amount']])
X['time_spent'] = a
X['first_deposit_amount'] = b


# Store `user_id` and `churned` in a separate dataframe and other properties as another dataframe
all_id = X['user_id']
all_ch = X['churned']
all_other = X.drop(labels=['user_id','churned'], axis=1)


# Plot the histogram of number of un-churned(0) and churned(1) users.
sns.countplot('churned', data=features).set_title('Churned Distribution Before Resampling')


# But we can discover that these two variables are not evenly distributed. If such circumstance happens, the model will not train properly because it will give more weight with the label with more numbers. Therefore, I need to resample to make two labels evenly distributed as below.
X_no = X[X.churned == 0]
X_yes = X[X.churned == 1]

X_yes_new = X_yes.sample(n=len(X_no), replace=True, random_state=42)
X_new = X_no.append(X_yes_new).reset_index(drop=True)
sns.countplot('churned', data=X_new).set_title('Class Distribution After Resampling')


# Separate the X and y as the properties's value and churned's value and generate train & test dataframe.
X = X_new.drop(['user_id', 'churned'], axis=1)
y = X_new['churned'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In this project, after trying several classifiers such as logistics, KNN, and decision tree, I finally choose to user random forest classifier. Firstly setup the parameters for this classifier and see what the result is.
clf_forest = RandomForestClassifier(n_estimators=100, max_depth=20)
clf_forest.fit(X_train, y_train)


# The predicted `churned` label of the training set is stored as `pred`, and the training accuracy approximately 97%.
pred = clf_forest.predict(X_train)
accuracy_score(y_train, pred)


# The predicted `churned` label of the test set is stored as `pred_test`, and the test accuracy is about 87%.
pred_test = clf_forest.predict(X_test)
accuracy_score(y_test, pred_test)


# Both accuracies show that this classifer works good now, but we still need to find the parameters of this classifer using `GridSearchCV`
parameters = {'n_estimators':[150,200,250,300], 'max_depth':[15,20,25]}
forest = RandomForestClassifier()
clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5)
clf.fit(X, y)


# Below, it shows the best parameters with the best prediction score.
print(clf.best_params_)
print(clf.best_score_)


# Plug the best parameters back to the decision tree classifer and fit the training set.
clf_forest = RandomForestClassifier(n_estimators=150, max_depth=25)
clf_forest.fit(X_train, y_train)


# Now, the training accuracy improves to 98% even though the test accuracy remains the same.
pred = clf_forest.predict(X_train)
accuracy_score(y_train, pred)

pred_test = clf_forest.predict(X_test)
accuracy_score(y_test, pred_test)


# Choose this trained model to predict all churned label for the whole `features` dataset.
features['predict'] = clf_forest.predict(all_other)


# Filter out which the actual `churned` label is different with the predictive `churned` label.
diff = features[features.churned != features.predict]
print((1 - len(diff)/len(features))*100)

# Conclusion: The accuracy for this model is 94%.

# Here, I add `predict_score_max` into `features` data for showing that the predicted probability that the computer predicts an user is churned or not churned.
predict_probability = clf_forest.predict_proba(all_other)
pred_prob = []

for prob in predict_probability:
    pred_prob.append(max(prob))

features['predict_score_max'] = pred_prob
