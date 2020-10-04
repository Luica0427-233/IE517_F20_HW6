# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 00:18:44 2020

@author: Lucia
"""

import pandas as pd
import numpy as np
df = pd.read_csv('D:\Desktop\IE517 ML in Fin Lab\Module6\HW6\ccdefault.csv')

X = df.iloc[:,1:24]
y = df.iloc[:,24]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#dt = DecisionTreeClassifier(max_depth = 6, random_state = 1)


# Part 1: Random test train splits


acc_train = []
acc_test = []

for random_numbers in range(1,11):
    # train splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = random_numbers, stratify=y)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    # decision tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train_std, y_train)
    y_train_pred = dt.predict(X_train_std)
    y_test_pred = dt.predict(X_test_std)
    # accuracy scores
    acc_train.append(accuracy_score(y_train, y_train_pred))
    acc_test.append(accuracy_score(y_test, y_test_pred))
    
# the mean and the standard deviation
print('Accuracy scores for training set:{}'.format(acc_train))
print('Accuracy scores for test set:{}'.format(acc_test))
print('Mean of accuracy scores for training set:{}'.format(np.mean(acc_train)))
print('Mean of accuracy scores for test set:{}'.format(np.mean(acc_test)))
print('Standard deviation of accuracy scores for training set:{}'.format(np.std(acc_train)))
print('Standard deviation of accuracy scores for test set:{}'.format(np.std(acc_test)))

    
# Part 2: Cross validation


# train splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train_std, y_train)

# accuracy scores
CV_scores = cross_val_score(dt, X_train_std, y_train, cv=10,
                                scoring='accuracy',
                                n_jobs=-1)

# the mean and the standard deviation of the fold scores
CV_mean = np.mean(CV_scores)
CV_std = np.std(CV_scores)
print('The individual fold accuracy scores:', [float('{}'.format(i)) for i in CV_scores])
print('The mean of the fold scores: ', CV_mean)
print('The standard deviation of the fold scores: ', CV_std)

# the out-of-sample accuracy scores
CV_scores_test = cross_val_score(dt, X_test_std, y_test, cv=10,
                                scoring='accuracy',
                                n_jobs=-1)
CV_mean_test = np.mean(CV_scores_test)
CV_std_test = np.std(CV_scores_test)
print('The out-of-sample accuracy score:', CV_scores_test)
print('The mean of the out-of-sample accuracy score: ', CV_mean_test)
print('The standard deviation of the out-of-sample accuracy score: ', CV_std_test)



print("-----------")
print("My name is Lu Yuan")
print("My NetID is: luyuan3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
    