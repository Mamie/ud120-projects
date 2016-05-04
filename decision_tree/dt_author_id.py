#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print len(features_train[0])



#########################################################
### your code goes here ###

clf = DecisionTreeClassifier(min_samples_split=40)
time0 = time()
clf.fit(features_train, labels_train)
print "Time for training the DT classifier is " + str(time()-time0)
time1 = time()
pred = clf.predict(features_test)
print "Time for prediction is " + str(time()-time1)
acc = accuracy_score(pred, labels_test)
print "The accuracy of DT classifier is " + str(acc)

#########################################################


