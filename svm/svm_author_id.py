#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#size = len(features_train)/100
#features_train = features_train[:size]
#labels_train = labels_train[:size]

#########################################################
### your code goes here ###
def train(C):
    print "C paramter is: " + str(C)
    clf = SVC(kernel="rbf", C=C)
    time0 = time()
    clf.fit(features_train, labels_train)
    print "Time to train the SVM classifier is: " + str(time()-time0)
    time1 = time()
    pred = clf.predict(features_test)
    print "Time to predict the test set is: " + str(time()-time1)
    acc = accuracy_score(pred, labels_test)
    print "The accuracy of the SVM classifier is: " + str(acc)
    return pred

if __name__=='__main__':
    #C = [10.0, 100.0, 1000.0, 10000.0]
    #for c in C:
    #    train(c)
    pred = train(10000.0)
    #test = [10, 26, 50]
    #for t in test:
    #    print "The element %d of the prediction set is %d" % (t, pred[t])
    print "%d test events are predicted to be in the 'Chris' (1) class" % sum(pred)
#########################################################


