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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
def dt_author_id():
    #from sklearn.svm import svc
    from sklearn import tree
    ## create Classifier
    clf = tree.DecisionTreeClassifier(min_samples_split=40)

    #fit the classifier on the training features and labels
    t0 = time()
    clf = clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")

    #use the train classifier to predict labels for the test features
    t0 = time()
    predict = clf.predict(features_test)
    print("Predicting Time:", round(time()-t0, 3), "s")

    from sklearn.metrics import accuracy_score
    #accurracy = accuracy_score(pred,labels_test)
    acc_min_samples_split_40 = accuracy_score(predict,labels_test)
    accurracy = acc_min_samples_split_40
    return accurracy
r = dt_author_id()
print('Accurracy: ', r)
print('The number of featrues:', len(features_train[1]))
#########################################################
'''
min_samples_split=40
Training Time: 38.761 s
Predicting Time: 0.025 s
Accurracy:  0.9766780432309442
The number of featrues: 3785
from ../tools/email_preprocess.py: selector = SelectPercentile(f_classif, percentile=10)

Training Time: 2.972 s
Predicting Time: 0.002 s
Accurracy:  0.9670079635949943
The number of featrues: 379
from ../tools/email_preprocess.py: selector = SelectPercentile(f_classif, percentile=1)

'''

#########################################################


