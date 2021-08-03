#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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


##############################################################
# Enter Your Code Here
def nb_author_id():
    from sklearn.naive_bayes import GaussianNB
    ## create Classifier
    clf = GaussianNB()
    
    #fit the classifier on the training features and labels
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")
    
    #use the train classifier to predict labels for the test features
    t0 = time()
    pred = clf.predict(features_test)
    print("Predicting Time:", round(time()-t0, 3), "s")
    
    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    from sklearn.metrics import accuracy_score
    #accuracy = accuracy_score(labels_test, labels_train)#TODO
    accuracy = accuracy_score(pred, labels_test)
    #accuracy = clf.score(features_test, labels_test)
    return accuracy
r = nb_author_id()
print(r)

'''
Training Time: 1.518 s
Predicting Time: 0.24 s
0.9732650739476678
'''



##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################
