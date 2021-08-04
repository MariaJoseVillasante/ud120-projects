#!/usr/bin/python3

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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
### your code goes here ###

def svm_author_id():
    #from sklearn.svm import svc
    from sklearn import svm
    ## create Classifier
    clf = svm.SVC(kernel="rbf",C=10000)#kernel="rbf",C=10000)#, kernel="rbf")

    #fit the classifier on the training features and labels
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")

    #use the train classifier to predict labels for the test features
    t0 = time()
    pred = clf.predict(features_test)
    print('pred 10', pred[10])
    print('pred 26', pred[26])
    print('pred 50', pred[50])
    #cero = pred.count(0)
    #uno = pred.count(1)
    print('Pred=1 Chris', (pred==0).sum())
    print('Pred=0 Sara', (pred==1).sum())
#    print(pred[10])
	#print(pred[26])
#	print(pred[50])

    print("Predicting Time:", round(time()-t0, 3), "s")



    from sklearn.metrics import accuracy_score
    #accuracy = accuracy_score(labels_test, labels_train)#TODO
    accurracy = accuracy_score(pred,labels_test)
    #accuracy = clf.score(features_test, labels_test)
    return accurracy
r = svm_author_id()
print('Accurracy: ', r)
'''
No. of Chris training emails :  7936
No. of Sara training emails :  7884

kernel = "linear"
Training Time: 81.752 s
Predicting Time: 8.058 s
Accurracy: 0.9840728100113766

kernel = "linear"
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
Training Time: 0.029 s
Predicting Time: 0.253 s
Accurracy:  0.8845278725824801

kernel = "rbf"
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
Training Time: 0.031 s
Predicting Time: 0.541 s
Accurracy:  0.8953356086461889 #Dice que está mal, pero no encuentro el error

C=10.0
kernel = "rbf"
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
Training Time: 0.032 s
Predicting Time: 0.557 s
Accurracy:  0.8998862343572241

C=100.
kernel = "rbf"
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
Training Time: 0.031 s
Predicting Time: 0.533 s
Accurracy:  0.8998862343572241

C=1000.
kernel = "rbf"
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
Training Time: 0.029 s
Predicting Time: 0.528 s
Accurracy:  0.8998862343572241

C=10000, kernel = "rbf"
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
Training Time: 0.03 s
Predicting Time: 0.542 s
Accurracy:  0.8998862343572241

kernel = "rbf"
Training Time: 108.249 s
Predicting Time: 15.624 s
Accurracy:  0.9926052332195677

C=10, kernel = "rbf"
Training Time: 96.103 s
Predicting Time: 14.203 s
Accurracy:  0.9948805460750854

C=100, kernel = "rbf"
Training Time: 91.293 s
Predicting Time: 13.269 s
Accurracy:  0.9960182025028441

C=1000 kernel = "rbf"
Training Time: 95.018 s
Predicting Time: 13.325 s
Accurracy:  0.9960182025028441

C=10000 kernel = "rbf"
Training Time: 84.14 s
Predicting Time: 12.178 s
Accurracy:  0.9960182025028441
Training Time: 80.817 s
pred 10 1
pred 26 0
pred 50 1
Pred=1 Chris 892
Pred=0 Sara 866
Predicting Time: 11.549 s
Accurracy:  0.9960182025028441
'''

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
