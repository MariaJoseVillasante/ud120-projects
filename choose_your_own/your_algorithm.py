#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

def knc_author_id():
    #from sklearn.svm import svc
    from sklearn.neighbors import KNeighborsClassifier
    ## create Classifier
    clf = KNeighborsClassifier(n_neighbors=3)

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
    return accurracy, clf
r = knc_author_id()
print('Accurracy: ', r)
print('The number of features:', len(features_train[1]))








try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
