#
# 1. The required imports
#

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#
# 2. Read in feature vectors (X for the input) & label vector (y)
#
os.chdir('features')                    # the folder with features saved
X = np.load('malimg_features.npy')
y = np.load('malimg_labels.npy')

#
# 3. Divide the data into training set & test set
#    (70% training and 30% test)
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)

#
# 4. Build the model, run it, and evaluate it
#

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
