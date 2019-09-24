#
# 1. The required imports
#

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import metrics

#
# 2. Read in feature vectors (X for the input) & label vector (y)
#
os.chdir('features')                    # the folder with features saved
X = np.load('malimg_features.npy')
y = np.load('malimg_labels.npy')

#
# 3. Divide the data into folds
#

n_samples, n_features = X.shape 

kfold = 10                              # no. of folds (better to have this at the start of the code)

skf = StratifiedKFold(n_splits=kfold, shuffle=True)
train_indices_arr = []                  # array with the indices for a split
test_indices_arr = []
for train_index, test_index in skf.split(X, y):
  train_indices_arr.append(train_index)
  test_indices_arr.append(test_index)

#
# 4. Build the model, run it, and evaluate it
#

# 10-fold Cross Validation

accuracy = 0
for i in range(kfold):
 train_indices = train_indices_arr[i]
 test_indices = test_indices_arr[i]
 X_train = X[train_indices]
 y_train = y[train_indices]
 X_test = X[test_indices]
 y_test = y[test_indices]
 
#Create a svm Classifier
 clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
 clf.fit(X_train, y_train)

#Predict the response for test dataset
 y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
 accuracy += metrics.accuracy_score(y_test, y_pred)

# Print the final result
print("Accuracy:",accuracy/kfold)
