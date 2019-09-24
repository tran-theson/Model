#
# 1. The required imports
#

import os, random, time
import _pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#
# 2. Read in feature vectors (X for the input), label vector (y), & no_imgs
#
os.chdir('features')                    # the folder with features saved
with open('malimg.p', 'rb') as f:
  list_fams = _pickle.load(f)
X = np.load('malimg_features.npy')
y = np.load('malimg_labels.npy')
no_imgs = np.load('malimg_no_imgs.npy')

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

conf_mat = np.zeros((len(no_imgs),len(no_imgs))) # Initializing the Confusion Matrix

n_neighbors = 1;                        # better to have this at the start of the code

# 10-fold Cross Validation

for i in range(kfold):
 train_indices = train_indices_arr[i]
 test_indices = test_indices_arr[i]
 clf = []
 clf = KNeighborsClassifier(n_neighbors,weights='distance') 
 X_train = X[train_indices]
 y_train = y[train_indices]
 X_test = X[test_indices]
 y_test = y[test_indices]
 
 # Training
 tic = time.time()
 clf.fit(X_train,y_train) 
 toc = time.time()
 print("training time= ", toc-tic)      # roughly 2.5 secs
 
 # Testing
 y_predict = []
 tic = time.time()
 y_predict = clf.predict(X_test)        # output is labels and not indices
 toc = time.time()
 print("testing time = ", toc-tic)      # roughly 0.3 secs

 # Compute confusion matrix
 cm = []
 cm = confusion_matrix(y_test,y_predict)
 conf_mat = conf_mat + cm 
 
conf_mat = conf_mat.T                   # since rows and  cols are interchanged
avg_acc = np.trace(conf_mat)/sum(no_imgs)
conf_mat_norm = conf_mat/no_imgs        # Normalizing the confusion matrix

#
# 5. View the results
#

print("The average classification accuracy is: ", avg_acc)

# Viewing the confusion matrix

#conf_mat2 = np.around(conf_mat_norm,decimals=2) # rounding to display in figure
#plt.imshow(conf_mat2,interpolation='nearest')
#for x in range(len(list_fams)):
#  for y in range(len(list_fams)):
#    plt.annotate(str(conf_mat2[x][y]),xy=(y,x),ha='center',va='center')

#plt.xticks(range(len(list_fams)),list_fams,rotation=90,fontsize=11)
#plt.yticks(range(len(list_fams)),list_fams,fontsize=11)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.show()
#plt.savefig('confusion_matrix.png')
