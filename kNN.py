#
# 1. The required imports
#

import os, random, time
import _pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
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
# 3. Divide the data into training set & test set
#    (70% training and 30% test)
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#
# 4. Build the model, run it, and evaluate it
#

conf_mat = np.zeros((len(list_fams),len(list_fams))) # Initializing the Confusion Matrix

n_neighbors = 2;                        # better to have this at the start of the code

clf = KNeighborsClassifier(n_neighbors,weights='distance') 
 
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
conf_mat = confusion_matrix(y_test,y_predict)
 
#conf_mat = conf_mat.T                   # since rows and  cols are interchanged
#conf_mat_norm = conf_mat/no_imgs        # Normalizing the confusion matrix
#confusion matrix re- calculation !!!!
conf_mat_norm = conf_mat.astype('float')/conf_mat.sum(axis=1)[:,np.newaxis]

#
# 5. View the results
#

print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
print("Loss:",metrics.zero_one_loss(y_test, y_predict))
print("Precision:", metrics.precision_score(y_test,y_predict, average='macro'))

#print("Recal score:", metrics.recall_score(y_test,y_predict,average='micro'))
print("F-1 score:", metrics.f1_score(y_test,y_predict,average='macro'))

# Viewing the confusion matrix
os.chdir('..')
conf_mat2 = np.around(conf_mat_norm,decimals=2) # rounding to display in figure
avg_diagonal = np.trace(conf_mat2)/len(list_fams)
print('Average of the Confusion matrix diagonal: ', avg_diagonal) # average diagonal values

plt.imshow(conf_mat2,interpolation='nearest')
for x in range(len(list_fams)):
  for y in range(len(list_fams)):
    fg=plt.annotate(str(conf_mat2[x][y]),xy=(y,x),ha='center',va='center')
    fg.set_fontsize(5)

plt.xticks(range(len(list_fams)),list_fams,rotation=90,fontsize=7)
plt.yticks(range(len(list_fams)),list_fams,fontsize=7)
plt.title('Confusion matrix')
plt.colorbar()
plt.show()
plt.savefig('./fig/confusion_matrix.png')

# Save conf. matrix into a dat file for post-processing
with open('./dat/conf_mat.dat','wb') as f:
    for line in np.matrix(conf_mat2):
        np.savetxt(f,line,fmt='%.2f',)
f.close()

