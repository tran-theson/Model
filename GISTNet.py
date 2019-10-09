#
# 1. The required imports
#

import os
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import _pickle
import math

#
# 2. Read in feature vectors (X for the input) & label vector
#    and generate the y vectors for the output
#

os.chdir('features')                    # the folder with features saved
with open('malimg.p', 'rb') as f:
  list_fams = _pickle.load(f)
X = np.load('malimg_features.npy')
total = len(X)                          # dataSize 
Y = np.load('malimg_labels.npy')
Y = np.array(Y).astype('int')

# Assign y with the class index (0 ~ no_fams-1)
# Each vector is 'no_fams' long with zeros padded,
# except for the bit corresponding to the index set to 1

no_fams = int(Y[total-1]+1)             # number of families in the dataset
y = np.zeros(shape = (total, no_fams))  ### output vectors ###
for i in range(total):
    y[i][Y[i]] = 1

#
# 3. Split the data into X_train, X_test, y_train, & y_test
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#
# 4. Build the model, run it, and evaluate it
#

batch_size = 64
epochs = 10
# Implement of learning rate schedule
def lr_decay(epoch):
    return 0.01*math.pow(0.8,epoch)
lr_decay_callback = keras.callbacks.LearningRateScheduler(lr_decay,verbose=1)


model = Sequential()
model.add(Dense(160, activation='relu', input_dim=320))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(40, activation='relu'))
model.add(Dense(no_fams, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          callbacks=[lr_decay_callback],
          validation_data=(X_test, y_test))

y_predict = model.predict(X_test,batch_size=batch_size, verbose=0)
conf_mat = confusion_matrix(y_test.argmax(axis=1),y_predict.argmax(axis=1))
conf_mat_norm = conf_mat.astype('float')/conf_mat.sum(axis=1)[:,np.newaxis]

test_eval = model.evaluate(X_test, y_test, verbose=0)

# Display results
print('The accuracy is:', test_eval[1])
print('The evaluation loss is:', test_eval[0])

conf_mat2 = np.around(conf_mat_norm,decimals=2) # rounding to display in figure
##plt.imshow(conf_mat2,interpolation='nearest')
##for x in range(len(list_fams)):
##  for y in range(len(list_fams)):
##    fg=plt.annotate(str(conf_mat2[x][y]),xy=(y,x),ha='center',va='center')
##    fg.set_fontsize(5)
##
##plt.xticks(range(len(list_fams)),list_fams,rotation=90,fontsize=8)
##plt.yticks(range(len(list_fams)),list_fams,fontsize=8)
##plt.title('Confusion matrix')
##plt.colorbar()
##plt.show()
##plt.savefig('gist_matrix.png')

# Save conf. matrix into a dat file for post-processing
os.chdir('..')
with open('../dat/conf_mat_gist.dat','wb') as f:
    for line in np.matrix(conf_mat2):
        np.savetxt(f,line,fmt='%.2f',)
f.close()
