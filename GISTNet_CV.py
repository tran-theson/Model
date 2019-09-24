#
# 1. The required imports
#

import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold

#
# 2. Read in feature vectors (X for the input) & label vector
#    and generate the y vectors for the output
#

os.chdir('features')                    # the folder with features saved
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
# 3. Divide the data into folds
#

n_samples, n_features = X.shape 

kfold = 10                              # no. of folds (better to have this at the start of the code)

skf = StratifiedKFold(n_splits=kfold, shuffle=True)
train_indices_arr = []                  # array with the indices for a split
test_indices_arr = []
for train_index, test_index in skf.split(X, Y):
  train_indices_arr.append(train_index)
  test_indices_arr.append(test_index)

#
# 4. Build the model, run it, and evaluate it
#

accuracy = 0
batch_size = 64
epochs = 10

for i in range(kfold):
# 10-fold Cross Validation
    train_indices = train_indices_arr[i]
    test_indices = test_indices_arr[i]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    model = Sequential()
    model.add(Dense(160, activation='relu', input_dim=320))
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(no_fams, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test, y_test))
    test_eval = model.evaluate(X_test, y_test, verbose=0)
    accuracy += test_eval[1]

# Print the final result
print("Accuracy:",accuracy/kfold)
