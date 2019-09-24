#
# 1. The required imports
#

import os, numpy, glob
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

#
# 2. Read in images, get the image vectors (X for the input)
#    & label vectors (y for the output)
#

os.chdir('malimg')                  # the parent folder with sub-folders

list_fams = os.listdir(os.getcwd()) # vector of strings with family names
no_fams = len(list_fams)            # No. of families

no_imgs = []                        # No. of samples per family
X = []                              ## for a dynamic input vectors ##
#for i in range(no_fams):
for index, i in enumerate([8, 22]): # just for two families with similar resolution
    os.chdir(list_fams[i])
    img_list = glob.glob('*.png')   # getting only 'png' files in a folder
    len1 = len(img_list)            # number of images in this family
    no_imgs.append(len1)
    for j in range(len1):
        im = Image.open(img_list[j])    
        im1 = im.resize((32,32),Image.ANTIALIAS) # resize; just a simple way!!
        X.append(numpy.array(im1)) # and save
    os.chdir('..')
X = numpy.array(X).astype('float32')# change the from integer to float
total = sum(no_imgs)                # total number of all samples
X = X.reshape(total, 32, 32, 1)     # add channel number 1 for B&W

# Assign y with the class index (0 ~ no_fams-1)
# Each vector is 'no_fams' long with zeros padded,
# except for the bit corresponding to the index set to 1 

y = numpy.zeros(shape = (total, no_fams)) ### output vectors ###

i = 0
for index, number in enumerate(no_imgs):
    for j in range(number):
        y[i][index] = 1
        i += 1

#
# 3. Split the data into X_train, X_test, y_train, & y_test
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#
# 4. Build the model, run it, and evaluate it
#

batch_size = 64
epochs = 10

Malware_Model = Sequential()
Malware_Model.add(Conv2D(32, kernel_size=(3,3),activation='linear',input_shape=(32,32,1),padding='same'))
Malware_Model.add(LeakyReLU(alpha=0.1))
Malware_Model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
Malware_Model.add(Conv2D(64,(3, 3), activation='linear', padding='same'))
Malware_Model.add(LeakyReLU(alpha=0.1))
Malware_Model.add(Dense(1024, activation='linear'))
Malware_Model.add(LeakyReLU(alpha=0.1))
Malware_Model.add(Dropout(0.4))
Malware_Model.add(Flatten())
Malware_Model.add(Dense(no_fams, activation='softmax'))
Malware_Model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
Malware_Model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test, y_test))
test_eval = Malware_Model.evaluate(X_test, y_test, verbose=0)
print('The accuracy of the Test is:', test_eval[1])
