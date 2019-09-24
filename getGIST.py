#
# 1. Read in images, get the family name vector & label vector
#

import os,glob,numpy

os.chdir('malimg') # the parent folder with sub-folders

list_fams = os.listdir(os.getcwd()) ### vector of strings with family names ###

no_imgs = [] # No. of samples per family

for i in range(len(list_fams)):
 os.chdir(list_fams[i])
 len1 = len(glob.glob('*.png')) # assuming the images are stored as 'png'
 no_imgs.append(len1)
 os.chdir('..')

# Assign y with the class index (0~24)

total = sum(no_imgs) # total number of all samples
y = numpy.zeros(total) ### label vector ###

temp1 = numpy.zeros(len(no_imgs)+1)
temp1[1:len(temp1)]=no_imgs
temp2 = int(temp1[0]); # now temp2 is [0 no_imgs]

for jj in range(len(no_imgs)): 
    temp3 = temp2 +int(temp1[jj+1])
    for ii in range(temp2,temp3): 
       y[ii] = jj
    temp2 = temp2+ int(temp1[jj+1])

#
# 2. Compute the features
#    NB: It is not easy, if not impossible, to run the following in Windows due to 'import leargist' problems
#

import Image,leargist 

X = numpy.zeros((sum(no_imgs),320)) # Feature Matrix
cnt = 0
for i in range(len(list_fams)):
   os.chdir(list_fams[i])
   img_list = glob.glob('*.png') # Getting only 'png' files in a folder
   for j in range(len(img_list)):
    im = Image.open(img_list[j])
    im1 = im.resize((64,64),Image.ANTIALIAS); # for faster computation
    des = leargist.color_gist(im1)
    X[cnt] = des[0:320]
    cnt = cnt + 1 
   os.chdir('..')

#
# 3. Save the features
#

import _pickle

os.chdir('../features') # The folder to save the computed features
_pickle.dump(list_fams, open('malimg.p', 'wb')) # binary dump
numpy.save('malimg_no_imgs.npy', no_imgs)
numpy.save('malimg_features.npy', X) # numpy files
numpy.save('malimg_labels.npy', y)
numpy.savetxt('malimg_features.txt', X) # text files: bigger than numpy files
numpy.savetxt('malimg_labels.txt', y)
