import csv
import numpy as np
from sklearn import svm
import sys
import os
import os.path
import re
from sklearn.externals import joblib

Nbread=1504
Nburger=1720
Nburrito=1024
Nchicken=1040
Ndonuts=1280
Npie=232
Npizza=1264
Nsalad=1176
Nsandwich=2744

x=list()
filenames=os.walk(sys.argv[1])

for root,dirs,files in filenames:
   files.sort()
   for f in files:
       each=list()
       if not f.startswith('.'):
           filepath=sys.argv[1]+'/'+f
           with open(filepath,'rb') as csvfile:
               spamreader = csv.reader(csvfile)
               for row in spamreader:
                   each.append(row)
           x.append(each)
           
X = np.array(x)
dataset_size = len(X)
TwoDim_dataset = X.reshape(dataset_size,-1)

#category for all the images
y=[1]*Nbread+[2]*Nburger+[3]*Nburrito+[4]*Nchicken+[5]*Npie+[6]*Npizza+[7]*Nsalad+[8]*Nsandwich
print "Start training......"

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(TwoDim_dataset, y)

joblib.dump(clf,'test.pkl')
print 'Summary of SVM:\n',clf
