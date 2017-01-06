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
       if not f.startswith('.'):
           print (sys.argv[1] + '/' + f)
           filepath=sys.argv[1]+'/'+f
           with open(filepath,'rb') as csvfile:
               spamreader = csv.reader(csvfile)
               for row in spamreader:
                   x.append(row)

#category for all the images
y=[1]*Nbread+[2]*Nburger+[3]*Nburrito+[4]*Nchicken+[5]*Ndonuts+[6]*Npie+[7]*Npizza+[8]*Nsalad+[9]*Nsandwich
print "Start training......"

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(x, y)

joblib.dump(clf,'test.pkl')
print 'Summary of SVM:\n',clf
