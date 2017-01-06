from __future__ import print_function
import numpy as np
import os
import os.path
import sys
import re
import csv
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from PIL import Image
from skimage.color import label2rgb
from skimage import io
from skimage import color

radius = 3
n_points = 8 * radius
METHOD = 'uniform'

filenames=os.walk(sys.argv[1])
csv_filename=sys.argv[2]#store the csv dir

for root,dirs,files in filenames:
   files.sort()
   for f in files:
        if not f.startswith('.'):
            img=io.imread(sys.argv[1]+'/'+f)
            image=color.rgb2gray(img)
            lbp = local_binary_pattern(image, n_points, radius, METHOD)

            with open(csv_filename+'/'+f+'.csv','wb') as csvfile:
                spamwriter = csv.writer(csvfile)
                for row in lbp:
                    spamwriter.writerow(row)
         