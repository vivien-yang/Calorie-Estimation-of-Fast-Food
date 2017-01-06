import matplotlib.pyplot as plt
import os
import os.path
import sys
import re

from skimage.feature import hog
from skimage import data, color, exposure
from skimage import io
from PIL import Image
import csv

filenames=os.walk(sys.argv[1])
target_filenames=sys.argv[2]#crop target dir
csv_filename=sys.argv[3]#store the csv dir
size=120

for root,dirs,files in filenames:
   files.sort()
   for f in files:
        print (f)
        #resize
        pri_image = Image.open(sys.argv[1] + '/' + f)  
        pri_image.resize((size,size),Image.ANTIALIAS ).save(target_filenames+'/'+f)  

        img_111=io.imread(target_filenames+'/'+f)
        image=color.rgb2gray(img_111)
        hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=False)
        row = hog_image.shape[0]

        with open(csv_filename+'/'+f+'.csv','wb') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(hog_image)