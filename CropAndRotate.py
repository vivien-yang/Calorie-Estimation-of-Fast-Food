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

filenames=os.walk(sys.argv[1])#original pic dir
target_filenames=sys.argv[2]#dir for pic after crop
csv_filename=sys.argv[3]#store the csv dir
type=sys.argv[4] #category of pic
for root,dirs,files in filenames:
   files.sort()
   for f in files:
        print (f)
        img = Image.open(sys.argv[1] + '/' + f)
        width = img.size[0]
        height = img.size[1]
        pic = img.crop((width / 5, height /3, width * 3 / 4, height))
        pic.save(target_filenames+'/'+type+'_0_'+f)
        
        for i in range(0,8):
            img_r = pic.rotate(45*i)
            img_r.save(target_filenames+'/'+type+'_'+str(i)+'_'+f)