import csv
import numpy as np
from sklearn import svm
import sys
import os
import os.path
import re
from sklearn.externals import joblib

clf=joblib.load('test.pkl')

category=['bread','breakfastSandwich','burger','burrito','chicken','donuts','pie','pizza','salad','toastSandwich']
test=list()
real=list()
testResult=list()
test_filenames=os.walk(sys.argv[1])#test file dir
for root,dirs,files in test_filenames:
    files.sort()
    for f in files:
        each=list()
        if not f.startswith('.'):
            ca = re.findall('(\S+?)_',f)
            index = category.index(ca[0]) + 1
            real.append(index)
            test_filepath=sys.argv[1]+'/'+f
            with open(test_filepath,'rb') as csvfile:
                spamreader = csv.reader(csvfile)           
                for row in spamreader:
                    each.append(row)
            test.append(each)
test=np.array(test)
testset_size = len(test)
TwoDim_testset = test.reshape(testset_size,-1)
print "Begin predict:"
testResult=clf.predict(TwoDim_testset)
print(testResult)

#order the list and calculate accuracy
accuracy=[]
Abread=[]
Rbread=[]
AbreakfastSandwich=[]
RbreakfastSandwich=[]
Aburger=[]
Rburger=[]
Aburrito=[]
Rburrito=[]
Achicken=[]
Rchicken=[]
Adonuts=[]
Rdonuts=[]
Apie=[]
Rpie=[]
Apizza=[]
Rpizza=[]
Asalad=[]
Rsalad=[]
AtoastSandwich=[]
RtoastSandwich=[]

for i in range(len(testResult)):
    if testResult[i]==real[i]:
        accuracy.extend([1])
        if real[i]==1:
            Abread.extend([1])
            Rbread.extend([testResult[i]])
        elif real[i]==2: 
            AbreakfastSandwichextend([1])
            RbreakfastSandwich.extend([testResult[i]])
        elif real[i]==3:
            Aburger.extend([1])
            Rburger.extend([testResult[i]])
        elif real[i]==4: 
            Aburrito.extend([1])
            Rburrito.extend([testResult[i]])
        elif real[i]==5: 
            Achicken.extend([1])
            Rchicken.extend([testResult[i]])
        elif real[i]==6:
            Adonuts.extend([1])
            Rdonuts.extend([testResult[i]])
        elif real[i]==7: 
            Apie.extend([1])
            Rpie.extend([testResult[i]])
        elif real[i]==8: 
            Apizza.extend([1])
            Rpizza.extend([testResult[i]])
        elif real[i]==9: 
            Asalad.extend([1])
            Rsalad.extend([testResult[i]])
        elif real[i]==10: 
            AtoastSandwich.extend([1]) 
            RtoastSandwich.extend([testResult[i]])
    else:
        accuracy.extend([0])
        if real[i]==1:
            Abread.extend([0])
            Rbread.extend([testResult[i]])
        elif real[i]==2: 
            AbreakfastSandwichextend([0])
            RbreakfastSandwich.extend([testResult[i]])
        elif real[i]==3:
            Aburger.extend([0])
            Rburger.extend([testResult[i]])
        elif real[i]==4: 
            Aburrito.extend([0])
            Rburrito.extend([testResult[i]])
        elif real[i]==5: 
            Achicken.extend([0])
            Rchicken.extend([testResult[i]])
        elif real[i]==6:
            Adonuts.extend([0])
            Rdonuts.extend([testResult[i]])
        elif real[i]==7: 
            Apie.extend([0])
            Rpie.extend([testResult[i]])
        elif real[i]==8: 
            Apizza.extend([0])
            Rpizza.extend([testResult[i]])
        elif real[i]==9: 
            Asalad.extend([0])
            Rsalad.extend([testResult[i]])
        elif real[i]==10: 
            AtoastSandwich.extend([0]) 
            RtoastSandwich.extend([testResult[i]])
print 'Total Accuracy = ',float(sum(accuracy))/len(accuracy)
print 'bread:',float(sum(Abread))/len(Abread),'\nbreakfastSandwich:',float(sum(AbreakfastSandwich))/len(AbreakfastSandwich),'\nburger:',float(sum(Aburger))/len(Aburger),'\nburrito:',float(sum(Aburrito))/len(Aburrito),'\nchicken:',float(sum(Achicken))/len(Achicken),'\ndonuts:',float(sum(Adonuts))/len(Adonuts),'\npie:',float(sum(Apie))/len(Apie),'\npizza:',float(sum(Apizza))/len(Apizza),'\nsalad:',float(sum(Asalad))/len(Asalad),'\nsandwich:',float(sum(AtoastSandwich))/len(AtoastSandwich)
print 'Prediction for bread:',Rbread,'\nPrediction for breakfastSandwich:',RbreakfastSandwich,'\nPrediction for burger:',Rburger,'\nPrediction for burrito:',Rburrito,'\nPrediction for chicken:',Rchicken,'\nPrediction for donuts',Rdonuts,'\nPrediction for pie:',Rpie,'\nPrediction for pizza:',Rpizza,'\nPrediction for salad:',Rsalad,'\nPrediction for sandwich:',RtoastSandwich

#============     confusion matrix creation     =============
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(real, testResult)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()