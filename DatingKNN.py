# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:49:54 2019

"""
import os
import numpy as np
import operator

filename = 'datingTestSet2.txt'
#fr = open(C:\Users\Binchi\Desktop\Files\AiLearning-master\data\2.KNN\datingTestSet2.txt)
filepath = os.path.join('C:\\Users\\n927287\\','Desktop','ML-master','ML-master','AiLearning-master','data','2.KNN',filename)

#print(filepath)

#data = open(filepath)

"import data from raw file"
#important note here: data.readlines() will only be operated once. The file pointer is not at the end of the file. readlines() will not return data.
def openfile(filepath):
    rows = 0
    with open(filepath, 'r') as data:
        if len(data.read(1)) == 0:
            print('FILE IS EMPTY')
        else:
            data.seek(0)
            for line in data:
    #            print(line)
                rows+= 1 
    #        print(rows)
            
            
    returnmat = np.zeros((rows,3))
    labelmat = []
    index = 0
    with open(filepath, 'r') as data:
        data.seek(0)
        for line in data :
             line = line.strip()
             singleline = line.split('\t')
             returnmat[index,:] = singleline[0:3]
             labelmat.append(int(singleline[-1]))
             index += 1
    return returnmat,labelmat

def autonorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    
    ranges = maxVals - minVals
    m = dataset.shape[0]
    normdataset = dataset - np.tile(minVals,(m,1))
    normdataset = normdataset / np.tile(ranges,(m,1))
    return normdataset, ranges, minVals


def classify0(inX, dataset, labels, k):
    
#Euclidian Space
    datasetsize = dataset.shape[0]
    diffmat = dataset - np.tile(inX,(datasetsize,1))
    squared = diffmat**2
    sum_sq = squared.sum(axis = 1)
    distances = sum_sq**0.5
    
    classcount = {}
    sortedDistance = distances.argsort()
    for i in range(k):
        votelabel = labels[sortedDistance[i]]
        classcount[votelabel] = classcount.get(votelabel,0) + 1
        sortedclasscount = sorted(classcount.items(), key= operator.itemgetter(1), reverse = True)
    return sortedclasscount[0][0]

def datingClassTest(filepath,hoRatio):
     # Testing Sample Percentage
    returnmat,labelmat = openfile(filepath)
    normdata, ranges, minVals = autonorm(returnmat)
    m = normdata.shape[0]
    numTestVecs = int(m * hoRatio)
    errorcount = 0
    for i in range(numTestVecs):
        classifyresult = classify0(normdata[i,:],normdata[numTestVecs:m,:],labelmat[numTestVecs:m],3)
        if (classifyresult != labelmat[i]):
            errorcount += 1
    errorrate = errorcount / numTestVecs
    print('Error number is', errorcount)
    print('Error Rate is',errorrate)
     
    
def newinput(personal_database):
    var1 = float(input("percentage of time spent playing video games ?"))
    var2 = float(input("frequent filer miles earned per year?"))
    var3 = float(input("liters of ice cream consumed per year?"))
    inputarray = np.array([var1,var2,var3])
    returnmat,labelmat = openfile(personal_database)
    normdataset,ranges,minVals = autonorm(returnmat)
    result = classify0((inputarray-minVals)/ranges,normdataset,labelmat,3)
    mapping = {1:'not likely',2:'Plain',3:'Like'}
    print('I like this one',mapping[result])
    
    