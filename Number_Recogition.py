# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:26:50 2019

@author: n927287
"""
import os
import numpy as np
import operator
from math import *

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVect[0,i*32+j] = int(line[j])
    return returnVect


def load_training():
    filename_train = 'trainingDigits'
    filepath_train = os.path.join('C:\\Users\\n927287\\','Desktop','ML-master','ML-master','AiLearning-master','data','2.KNN',filename_train)
    trainingFileList = os.listdir(filepath_train)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    ImgLabels_train = []
    for i in range(m):
        fileStr_suffix = trainingFileList[i]
        fileStr_clean = fileStr_suffix.split('.')[0]
        file_label = int(fileStr_clean.split('_')[0])
        ImgLabels_train.append(file_label)
        singlePath = os.path.join('C:\\Users\\n927287\\','Desktop','ML-master','ML-master','AiLearning-master','data','2.KNN',filename_train,trainingFileList[i])
        trainingMat[i,:] = img2vector(singlePath)

    return trainingMat,ImgLabels_train,m
    
def load_testing():    
    filename_test = 'testDigits'
    filepath_test = os.path.join('C:\\Users\\n927287\\','Desktop','ML-master','ML-master','AiLearning-master','data','2.KNN',filename_test)
    
    testingFileList = os.listdir(filepath_test)
    n = len(testingFileList)
    testingMat = np.zeros((n,1024))
    ImgLabels_test = []
    
    for i in range(n):
        fileStr_suffix = testingFileList[i]
        fileStr_clean = fileStr_suffix.split('.')[0]
        file_label = int(fileStr_clean.split('_')[0])
        ImgLabels_test.append(file_label)
        singlePath = os.path.join('C:\\Users\\n927287\\','Desktop','ML-master','ML-master','AiLearning-master','data','2.KNN',filename_test,testingFileList[i])
        testingMat[i,:] = img2vector(singlePath)
    return testingMat,ImgLabels_test,n

#prepare auxMat for metrics
auxMat = np.zeros((32,32))

for i in range(32):
    for j in range(32):
        if (i <16):
            auxMat[i][j] = exp(sqrt((i-15.5)**2 + (j-15.5) **2))
        else:
            auxMat[i][j] = -exp(sqrt((i-15.5)**2 + (j - 15.5) **2))

auxMat = auxMat.ravel()
trainingMat,training_labels,n_train = load_training()

def distance(vec1,vec2):
    return abs(sum(vec1*auxMat) - sum(vec2*auxMat))

def Classify_Num(inX,k):
#     testingMat,testing_labels,n_test = load_testing()
     distance_list = np.zeros(n_train)
     for i in range(n_train):
         distance_list[i] = distance(trainingMat[i,:],inX)
         
     classcount = {}
     sorteddistance = distance_list.argsort()
     for d in range(k):
         label = training_labels[sorteddistance[d]]
         classcount[label] = classcount.get(label,0) + 1
         sortedclasscount = sorted(classcount.items(), key= operator.itemgetter(1), reverse = True)
     return sortedclasscount[0][0]
         
def Testing():
    testingMat,actual_labels,n_test = load_testing()
    predicted = []
    count = 0
    for i in range(n_test):
        predicted.append(Classify_Num(testingMat[i,:],3))
        if (predicted[i] == actual_labels[i]):
            count +=1
    print(count)
    
Testing()     
     

