# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:21:16 2019

@author: n927287
"""
import os
import numpy as np
from math import *
import matplotlib.pyplot as plt
import random

def load():
    dataMat = []
    label = []
    filename = "TestSet.txt"
    filepath = os.path.join('C:\\Users\\n927287\\','Desktop','ML-master','ML-master','AiLearning-master','data','5.Logistic',filename)
    fr = open(filepath)
    for lines in fr.readlines():
        vector = lines.strip().split()
        dataMat.append([1.0,float(vector[0]),float(vector[1])])
        label.append(int(vector[2]))
    return dataMat, label

def sigmoid(inX):
    
    return 2 * 1.0/(1 + np.exp(-2*inX)) - 1


def gradascent(data,labels):
    
    datamatrix = np.mat(data) # 100*3 numpy matrix
    labelmatrix = np.mat(labels).transpose()    #100*1 numpy matrix
    m,n = np.shape(datamatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    
    for k in range(maxCycles):
        h = sigmoid(datamatrix * weights) # 100 * 1 matrix
        error = (labelmatrix - h) # 100 * 1 matrix
    
        weights = weights+ alpha * datamatrix.transpose() * error
    return np.array(weights)

def plotbestfit(dataArr,labelMat,weights):
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30, c = 'red', marker = 's')
    ax.scatter(xcord2,ycord2,s=30,c= 'green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X');plt.ylabel('Y')
    plt.show()
    
def testLR():
    dataMat,labelMat = load()
    dataarr = np.array(dataMat)
    weights = stocGradAscent1(dataMat,labelMat)
    plotbestfit(dataarr,labelMat,weights)
    
def stocGradAscent0(dataMatrix,classLabels):
        m,n = np.shape(dataMatrix)
        alpha = 0.01
        weights = np.ones(n)
        for i in range(m):
            
            h = sigmoid(sum(dataMatrix * weights))
            error = classLabels[i] - h
            print(weights, "*"*10,dataMatrix[i],"*"*10,error)
            weights = weights + alpha * error * dataMatrix[i]
        return weights

def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    dataMatrix = np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    
    for j in range(numIter):
        
        dataIndex = list(range(m))
        for i in range(m):
            #Alpha is descending.
            alpha = 4/(1 + j + i) + 0.0001
            
            randIndex = int(random.uniform(0,len(dataIndex)))
            
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))
            
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del(dataIndex[randIndex])
    return weights
            
        