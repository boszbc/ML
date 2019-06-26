# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:31:04 2019

@author: n927287
"""
import numpy as np

def loadDataSet():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataset):
    wordlist = set([])
    for ob in dataset:
        wordlist = wordlist | set(ob)
    return list(wordlist)
    
def setword2vec(inX,vocablist):
    returnvec = [0] * len(vocablist)
    
    for word in inX:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else:
            print(word," is not in vocabulary")
    return returnvec

data,labels = loadDataSet()
vocabset = createVocabList(data)

def datatotrainmat(dataset,vocablist):
    returnvec = []
    for ob in dataset:
        returnvec.append(setword2vec(ob,vocablist))
    return returnvec
        
def trainNB(trainMat,TrainLabel):
    numofdocs = len(trainMat)
    numofwords = len(trainMat[0])
    
    p_1 = TrainLabel.count(1)/numofdocs

    
    p1Num = np.ones(numofwords)
    p2Num = np.ones(numofwords)
    
    for i in range(numofdocs):
        if TrainLabel[i] == 1:
            p1Num += trainMat[i]
        else:
            p2Num+= trainMat[i]
    pw_1 = log(p1Num/(sum(p1Num)-12))
    pw_2 = log(p2Num/(sum(p2Num)-12))

    return pw_1,pw_2,p_1
    