import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import listdir
from copy import deepcopy
import numpy.linalg as npl
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


def ProcessingData(DataLocation):
    X_traindf, y_traindf = DataLocation['Train']
    X_testdf, y_testdf = DataLocation['Test']
    
    Xtrain = np.array(X_traindf.iloc[:,:-4])
    Xtest = np.array(X_testdf.iloc[:,:-4])
    
    ytrain_loc = [list(y_traindf.iloc[i,:]).index(1)for i in range(y_traindf.shape[0]) ]
    ytest_loc = [list(y_testdf.iloc[i,:]).index(1) for i in range(y_testdf.shape[0]) ]
    
    return Xtrain,ytrain_loc,Xtest, ytest_loc

def dummyprediction(X):
    pred = np.argmax(abs(X),axis=1)
    return pred



def modelPCALocation(Xtrain,Xtest,k):
    model = PCA(n_components=k)
    model.fit(Xtrain)
    ntrain,d = Xtrain.shape
    ntest,d = Xtest.shape

    r_train_hat = model.inverse_transform(model.transform(Xtrain))
    r_test_hat = model.inverse_transform(model.transform(Xtest))


    norms_train = np.zeros((ntrain,d))
    norms_test = np.zeros((ntest,d))


    for i in tqdm(range(ntrain)):
        for j in (range(d)):
            norms_train[i,j] = abs(r_train_hat[i,j]-Xtrain[i,j])


    for i in tqdm(range(ntest)):
        for j in (range(d)):
            norms_test[i,j] = abs(r_test_hat[i,j]-Xtest[i,j])
    return norms_train, norms_test


def PCAprediction(X,norms):
    n,d = X.shape
    pred = [np.argmax(norms[i,:]) for i in range(n)]
    return pred

def EvaluationPred(yPred,yTrue):
    acc = accuracy_score(yTrue,yPred)
    rec = recall_score(y_true = yTrue,y_pred = yPred,average='weighted')
    prec = precision_score(y_true = yTrue,y_pred = yPred,average='weighted')
    f1 = f1_score(yTrue,yPred,average='weighted')
    return acc,rec,prec,f1

def printRes(Scores,setName):
    print('------------Scores on '+ str(setName)+ '-------------')
    print(f'Accuracy: {Scores[0]}')
    print(f'Recall: {Scores[1]}')
    print(f'Precision: {Scores[2]}')
    print(f'F1-score: {Scores[3]}\n')
    return 0



def idxAnoNotExtreme(idxExtreme,yloc):
    n = len(idxExtreme)
    idxNotExtreme = []
    idxIsExtreme = []
    for i in range(n):
        if idxExtreme[i] != yloc[i]:
            idxNotExtreme.append(i)
        else:
            idxIsExtreme.append(i)
    return idxNotExtreme,idxIsExtreme

def ProcessingNotExtremum(X,ydf,yloc):
    n,d = X.shape

    idxExtreme = np.argmax(np.array(X.iloc[:,:-4]),axis=1)
    idxNotExtreme,idxIsExtreme = idxAnoNotExtreme(idxExtreme,yloc)
    
    X['idx'] = np.arange(n)
    XNotExtreme = X[X['idx'].isin(idxNotExtreme)]
    
    ydf['idx'] = np.arange(n)
    yNotExtreme = ydf[ydf['idx'].isin(idxNotExtreme)]
    
    X = np.array(XNotExtreme.iloc[:,:-4])

    y_loc = [list(yNotExtreme.iloc[i,:-1]).index(1)for i in range(yNotExtreme.shape[0]) ]
    
    return X, y_loc

def print_scores (true_label,pred_label):
    print(f'Accuracy on Train : {accuracy_score(true_label,pred_label)}')
    print(f'Recall on Train : {recall_score(true_label,pred_label)}')
    print(f'Precision on Train : {precision_score(true_label,pred_label)}')
    print(f'f1-score on Train : {f1_score(true_label,pred_label)}\n')
    
    
    
    
    
  




    
    
    

    

    




    
                                      