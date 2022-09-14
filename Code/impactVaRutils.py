import numpy as np
import pylab as P
import pandas as pd 

import seaborn as sns 
from matplotlib import cm
import numpy.linalg as npl
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

import torch
import sys 

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import numpy.linalg as npl
from decomp_utils import *
from numpy import load,save 
from scipy.stats import norm
import matplotlib.pyplot as plt
from copy import deepcopy 

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

plt.rcParams["figure.figsize"] = (16,8)


def contamImputation(contamParams,s):
    n = s.shape[0]
    upBound,contamRate = contamParams
    nAno = int(contamRate*n)
    sAno = deepcopy(s)
    sImpFF = deepcopy(s)
    sImpBF = deepcopy(s)

    posAno = random.choices(list(np.arange(1,n-1)),k=nAno)
    AnoSign = random.choices([-1,1],k=nAno)
    for AnoLoc in posAno:
        sAno[AnoLoc] = s[AnoLoc]*(1+np.random.uniform(0,upBound))
        sImpFF[AnoLoc] = sImpFF[AnoLoc-1]
        sImpBF[AnoLoc] = sImpBF[AnoLoc+1]
    return sAno,sImpFF,sImpBF,posAno


def idxAnonotInTest(TRUELOC,trueLoc):
    idximputTrue = []
    for i in TRUELOC:
        if i not in trueLoc:
            idximputTrue.append(i)
    return idximputTrue


def returnLOC(yLabel):
    TRUELOC = []
    for i,l in enumerate(yLabel):
        if l==1:
            TRUELOC.append(i)
    return TRUELOC


def getIdxAnomdf(n,d):
    dflabelidx = pd.DataFrame(columns=[str(i) for i in range(d)],index = np.arange(n))
    for i in range(n):
        
        dflabelidx.iloc[i,:] = np.arange(d)
    return dflabelidx


def idxfromSliceToOriginal(yslice,ysliceidx):
    idx2impute = []
    for k in range(len(yslice)):
        idx2impute.append(int(ysliceidx[k][yslice[k]]))
    return np.array(idx2impute)

def ImputeAfterAnoLocation(X,AnoLoc):
    n = len(AnoLoc)
    d = len(X)
    XBF = deepcopy(X)
    XFF = deepcopy(X)
    for i in range(n):
        idxAno = int(AnoLoc[i])
        if (idxAno!=d-1) & (idxAno!=0) :
            XBF[idxAno] = XBF[idxAno+1]
            XFF[idxAno] = XFF[idxAno-1]
        elif (AnoLoc[i]==d):
            XBF[idxAno] = XBF[idxAno-1]
            XFF[idxAno] = XFF[idxAno-1]
        elif (AnoLoc[i]==0):
            XBF[idxAno] = XBF[idxAno+1]
            XFF[idxAno] = XFF[idxAno+1]
            
    return XBF,XFF


def computeCov(StocksPaths,vol):
    corrmat = np.corrcoef((np.log((StocksPaths[:,1:]) ) - np.log(StocksPaths[:,:-1])))
    n = corrmat.shape[0]
    cov=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            cov[i,j] = corrmat[i,j]*vol[i]*vol[j]
    return cov
def VaRonReturns(mu,Sigma,alpha):
    n = len(mu)
    w = np.ones(n)
    wSigmaw = np.dot(w.T,np.dot(Sigma,w))
    print(wSigmaw)
    wmu = np.dot(w.T,mu)
    print(wmu)
    return norm.ppf(1-alpha) * np.sqrt(wSigmaw) + wmu

def error(pred,true,typeError='Absolute'):
    if typeError=='Absolute':
        err = abs(pred-true)
    else:
        err = abs(pred-true)/true
    return err

def getIdxAnomdf(n,d):
    dflabelidx = pd.DataFrame(columns=[str(i) for i in range(d)],index = np.arange(n))
    for i in range(n):
        
        dflabelidx.iloc[i,:] = np.arange(d)
    return dflabelidx

def idxfromSliceToOriginal(yslice,ysliceidx):
    idx2impute = []
    for k in range(len(yslice)):
        idx2impute.append(int(ysliceidx[k][yslice[k]]))
    return np.array(idx2impute)

def returnLOC(yLabel):
    TRUELOC = []
    for i,l in enumerate(yLabel):
        if l==1:
            TRUELOC.append(i)
    return TRUELOC

def idxAnonotInTest(TRUELOC,trueLoc):
    idximputTrue = []
    for i in TRUELOC:
        if i not in trueLoc:
            idximputTrue.append(i)
    return idximputTrue


