import math 
import time

import torch

import numpy as np
import torchvision
import pandas as pd

import torch.nn as nn

import seaborn as sns
from numpy import load 
from torch import optim

import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from tensorflow.keras import layers, losses
from torchvision import datasets, transforms

#%%

def batch_iterate(features, labels, dest_features, dest_labels, batch_size):
    # handy way I came up with to reduce the number of unnecessary memory allocations
    # in particular this copies to the GPU (and ensures contiguity) only once every batch iteration
    #assert features.shape[0] == labels.shape[0], 'features and labels must have same size along first axis'
    for batch_idx in range((features.shape[0]+batch_size-1)//batch_size):
        start_idx = batch_idx*batch_size
        tmp_features_batch = features[start_idx:(batch_idx+1)*batch_size]
        eff_batch_size = tmp_features_batch.shape[0]
        dest_features[:eff_batch_size] = tmp_features_batch
        if labels is None:
            yield start_idx, eff_batch_size, dest_features[:eff_batch_size], None
        else:
            tmp_labels_batch = labels[start_idx:(batch_idx+1)*batch_size]
            dest_labels[:eff_batch_size] = tmp_labels_batch
            yield start_idx, eff_batch_size, dest_features[:eff_batch_size], dest_labels[:eff_batch_size]
            
##########################################################################################################################################
#
#--------------------------------------------------------NN Architecture------------------------------------------------------------------#
#
###########################################################################################################################################

class NeuralNetwork(nn.Module):
    def __init__(self,inputSize=206,hiddenSize=103,outputSize=1):
        super(NeuralNetwork,self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.CalibParams = None 
        self.PerfScores = None 
        self.Losses = None 
        
        
        self.W1 = nn.Parameter(torch.normal(mean = 0.,std = 1., size=(self.inputSize,self.hiddenSize)))
        self.W2 = nn.Parameter(torch.normal(mean = 0.,std = 1.,size=(self.hiddenSize,self.outputSize)))
        
        self.b1 = nn.Parameter(torch.rand(1))
        self.b2 = nn.Parameter(torch.rand(1))
    
    
    
    def forward(self,X):
        self.y = torch.matmul(X,self.W1)+self.b1
        self.z = torch.relu(self.y)
        self.y2 = torch.matmul(self.z,self.W2)+self.b2
        #outputs = torch.sigmoid(self.y2).reshape(-1,1)
        outputs = self.y2
        return outputs
    
    
##########################################################################################################################################
#
#--------------------------------------------------------Training Function Parametric Estimation Of Density------------------------------#
#
##########################################################################################################################################

#----------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------ Necessary Function-------------------------------------------------------------- 
#----------------------------------------------------------------------------------------------------------------------------------------




def EstimateParamsDist(Scores):
    n = Scores.shape[0]
    mu = torch.mean(Scores)
    sigma = torch.sqrt(torch.mean((Scores-mu)**2) * (n)/(n-1))
    return mu,sigma

def pdfNormalDist(Params,x):
    mu,sigma = Params[0],Params[1]
    f = 1/(sigma*torch.sqrt(2*torch.tensor(math.pi))) * torch.exp(-(x-mu)**2/(2*sigma**2))
    return f

def DefineThreshold(fn,fo,limit,x):
    diff = fn - fo
    limitLeft = next(x for x, val in enumerate(abs(diff.detach().numpy())) if val>10**(-1))
    if limitLeft ==0 :
        limitLeft = 40
    if limitLeft >40:
        limitLeft = 40
    diffReverse = np.array(list(diff.detach().numpy())[::-1])
    limitRight = next(x for x,val in enumerate(abs(diffReverse)) if  val>10**-1)
    if limitRight ==0 :
        limitRight = 40
    if limitRight >40:
        limitRight = 40
    
    t_idx = torch.argmin(abs(diff[limitLeft:-limitRight])) + limitLeft
    tHat = x[t_idx]
    return diff,t_idx,tHat

def DefineThreshold(fn,fo,limit,x):
    diff = fn - fo
    
    t_idx = torch.argmin(abs(diff[limit:-limit])) + limit
    tHat = x[t_idx]
    return diff,t_idx,tHat
def DefineThreshold(fn,fo,limit,x):
    limitL,limitR = limit[0],limit[1]
    diff = fn - fo
    
    t_idx = torch.argmin(abs(diff[limitL:-limitR])) + limitL
    tHat = x[t_idx]
    return diff,t_idx,tHat






def DefineThreshold(fn,fo,limit,x):
    # Define Threshold to use for Raw time series
    diff = fn - fo
    
    t_idx = torch.argmin(abs(diff[limit:-limit])) + limit
    tHat = x[t_idx]
    return diff,t_idx,tHat

def DefineThreshold(fn,fo,limit,x):
    #limit 0.5 for training on residuals 
    # limit 1 for training on residuals reconstruction errors 
    diff = fn - fo
    limitLeft = next(x for x, val in enumerate(abs(diff.detach().numpy())) if val>0.1)
    if limitLeft ==0 :
        limitLeft = 40
    if limitLeft >40:
        limitLeft = 40
    diffReverse = np.array(list(diff.detach().numpy())[::-1])
    limitRight = next(x for x,val in enumerate(abs(diffReverse)) if  val>0.1)
    if limitRight ==0 :
        limitRight = 40
    if limitRight >40:
        limitRight = 40
    
    t_idx = torch.argmin(abs(diff[limitLeft:-limitRight])) + limitLeft
    tHat = x[t_idx]
    return diff,t_idx,tHat

def DefineThreshold(fn,fo,limit,x):
    # Define Threshold to use for Raw time series
    diff = fn - fo
    
    t_idx = torch.argmin(abs(diff[limit:-limit])) + limit
    tHat = x[t_idx]
    return diff,t_idx,tHat


def ComputeIntegralNew(fn,fo,x,t_idx):

    fIntegralNormal = fn[t_idx:]
    fIntegralOutlier = fo[:t_idx]
    
    
    
    step = x[1] - x[0]
    AUCDensityNormal = step * 0.5 * (torch.sum(fIntegralNormal[:-1] + fIntegralNormal[1:]))

    AUCDensityOutlier = step *0.5* (torch.sum(fIntegralOutlier[:-1] + fIntegralOutlier[1:]))
    
    return AUCDensityNormal,AUCDensityOutlier


def ComputeIntegralNewX(ParamsNormal,ParamsOutlier,t,Bins):
    xNormal = torch.tensor(torch.linspace(t,1,Bins))
    xOutlier = torch.tensor(torch.linspace(0,t,Bins))
    
    fIntegralNormal = pdfNormalDist(ParamsNormal,xNormal)
    fIntegralOutlier = pdfNormalDist(ParamsOutlier,xOutlier)
    
    
    
    stepNormal = xNormal[1] - xNormal[0]
    AUCDensityNormal = stepNormal * 0.5 * (torch.sum(fIntegralNormal[:-1] + fIntegralNormal[1:]))
    
    stepOutlier = xOutlier[1] - xOutlier[0]
    AUCDensityOutlier = stepOutlier *0.5* (torch.sum(fIntegralOutlier[:-1] + fIntegralOutlier[1:]))
    
    return AUCDensityNormal,AUCDensityOutlier
    
#----------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------ Training Function--------------------------------------------------------------- 
#----------------------------------------------------------------------------------------------------------------------------------------





def train(model,xTrain,yTrain,learningRate=0.001,epochs=10,batchsize=128,limit=40,Bins=100):
    Lvect = {'All':[],'BCE':[],'AUCn':[],'AUCo':[]}
    allL,bce,aucn,auco = 0.,0.,0.,0.
    acc,rec,prec = 0.,0.,0.
    perfScores = {'Acc':[],'Prec':[],'Rec':[]}
    CalibParams = {'Min':[],'Max':[],'That':[]}
    
    xFit = torch.tensor(np.linspace(0,1,Bins))
    criterion = nn.BCELoss()
    bar = trange(epochs)
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
    t_x_batch = torch.empty((batchsize, xTrain.shape[1]), dtype=torch.float32, device=None)
    t_y_batch = torch.empty((batchsize, yTrain.shape[1]), dtype=torch.float32, device=None)
    
    for epoch in bar:
        for _,_,xbatch,ybatch in batch_iterate(xTrain,yTrain,t_x_batch,t_y_batch,batchsize):
            
            # zero the parameter gradients 
            optimizer.zero_grad()

            # forward + backward + optimize
            S = model(xbatch)
            
            minVal,maxVal = min(S),max(S)
            
            S = (S-minVal)/(maxVal-minVal)
            
            SNormal = S[ybatch==0]
            SOutlier = S[ybatch==1]

            muHatNormal,sigmaHatNormal = EstimateParamsDist(SNormal)
            muHatOutlier,sigmaHatOutlier = EstimateParamsDist(SOutlier)

            
            f_xn_Sn_D = pdfNormalDist([muHatNormal,sigmaHatNormal],xFit)
            f_xn_So_D = pdfNormalDist([muHatOutlier,sigmaHatOutlier],xFit)
            
            diff,t_idx,tHat = DefineThreshold(f_xn_Sn_D,f_xn_So_D,limit,xFit)
            

            AUCDensityNormal, AUCDensityOutlier = ComputeIntegralNewX([muHatNormal,sigmaHatNormal],[muHatOutlier,sigmaHatOutlier],
                                                                      tHat,Bins)
            S = torch.tensor([1 if s>tHat.detach().numpy() else 0 for s in S.detach().numpy()])
            
            L = criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32)) + AUCDensityNormal + AUCDensityOutlier
            
            L.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            for _,eff,xbatch,ybatch in batch_iterate(xTrain,yTrain,t_x_batch,t_y_batch,batchsize):
                
                S = model(xbatch)
                minValTrack,maxValTrack = min(S),max(S)
                S = (S-minValTrack)/(maxValTrack-minValTrack)
                SNormal = S[ybatch==0]
                SOutlier = S[ybatch==1]
    
    
                muHatNormal,sigmaHatNormal = EstimateParamsDist(SNormal)
                muHatOutlier,sigmaHatOutlier = EstimateParamsDist(SOutlier)
                
    
                
                f_xn_Sn_D = pdfNormalDist([muHatNormal,sigmaHatNormal],xFit)
                f_xn_So_D = pdfNormalDist([muHatOutlier,sigmaHatOutlier],xFit)
                
                diff,t_idx,tHatTrack = DefineThreshold(f_xn_Sn_D,f_xn_So_D,limit,xFit)
    
                AUCDensityNormal, AUCDensityOutlier = ComputeIntegralNewX([muHatNormal,sigmaHatNormal],[muHatOutlier,sigmaHatOutlier],
                                                                      tHatTrack,Bins)
                
                bce += criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32))*eff
                aucn += AUCDensityNormal*eff
                auco += AUCDensityOutlier*eff
                allL += (criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32)) + 
                     AUCDensityNormal + AUCDensityOutlier)*eff

                yPred =torch.tensor([1 if s >tHatTrack else 0 for s in S])
                acc += accuracy_score(yPred,ybatch)*eff
                rec += recall_score(ybatch,yPred)*eff
                prec += precision_score(ybatch,yPred)*eff

                
            allL = allL/xTrain.shape[0]
            aucn = aucn/xTrain.shape[0]
            auco = auco/xTrain.shape[0]
            bce = bce/xTrain.shape[0]
            
            acc = acc/xTrain.shape[0]
            rec = rec/xTrain.shape[0]
            prec = prec/xTrain.shape[0]
            
            
            allL = float(allL.data.item())
            bar.set_postfix(loss=f'{allL :.6f}')    

            Lvect['All'].append(allL)
            Lvect['AUCn'].append(float(aucn.data.item()))
            Lvect['AUCo'].append(float(auco.data.item()))
            Lvect['BCE'].append(float(bce.data.item()))
            
            perfScores['Acc'].append(acc)
            perfScores['Prec'].append(prec)
            perfScores['Rec'].append(rec)
            
            if epoch %2 == 0:
                S = model(xTrain.type(torch.FloatTensor))
                minValGraphs,maxValGraphs = min(S),max(S)
                S = (S-minValGraphs)/(maxValGraphs-minValGraphs)
                SNormal = S[yTrain==0]
                SOutlier = S[yTrain==1]
    
    
                muHatNormal,sigmaHatNormal = EstimateParamsDist(SNormal)
                muHatOutlier,sigmaHatOutlier = EstimateParamsDist(SOutlier)
                
                f_xn_Sn_D = pdfNormalDist([muHatNormal,sigmaHatNormal],xFit)
                f_xn_So_D = pdfNormalDist([muHatOutlier,sigmaHatOutlier],xFit)
                
                diff,t_idx,tHatGraphs = DefineThreshold(f_xn_Sn_D,f_xn_So_D,limit,xFit)
                print(tHatGraphs)
    
                AUCDensityNormal, AUCDensityOutlier = ComputeIntegralNewX([muHatNormal,sigmaHatNormal],[muHatOutlier,sigmaHatOutlier],
                                                                      tHatGraphs,Bins)
                
                plt.subplot(1,2,1)
                plt.scatter(xFit.detach().numpy() ,f_xn_Sn_D.detach().numpy() ,label='NormalD',alpha=0.6)

                plt.scatter(xFit.detach().numpy() ,f_xn_So_D.detach().numpy() ,label='OutlierD',alpha=0.6)
                maxValGraph = max(max(f_xn_So_D),max(f_xn_Sn_D))

                plt.scatter(xFit.detach().numpy() ,abs(diff.detach().numpy() ),label='Diff',alpha=0.6)
                plt.vlines(tHatGraphs.detach().numpy() ,0,maxValGraph)
                plt.legend()


                plt.subplot(1,2,2)
                outputNormal = plt.hist(SNormal.detach().numpy(),label='Normal',bins=100,alpha=0.4)
                outputOutlier = plt.hist(SOutlier.detach().numpy(),label='Outlier',bins=100,alpha=0.4)
                plt.vlines(tHatGraphs,0,max(outputNormal[0]))
                plt.legend()
                plt.show()
        
        
        
    
    CalibParams['Min'] = minVal
    CalibParams['Max'] = maxVal
    CalibParams['That'] = tHat
    model.CalibParams = CalibParams 
    model.PerfScores = perfScores  
    model.Losses = Lvect 
    return Lvect,perfScores,CalibParams,diff

##########################################################################################################################################
#
#-------------------------------------------Training Function Non Parametric Estimation Of Density---------------------------------------#
#
##########################################################################################################################################

#----------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------ Necessary Function-------------------------------------------------------------- 
#----------------------------------------------------------------------------------------------------------------------------------------




pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

def GaussianKernel(x):
    return torch.exp(-x**2/2)/ torch.sqrt(2*pi)
def KDEGaussianKernel(x,X,h):
    x = x.reshape(-1,)
    X = X.reshape(-1,)
    res = []
    for xi in x:
        res.append(torch.mean(GaussianKernel((xi - X)/h),dim=0)/h)
    res = torch.tensor(res)
    return res 


def EpanechnikovKernel(x):
    K = []
    for xi in x:
        if torch.abs(xi) < torch.sqrt(torch.tensor(5.)):
            K.append(0.75 * (1 - 0.2*xi**2)/torch.sqrt(torch.tensor((5.))))
        else:
            K.append(0)
    K = torch.tensor(K)
    return K

def KDEEpanchKernel(x,X,h):
    x = x.reshape(-1,)
    X = X.reshape(-1,)
    res = []
    for xi in x:
        res.append(torch.mean(EpanechnikovKernel((xi - X)/h).type(torch.FloatTensor),dim=0)/h)
    res = torch.tensor(res)
    return res 

def ComputeIntegralKDE(SNormal,SOutlier,t,Bins,h):
    xNormal = torch.tensor(np.linspace(t,1,Bins))
    xOutlier = torch.tensor(np.linspace(0,t,Bins))
    
    fIntegralNormal = KDEGaussianKernel(xNormal,SNormal,h)
    fIntegralOutlier = KDEGaussianKernel(xOutlier,SOutlier,h)
    
    
    
    stepNormal = xNormal[1] - xNormal[0]
    AUCDensityNormal = stepNormal * 0.5 * (torch.sum(fIntegralNormal[:-1] + fIntegralNormal[1:]))
    
    stepOutlier = xOutlier[1] - xOutlier[0]
    AUCDensityOutlier = stepOutlier *0.5* (torch.sum(fIntegralOutlier[:-1] + fIntegralOutlier[1:]))
    
    return AUCDensityNormal,AUCDensityOutlier



#----------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------ Training Function--------------------------------------------------------------- 
#----------------------------------------------------------------------------------------------------------------------------------------





def trainKDE(model,xTrain,yTrain,learningRate=0.001,epochs=10,batchsize=128,limit=40,Bins=100,h=0.5):
    Lvect = {'All':[],'BCE':[],'AUCn':[],'AUCo':[]}
    allL,bce,aucn,auco = 0.,0.,0.,0.
    acc,rec,prec = 0.,0.,0.
    perfScores = {'Acc':[],'Prec':[],'Rec':[]}
    CalibParams = {'Min':[],'Max':[],'That':[]}
    
    xFit = torch.tensor(np.linspace(0,1,Bins))
    criterion = nn.BCELoss()
    bar = trange(epochs)
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
    t_x_batch = torch.empty((batchsize, xTrain.shape[1]), dtype=torch.float32, device=None)
    t_y_batch = torch.empty((batchsize, yTrain.shape[1]), dtype=torch.float32, device=None)
    
    for epoch in bar:
        for _,_,xbatch,ybatch in batch_iterate(xTrain,yTrain,t_x_batch,t_y_batch,batchsize):
            
            # zero the parameter gradients 
            optimizer.zero_grad()

            # forward + backward + optimize
            S = model(xbatch)
            
            minVal,maxVal = min(S),max(S)
            S = (S-minVal)/(maxVal-minVal)
            
            SNormal = S[ybatch==0]
            SOutlier = S[ybatch==1]

            #muHatNormal,sigmaHatNormal = EstimateParamsDist(SNormal)
            #muHatOutlier,sigmaHatOutlier = EstimateParamsDist(SOutlier)
            
            f_xn_Sn_D = KDEGaussianKernel(xFit,SNormal,h)
            f_xn_So_D = KDEGaussianKernel(xFit,SOutlier,h)
            
            #f_xn_Sn_D = pdfNormalDist([muHatNormal,sigmaHatNormal],xFit)
            #f_xn_So_D = pdfNormalDist([muHatOutlier,sigmaHatOutlier],xFit)
            
            diff,t_idx,tHat = DefineThreshold(f_xn_Sn_D,f_xn_So_D,limit,xFit)

            #AUCDensityNormal, AUCDensityOutlier = ComputeIntegralKDE(f_xn_Sn_D,f_xn_So_D,xFit,t_idx)
            AUCDensityNormal, AUCDensityOutlier = ComputeIntegralKDE(SNormal,SOutlier,t_idx,Bins,h)

            L = criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32)) + AUCDensityNormal + AUCDensityOutlier
            
            L.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            for _,eff,xbatch,ybatch in batch_iterate(xTrain,yTrain,t_x_batch,t_y_batch,batchsize):
                
                S = model(xbatch)
                minVal,maxVal = min(S),max(S)
                S = (S-minVal)/(maxVal-minVal)
                SNormal = S[ybatch==0]
                SOutlier = S[ybatch==1]
    
    
                
    
                
                f_xn_Sn_D = KDEGaussianKernel(xFit,SNormal,h)
                f_xn_So_D = KDEGaussianKernel(xFit,SOutlier,h)

                #f_xn_Sn_D = pdfNormalDist([muHatNormal,sigmaHatNormal],xFit)
                #f_xn_So_D = pdfNormalDist([muHatOutlier,sigmaHatOutlier],xFit)

                diff,t_idx,tHat = DefineThreshold(f_xn_Sn_D,f_xn_So_D,limit,xFit)

                #AUCDensityNormal, AUCDensityOutlier = ComputeIntegralKDE(f_xn_Sn_D,f_xn_So_D,xFit,t_idx)
                AUCDensityNormal, AUCDensityOutlier = ComputeIntegralKDE(SNormal,SOutlier,t_idx,Bins,h)

                bce += criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32))*eff
                aucn += AUCDensityNormal*eff
                auco += AUCDensityOutlier*eff
                allL += (criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32)) + 
                     AUCDensityNormal + AUCDensityOutlier)*eff

                yPred =torch.tensor([1 if s >tHat else 0 for s in S])
                acc += accuracy_score(yPred,ybatch)*eff
                rec += recall_score(ybatch,yPred)*eff
                prec += precision_score(ybatch,yPred)*eff

                
            allL = allL/xTrain.shape[0]
            aucn = aucn/xTrain.shape[0]
            auco = auco/xTrain.shape[0]
            bce = bce/xTrain.shape[0]
            
            acc = acc/xTrain.shape[0]
            rec = rec/xTrain.shape[0]
            prec = prec/xTrain.shape[0]
            
            
            allL = float(allL.data.item())
            bar.set_postfix(loss=f'{allL :.6f}')    

            Lvect['All'].append(allL)
            Lvect['AUCn'].append(float(aucn.data.item()))
            Lvect['AUCo'].append(float(auco.data.item()))
            Lvect['BCE'].append(float(bce.data.item()))
            
            perfScores['Acc'].append(acc)
            perfScores['Prec'].append(prec)
            perfScores['Rec'].append(rec)
            
            if epoch %2 == 0:
                S = model(xTrain.type(torch.FloatTensor))
                minVal,maxVal = min(S),max(S)
                S = (S-minVal)/(maxVal-minVal)
                SNormal = S[yTrain==0]
                SOutlier = S[yTrain==1]
    
    
                f_xn_Sn_D = KDEGaussianKernel(xFit,SNormal,h)
                f_xn_So_D = KDEGaussianKernel(xFit,SOutlier,h)

                #f_xn_Sn_D = pdfNormalDist([muHatNormal,sigmaHatNormal],xFit)
                #f_xn_So_D = pdfNormalDist([muHatOutlier,sigmaHatOutlier],xFit)

                diff,t_idx,tHat = DefineThreshold(f_xn_Sn_D,f_xn_So_D,limit,xFit)

                #AUCDensityNormal, AUCDensityOutlier = ComputeIntegralKDE(f_xn_Sn_D,f_xn_So_D,xFit,t_idx)
                AUCDensityNormal, AUCDensityOutlier = ComputeIntegralKDE(SNormal,SOutlier,t_idx,Bins,h)

                plt.subplot(1,2,1)
                plt.scatter(xFit.detach().numpy() ,f_xn_Sn_D.detach().numpy() ,label='NormalD',alpha=0.6)

                plt.scatter(xFit.detach().numpy() ,f_xn_So_D.detach().numpy() ,label='OutlierD',alpha=0.6)
                maxValGraph = max(max(f_xn_So_D),max(f_xn_Sn_D))

                plt.scatter(xFit.detach().numpy() ,abs(diff.detach().numpy() ),label='Diff',alpha=0.6)
                plt.vlines(tHat.detach().numpy() ,0,maxValGraph)
                plt.legend()


                plt.subplot(1,2,2)
                outputNormal = plt.hist(SNormal.detach().numpy(),label='Normal',bins=100,alpha=0.4)
                outputOutlier = plt.hist(SOutlier.detach().numpy(),label='Outlier',bins=100,alpha=0.4)
                plt.vlines(tHat,0,max(outputNormal[0]))
                plt.legend()
                plt.show()
        
        
        
    
    CalibParams['Min'] = minVal
    CalibParams['Max'] = maxVal
    CalibParams['That'] = tHat
    return Lvect,perfScores,CalibParams,diff






##########################################################################################################################################
#
#--------------------------------------------------------Evaluation functions -----------------------------------------------------------#
#
##########################################################################################################################################


def plotPerfThroughTraining(Lvect,perfScores):
    plt.subplot(1,4,1)
    plt.plot(np.arange(len(Lvect['All'])),Lvect['All'],marker='x',label='All')
    plt.legend()

    plt.subplot(1,4,2)

    plt.plot(np.arange(len(Lvect['BCE'])),Lvect['BCE'],marker='x',label='BCE')
    plt.legend()

    plt.subplot(1,4,3)

    plt.plot(np.arange(len(Lvect['AUCn'])),Lvect['AUCn'],marker='x',label='AUCNormal')
    plt.legend()

    plt.subplot(1,4,4)

    plt.plot(np.arange(len(Lvect['AUCo'])),Lvect['AUCo'],marker='x',label='AUCOutlier')
    plt.legend()
    plt.show()

    #plt.subplot(1,2,1)
    plt.plot(np.arange(len(perfScores['Acc'])),perfScores['Acc'],marker='x',label='Acc')
    plt.plot(np.arange(len(perfScores['Rec'])),perfScores['Rec'],marker='x',label='Rec')
    plt.plot(np.arange(len(perfScores['Prec'])),perfScores['Prec'],marker='x',label='Prec')
    plt.title('Performance Indicator')
    plt.legend()
    plt.show()
    
    
def evalPerf(model,X,Y,label,CalibParams):
    with torch.no_grad():
        S = model(X.type(torch.FloatTensor))
        minVal,maxVal = CalibParams['Min'],CalibParams['Max']
        S = (S - minVal)/(maxVal-minVal)
        Pred = torch.tensor([1 if s > CalibParams['That'] else 0 for s in S])
        acc = accuracy_score(Y,Pred)
        pre = precision_score(Y,Pred)
        f1 = f1_score(Y,Pred)
        rec = recall_score(Y,Pred)
        
        str_print = "-------------------Scores on " + label + " set-------------------"
        print(str_print)
        print(f'Accuracy : {acc: .4f}')
        print(f'Precision : {pre: .4f}')
        print(f'Recall : {rec: .4f}')
        print(f'F1-score:{f1: .4f}')

        print("\n")
        scores = [acc,pre,rec,f1]
        return S,scores 
    
def evalPerfCaliT(model,X,Y,label,CalibParams):
    with torch.no_grad():
        S = model(X.type(torch.FloatTensor))
        minVal,maxVal = CalibParams['Min'],CalibParams['Max']
        S = (S - minVal)/(maxVal-minVal)
        Pred = torch.tensor([1 if s > 0 else 0 for s in S])
        acc = accuracy_score(Y,Pred)
        pre = precision_score(Y,Pred)
        f1 = f1_score(Y,Pred)
        rec = recall_score(Y,Pred)
        
        str_print = "-------------------Scores on " + label + " set-------------------"
        print(str_print)
        print(f'Accuracy : {acc: .4f}')
        print(f'Precision : {pre: .4f}')
        print(f'Recall : {rec: .4f}')
        print(f'F1-score:{f1: .4f}')

        print("\n")
        scores = [acc,pre,rec,f1]
        return S,scores 

    

        
    



##########################################################################################################################################
#
#--------------------------------Training Function Parametric Estimation Of Density & threshold Calibration------------------------------#
#
##########################################################################################################################################

#----------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------ NN Architecture-------------------------------------------------------------- 
#----------------------------------------------------------------------------------------------------------------------------------------

class NeuralNetworkThreshold(nn.Module):
    def __init__(self,inputSize=206,hiddenSize=103,outputSize=1):
        super(NeuralNetworkThreshold,self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.CalibParams = None 
        self.PerfScores = None 
        self.Losses = None 
        
        
        self.W1 = nn.Parameter(torch.normal(mean = 0.,std = 1., size=(self.inputSize,self.hiddenSize)))
        self.W2 = nn.Parameter(torch.normal(mean = 0.,std = 1.,size=(self.hiddenSize,self.outputSize)))
        
        self.b1 = nn.Parameter(torch.rand(1))
        self.b2 = nn.Parameter(torch.rand(1))
        self.t = nn.Parameter(torch.rand(1))
    
    
    
    def forward(self,X):
        self.y = torch.matmul(X,self.W1)+self.b1
        self.z = torch.relu(self.y)
        self.y2 = torch.matmul(self.z,self.W2)+self.b2
        minVal,maxVal = min(self.y2),max(self.y2)
        outputs = (self.y2-minVal)/(maxVal - minVal)
        outputs = torch.relu(outputs -self.t)
        return outputs
#----------------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------ Training Function--------------------------------------------------------------- 
#----------------------------------------------------------------------------------------------------------------------------------------




def trainThreshold(model,xTrain,yTrain,learningRate=0.001,epochs=10,batchsize=128,Bins=100):
    Lvect = {'All':[],'BCE':[],'AUCn':[],'AUCo':[]}
    allL,bce,aucn,auco = 0.,0.,0.,0.
    acc,rec,prec = 0.,0.,0.
    perfScores = {'Acc':[],'Prec':[],'Rec':[]}
    CalibParams = {'Min':[],'Max':[],'That':[]}
    
    xFit = torch.tensor(np.linspace(0,1,Bins))
    criterion = nn.BCELoss()
    bar = trange(epochs)
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
    t_x_batch = torch.empty((batchsize, xTrain.shape[1]), dtype=torch.float32, device=None)
    t_y_batch = torch.empty((batchsize, yTrain.shape[1]), dtype=torch.float32, device=None)
    
    for epoch in bar:
        print(f't : {model.t}')
        for _,_,xbatch,ybatch in batch_iterate(xTrain,yTrain,t_x_batch,t_y_batch,batchsize):
            
            # zero the parameter gradients 
            optimizer.zero_grad()

            # forward + backward + optimize
            S = model(xbatch)
            
            minVal,maxVal = min(S),max(S)
            S = (S-minVal)/(maxVal-minVal)

            SNormal = S[ybatch==0]
            SOutlier = S[ybatch==1]

            muHatNormal,sigmaHatNormal = EstimateParamsDist(SNormal)
            muHatOutlier,sigmaHatOutlier = EstimateParamsDist(SOutlier)
            

            
            f_xn_Sn_D = pdfNormalDist([muHatNormal,sigmaHatNormal],xFit)
            f_xn_So_D = pdfNormalDist([muHatOutlier,sigmaHatOutlier],xFit)
            
            #diff,t_idx,tHat = DefineThreshold(f_xn_Sn_D,f_xn_So_D,limit,xFit)

            AUCDensityNormal, AUCDensityOutlier = ComputeIntegralNewX([muHatNormal,sigmaHatNormal],[muHatOutlier,sigmaHatOutlier],
                                                                      model.t.detach().numpy()[0],Bins)
            L = criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32)) + AUCDensityNormal + AUCDensityOutlier
            
            L.backward()
            optimizer.step()
            
        with torch.no_grad():
            model.eval()
            for _,eff,xbatch,ybatch in batch_iterate(xTrain,yTrain,t_x_batch,t_y_batch,batchsize):
                
                S = model(xbatch)
                minVal,maxVal = min(S),max(S)
                S = (S-minVal)/(maxVal-minVal)
                SNormal = S[ybatch==0]
                SOutlier = S[ybatch==1]

    
    
                muHatNormal,sigmaHatNormal = EstimateParamsDist(SNormal)
                muHatOutlier,sigmaHatOutlier = EstimateParamsDist(SOutlier)
                
    
                
                f_xn_Sn_D = pdfNormalDist([muHatNormal,sigmaHatNormal],xFit)
                f_xn_So_D = pdfNormalDist([muHatOutlier,sigmaHatOutlier],xFit)
                
                #diff,t_idx,tHat = DefineThreshold(f_xn_Sn_D,f_xn_So_D,limit,xFit)
    
                AUCDensityNormal, AUCDensityOutlier = ComputeIntegralNewX([muHatNormal,sigmaHatNormal],[muHatOutlier,sigmaHatOutlier],
                                                                      model.t.detach().numpy(),Bins)
                
                bce += criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32))*eff
                aucn += AUCDensityNormal*eff
                auco += AUCDensityOutlier*eff
                allL += (criterion(S.to(torch.float32).reshape(-1,1),ybatch.reshape(-1,1).to(torch.float32)) + 
                     AUCDensityNormal + AUCDensityOutlier)*eff

                yPred =torch.tensor([1 if s >model.t else 0 for s in S])
                acc += accuracy_score(yPred,ybatch)*eff
                rec += recall_score(ybatch,yPred)*eff
                prec += precision_score(ybatch,yPred)*eff

                
            allL = allL/xTrain.shape[0]
            aucn = aucn/xTrain.shape[0]
            auco = auco/xTrain.shape[0]
            bce = bce/xTrain.shape[0]
            
            acc = acc/xTrain.shape[0]
            rec = rec/xTrain.shape[0]
            prec = prec/xTrain.shape[0]
            
            
            allL = float(allL.data.item())
            bar.set_postfix(loss=f'{allL :.6f}')    

            Lvect['All'].append(allL)
            Lvect['AUCn'].append(float(aucn.data.item()))
            Lvect['AUCo'].append(float(auco.data.item()))
            Lvect['BCE'].append(float(bce.data.item()))
            
            perfScores['Acc'].append(acc)
            perfScores['Prec'].append(prec)
            perfScores['Rec'].append(rec)
            
            if epoch %2 == 0:
                
                diff = f_xn_Sn_D - f_xn_So_D
                plt.subplot(1,2,1)
                plt.scatter(xFit.detach().numpy() ,f_xn_Sn_D.detach().numpy() ,label='NormalD',alpha=0.6)

                plt.scatter(xFit.detach().numpy() ,f_xn_So_D.detach().numpy() ,label='OutlierD',alpha=0.6)
                maxValGraph = max(max(f_xn_So_D),max(f_xn_Sn_D))

                plt.scatter(xFit.detach().numpy() ,abs(diff.detach().numpy() ),label='Diff',alpha=0.6)
                plt.vlines(model.t.detach().numpy() ,0,maxValGraph)
                plt.legend()


                plt.subplot(1,2,2)
                outputNormal = plt.hist(SNormal.detach().numpy(),label='Normal',bins=100,alpha=0.4)
                outputOutlier = plt.hist(SOutlier.detach().numpy(),label='Outlier',bins=100,alpha=0.4)
                plt.vlines(model.t.detach().numpy(),0,max(outputNormal[0]))
                plt.legend()
                plt.show()
        
        
        
    
    CalibParams['Min'] = minVal
    CalibParams['Max'] = maxVal
    CalibParams['That'] = model.t.detach().numpy()[0]
    model.CalibParams = CalibParams 
    model.PerfScores = perfScores  
    model.Losses = Lvect 
    return Lvect,perfScores,CalibParams,diff
































