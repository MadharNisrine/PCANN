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

sys.path.append(r'C:\Users\nmadhar\Desktop\Conv_deep\ACP_Autoencoder\Code')
from customizedLossSynthetic import *

def define_df_norms(norms,set_):
    df_norms = pd.DataFrame(columns=['Norms','Label'],dtype=float)
    df_norms['Label'] = set_[:,-1]
    df_norms['Norms'] = norms
    return df_norms

def get_densities_estimation(sample,kernel,bandwidth):
    n_balance = int(sample.shape[0]/2)
    max_err_reconst = max(int(max(sample['Norms'])),1)
    
    X_plot = np.linspace(0, max_err_reconst,n_balance)[:, np.newaxis]
    X_normal = np.array(sample[sample['Label']==0]['Norms']).reshape(-1,1)
    X_outlier = np.array(sample[sample['Label']==1]['Norms']).reshape(-1,1)
    
    kde_normal  = KernelDensity(kernel=kernel,bandwidth=bandwidth).fit(X_normal)
    kde_outlier  = KernelDensity(kernel=kernel,bandwidth=bandwidth).fit(X_outlier)

    log_dens_normal = kde_normal.score_samples(X_plot)
    log_dens_outlier = kde_outlier.score_samples(X_plot)
    
    return X_plot, log_dens_normal, log_dens_outlier

def find_threshold(x,dens_outlier,dens_normal,precision):
    n = len(dens_outlier)
    found = False 
    i = 0
    threshold = None 
    while not found and i<n:
        if abs(dens_normal[i]-dens_outlier[i])<precision:
            threshold = x[i]
            found = True
        else :
            i +=1 
    return threshold, i
def labeling_reconst_score(threshold, reconst_error):
    label_pred = [1 if n>threshold else 0 for n in reconst_error]
    return label_pred

def apply_pca(k,train_set):
    model = PCA(n_components=k)
    model.fit(train_set)
    return model

def get_reconstruction_errors(model,set_):
    set_hat = model.inverse_transform(model.transform(set_))
    return npl.norm(set_hat-set_,axis=1,ord=2)


def ScoresComputations(TrueLabel, PredLabel):
    acc = accuracy_score(TrueLabel,PredLabel)
    prec = precision_score(TrueLabel,PredLabel,pos_label=1)
    recall = recall_score(TrueLabel,PredLabel)
    f1 = f1_score(TrueLabel,PredLabel)
    return [acc,prec,recall,f1]
def PrintDetectionScores(Scores,setName):
    print('------------Scores on '+ setName+'------------------')
    print('Accuracy on '+f' :  {np.round(Scores[0],4)}')  
    print('Precision on '+f' :  {np.round(Scores[1],4)}')  
    print('Recall on '+f' :  {np.round(Scores[2],4)}')  
    print('F1 on '+f' :  {np.round(Scores[3],4)}\n')  
def MinMaxNormalizer(dfNormsTable,minValue,maxValue):
    df = deepcopy(dfNormsTable)
    if maxValue==None and minValue==None:
        maxValue = np.max(df.iloc[:,0])
        minValue = np.min(df.iloc[:,0])
    df.iloc[:,0] = (np.array(df.iloc[:,0])-minValue)/(maxValue-minValue)
    return df,minValue,maxValue

def Standandizer(dfNormsTable,meanValue=None,StdValue=None):
    if meanValue == None and StdValue == None :
        meanValue = np.mean(dfNormsTable.iloc[:,0])
        StdValue = np.std(dfNormsTable.iloc[:,0])
    dfNormsTable.iloc[:,0] = (np.array(dfNormsTable.iloc[:,0])-meanValue)/(StdValue)
    return dfNormsTable,meanValue,StdValue


def fitPCA(Xtrain=None,k=20):
    model = PCA(n_components=k)
    model.fit(Xtrain)
    return model

def predictionCalibration(df_errors=None,kernel='epanechnikov',bandwidth=0.2,precision=10**(-4),X = None,y=None,showplot = False,s=0):
    
    X_plot, log_dens_normal, log_dens_outlier = get_densities_estimation(sample=df_errors,kernel=kernel,bandwidth=bandwidth)
    seuil,ix = find_threshold(X_plot[s:,0],np.exp(log_dens_outlier)[s:],np.exp(log_dens_normal)[s:],precision)
    ix += s
    if showplot :
        plt.plot(X_plot[:, 0], np.exp(log_dens_normal), '-k',label = 'Normal')
        plt.plot(X_plot[:,0],np.exp(log_dens_outlier),label='Outlier')
        plt.scatter(seuil,np.exp(log_dens_outlier)[ix],label='Threshold Out.')
        plt.scatter(seuil,np.exp(log_dens_normal)[ix],label='Threshold Reg.')
        plt.legend()
        plt.show()
        print(f"Calibrated Threshold : {seuil}")
        print(f"Precision : {np.exp(log_dens_outlier)[ix]-np.exp(log_dens_normal)[ix]}")

    return seuil

def PredictLabels(df_errors,seuil):
    return labeling_reconst_score(threshold=seuil, reconst_error=df_errors['Norms'])


######################################################################
#
#---------------------------------Plots-------------------------------
#
#######################################################################

    
def plotDistAnoScores(Scores,TrueLabels,That,name,save=False):
    ScoresNormal = Scores[TrueLabels==0]
    ScoresOutliers = Scores[TrueLabels==1]
    
    sns.distplot(ScoresNormal,label='Normal',color='black',bins=20,norm_hist=True)
    sns.distplot(ScoresOutliers,label='Outliers',color = 'red',bins=20,norm_hist=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.vlines(That,0,8)
    plt.legend()
    if save :
        plt.savefig(name)
    plt.show()
    
def plotDistAnoScores(Scores,TrueLabels,That,plotThreshold=False,maxThreshold=10):
    ScoresNormal = Scores[TrueLabels==0]
    ScoresOutliers = Scores[TrueLabels==1]
    
    sns.distplot(ScoresNormal,label='Normal',color='black',bins=20,norm_hist=True)
    sns.distplot(ScoresOutliers,label='Outliers',color = 'red',bins=20,norm_hist=True)
    if plotThreshold:
        plt.vlines(That,0,maxThreshold,linestyles='dashdot',color='darkred')
    plt.ylabel('Density',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.legend()  

    
######################################################################
#
#---------------------------------Compute AUC-------------------------
#
#######################################################################

    
    
def ComputeIntegralKDEAUC(SNormal,SOutlier,t,Bins,h):
    xNormal = torch.tensor(np.linspace(t,SNormal.max().numpy(),Bins))
    xOutlier = torch.tensor(np.linspace(SOutlier.min().numpy(),t,Bins))
    
    fIntegralNormal = KDEGaussianKernel(xNormal,SNormal,h)
    fIntegralOutlier = KDEGaussianKernel(xOutlier,SOutlier,h)
    
    
    
    stepNormal = xNormal[1] - xNormal[0]
    AUCDensityNormal = stepNormal * 0.5 * (torch.sum(fIntegralNormal[:-1] + fIntegralNormal[1:]))
    
    stepOutlier = xOutlier[1] - xOutlier[0]
    AUCDensityOutlier = stepOutlier *0.5* (torch.sum(fIntegralOutlier[:-1] + fIntegralOutlier[1:]))
    
    return AUCDensityNormal,AUCDensityOutlier


###############################################################
#
#-----------------EvalPerf Baseline AD Algorithms ---------
#
#############################################################


def redefine_if_lof_pred(y_pred_clf):
    y_pred_ = []
    for i in y_pred_clf:
        if i == -1:
            y_pred_.append(1)
        else : 
            y_pred_.append(0)
    return y_pred_

def evalalg(yTrue,yPred):
    return accuracy_score(yTrue,yPred),recall_score(yTrue,yPred),precision_score(yTrue,yPred,pos_label=1),f1_score(yTrue,yPred)

def redefine_pred_DBSCAN(ypred):
    for i,pred in enumerate(ypred):
        if pred==-1:
            ypred[i]=1
        else : 
            ypred[i] = 0
    return ypred

def print_scores(y,ypred):
    print(f'Accuracy : {accuracy_score(y,ypred)}')

    print(f'Recall  : {recall_score(y,ypred)}')

    print(f'Precsion : {precision_score(y,ypred)}')

    print(f'F1-score  : {f1_score(y,ypred)}')
    
###############################################################
#
#-----------------latent space dimension selection  ---------
#
#############################################################

    
def EstimationDensityNonParametric(KernelName = 'Gaussian',ParamsKernel = 0.01,samples = None,xeval=None):
    y = np.zeros_like(xeval)
    h = ParamsKernel

    if KernelName == 'Gaussian':
        
        Params = [0.0,1.0]
        
        for i,x in enumerate(xeval) :
            xs = (x - samples)/h
            y[i] = np.mean(norm.pdf(xs,loc=0.0,scale=1.))/h
    if KernelName =='Epan':
        
        
        for i,x in enumerate(xeval) :
            xs = (x - samples)/h
            y[i] = torch.mean((3/4)*(1-xs**2)*torch.relu(1-torch.abs(xs))/(1-torch.abs(xs)))/h
        
    return y

    
def empirThrehsoldNaive(NaiveScore,y,prec=10**(-2),plot=True):
    SOutlier = NaiveScore[y==1]
    SNormal = NaiveScore[y==0]

    xevalOutlier = np.linspace(SNormal.min(),SNormal.max(),100)
    xevalNormal = np.linspace(SNormal.min(),SNormal.max(),100)

    yOutlier = EstimationDensityNonParametric(KernelName = 'Gaussian',ParamsKernel = 0.2,samples = SOutlier,xeval=xevalOutlier)
    yNormal = EstimationDensityNonParametric(KernelName = 'Gaussian',ParamsKernel = 0.2,samples = SNormal,xeval=xevalNormal)
    diff = abs(yNormal-yOutlier)<prec
    t = xevalOutlier[diff][0]
    
    if plot:
        plt.subplot(1,2,1)
        plt.scatter(xevalNormal,yNormal,label='Normal')
        plt.scatter(xevalOutlier,yOutlier,label='Outlier')
        plt.vlines(t,0,0.4)
        plt.legend()
    return t













































