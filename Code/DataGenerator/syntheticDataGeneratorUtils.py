import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import listdir
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from os.path import isfile, join 
from scipy.stats import random_correlation





def GBMsimulator(seed, So, mu, sigma, Cov, T, N):
    """
    Parameters

    seed:   seed of simulation
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """

    np.random.seed(seed)
    dim = np.size(So)
    t = np.linspace(0., T, int(N))
    A = np.linalg.cholesky(Cov)
    S = np.zeros([dim, int(N)])
    S[:, 0] = So
    for i in range(1, int(N)):    
        drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
        Z = np.random.normal(0., 1., dim)
        diffusion = np.matmul(A, Z) * (np.sqrt(t[i] - t[i-1]))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S, t


def GBMsimulator(seed, So, mu, sigma, Cov, T, N):
    """
    Parameters

    seed:   seed of simulation
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """

    np.random.seed(seed)
    dim = np.size(So)
    #t = np.linspace(0., T, int(N))
    dt = T/N
    A = np.linalg.cholesky(Cov)
    S = np.zeros([dim, int(N*T)])
    S[:, 0] = So
    for i in range(1, int(N*T)):    
        drift = (mu - 0.5 * sigma**2) * dt
        Z = np.random.normal(0., 1., dim)
        diffusion = np.matmul(A, Z) * (np.sqrt(dt))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S,dt


def GenerateAnoLocation(AnoNumber,N,windowSize):
    listidx = np.arange(N,dtype=int)
    AnoLocation = np.empty(AnoNumber,dtype=int)
    AnoLocation[0] = np.random.choice(np.arange(N,dtype=int))
    for i in range(1,AnoNumber):

        if AnoLocation[i-1]>windowSize and AnoLocation[0]+windowSize<N :
            idxLeft = np.arange(0,AnoLocation[i-1]-windowSize,dtype=int)
            idxRight = np.arange(AnoLocation[i-1]+windowSize,N,dtype=int)
            listidx = np.concatenate([idxLeft,idxRight])
        elif AnoLocation[i-1]+windowSize>N:
            idxLeft = np.arange(0,AnoLocation[i-1]-windowSize,dtype=int)
            listidx = idxLeft
        elif AnoLocation[i-1]<windowSize :
            idxRight = np.arange(AnoLocation[i-1]+windowSize,N,dtype=int)
            listidx = idxRight
            
        AnoLocation[i] = np.random.choice(listidx)
        
    return AnoLocation

def GenerateAnoAmplitude(St=None,alpha=1,upBound=0.1,AnoLocation=None):
    StShocked = deepcopy(St)
    nAno = len(AnoLocation)
    AnoSign = random.choices([-1,1],k=nAno)
    meanSt = np.mean(StShocked)
    stdSt = np.std(StShocked)
    for i,AnoLoc in enumerate(AnoLocation):
        if AnoSign[i] ==1:
            
            #while StShocked[AnoLoc]<meanSt + alpha*stdSt:
            StShocked[AnoLoc] = StShocked[AnoLoc]*(1+np.random.uniform(0,upBound))

        elif AnoSign[i] == -1:
            #while StShocked[AnoLoc]>meanSt - alpha*stdSt:
            StShocked[AnoLoc] = StShocked[AnoLoc]*(1-np.random.uniform(0,upBound))

    return StShocked


def GenerateAnoAmplitude(St=None,alpha=1,upBound=0.1,AnoLocation=None):
    StShocked = deepcopy(St)
    nAno = len(AnoLocation)
    AnoSign = random.choices([-1,1],k=nAno)
    meanSt = np.mean(StShocked)
    stdSt = np.std(StShocked)
    for i,AnoLoc in enumerate(AnoLocation):

        StShocked[AnoLoc] = StShocked[AnoLoc]*(1+AnoSign[i]*np.random.uniform(0,upBound))

    return StShocked


def LabelTS(N,AnoLocation):
    Y = np.zeros(N)
    for i in AnoLocation:
        Y[i] = 1
    return Y 


def GenerateAnoTS(St=None,alpha=1,upBound=0.1,AnoNumber=2,windowSize=206):
    N = St.shape[0]
    AnoLocation = GenerateAnoLocation(AnoNumber,N,windowSize)
    StShocked = GenerateAnoAmplitude(St,alpha,upBound,AnoLocation)
    Y = LabelTS(N,AnoLocation)
    return StShocked,Y

def AnomalyGeneratorDF(StocksDf,AnoNumber,windowSize,alpha,upBound):
    n,d = StocksDf.shape
    dfTSwAnomaly = pd.DataFrame().reindex_like(StocksDf)
    dfTSLabel = pd.DataFrame().reindex_like(StocksDf)
    for i in range(n):
        StShocked,Y = GenerateAnoTS(StocksDf.iloc[i,:],alpha,upBound,AnoNumber,windowSize)
        dfTSwAnomaly.iloc[i,:] = StShocked
        dfTSLabel.iloc[i,:] = Y
    return dfTSwAnomaly,dfTSLabel

def GetAnoLocation(dfTSLabel):
    LocationAno = []
    N,d = dfTSLabel.shape
    for i in range(N):
        LocationAno.append(list(np.arange(d)[dfTSLabel.iloc[i,:]==1]))
    return LocationAno

def AnomalyTrueValue(StocksDf,LocationAno):
    TrueValueAno = []
    n = len(LocationAno)
    for i in range(n):
        TrueValueAno.append(list(StocksDf.iloc[i,LocationAno[i]]))
    return TrueValueAno




    

def rollingwindow1D(windowSize=206,lag=1,St=None,Y=None ):
    T = len(St)
    nt = int(np.floor((T-windowSize)/lag) +1  )
    StSlidded = np.empty((nt,windowSize))
    YSlidded = np.empty((nt,windowSize))
    for i in range(nt):
        if St[i:i+windowSize].shape[0] == windowSize:
            StSlidded[i,:] = St[i:i+windowSize]
            YSlidded[i,:] = Y[i:i+windowSize]
    return StSlidded,YSlidded


def rollingwindowDf(windowSize=206,lag=1,df=None,Y=None ):
    n,T = df.shape
    sliddedS = pd.DataFrame()
    sliddedY = pd.DataFrame()
    for i in range(n):
        StSlidded,YSlidded = rollingwindow1D(windowSize,lag,df.iloc[i,:],Y.iloc[i,:])
        dfS = pd.DataFrame(StSlidded)
        dfY = pd.DataFrame(YSlidded)
        dfS['Stock'] = 'S'+str(i)
        dfY['Stock'] = 'S'+str(i)
        
        sliddedS = pd.concat([sliddedS,dfS],axis=0)
        sliddedY = pd.concat([sliddedY,dfY],axis=0)

    sliddedS['TrackID'] = np.arange(sliddedS.shape[0])
    sliddedY['TrackID'] = np.arange(sliddedY.shape[0])
    sliddedY['NbAnomaly'] = np.sum(sliddedY.iloc[:,:-2],axis=1)
        
    return sliddedS,sliddedY

def dropMaxAno(SliddedTS=None,SliddedLabels=None,MaxAno=1):
    N,d = SliddedTS.shape
    idxList = np.arange(N)
    idxMaxAno = idxList[SliddedLabels['NbAnomaly']<=MaxAno] # idx of elements to keep
    SliddedTS = SliddedTS.iloc[idxMaxAno,:]
    SliddedLabels = SliddedLabels.iloc[idxMaxAno,:]
    return SliddedTS, SliddedLabels


def GenerateData(nStocks,Sparams,DriftParams,SigmaParams,T,Ntrain,Ntest,iSimul,
                 ContaminationParams, SliddingParams,UnderSampParams):
    
    
    seed = iSimul
    random.seed(seed)
    ###################################################################################################
    #
    # ------------------------------------------Paths Diffusion --------------------------------------
    #
    ###################################################################################################
    locS,sigmaS = Sparams
    lowMu,upMu = DriftParams
    lowSigma,upSigma = SigmaParams
    N = Ntrain + Ntest 
    
    S0 = np.random.normal(locS,sigmaS,nStocks)
    mu = np.random.uniform(lowMu,upMu,nStocks)
    sigma = np.random.uniform(lowSigma,upSigma,nStocks)
    
    if nStocks==1:
        corr = np.array([1])
    else :
        eig = np.random.uniform(0.1,1,nStocks)
        eig = eig/np.sum(eig)*nStocks
        corr = random_correlation.rvs(eig)
        
    Cov = np.tensordot(sigma,sigma,0)*corr
    
    stocks,time = GBMsimulator(seed,S0,mu,sigma,Cov,T,N)
    StocksDFTrain = pd.DataFrame(stocks[:,:Ntrain])
    StocksDFTest = pd.DataFrame(stocks[:,Ntrain:])
    
    DataOriginal= {'Train': StocksDFTrain, 'Test':StocksDFTest}
    
    ###################################################################################################
    #
    # ------------------------------------------Paths Contamination -----------------------------------
    #
    ###################################################################################################
    
    alpha,windowSize,upBound,AnoNumberTrain,AnoNumberTest = ContaminationParams
    
    dfTSwAnomalyTrain,dfTSLabelTrain = AnomalyGeneratorDF(StocksDFTrain,AnoNumberTrain,windowSize,alpha,upBound)
    dfTSwAnomalyTest,dfTSLabelTest = AnomalyGeneratorDF(StocksDFTest,AnoNumberTest,windowSize,alpha,upBound)
    
    ###################################################################################################
    #
    # ------------------------------------------Data Augmentation : Slidding window -------------------
    #
    ###################################################################################################
    
    windowSize,lag = SliddingParams
    
    dfTSLabelOriginalTrain = pd.DataFrame(np.zeros(StocksDFTrain.values.shape))
    sliddedSOriginalTrain,sliddedYOriginalTrain  = rollingwindowDf(windowSize=206,lag=1,df=StocksDFTrain,Y=dfTSLabelOriginalTrain)
    
    dfTSLabelOriginalTest = pd.DataFrame(np.zeros(StocksDFTest.values.shape))
    sliddedSOriginalTest,sliddedYOriginalTest  = rollingwindowDf(windowSize=206,lag=1,df=StocksDFTest,Y=dfTSLabelOriginalTest)
    
    sliddedSTrain,sliddedYTrain  = rollingwindowDf(windowSize=206,lag=1,df=dfTSwAnomalyTrain,Y=dfTSLabelTrain)
    sliddedSTest,sliddedYTest  = rollingwindowDf(windowSize=206,lag=1,df=dfTSwAnomalyTest,Y=dfTSLabelTest)
    
    SliddedTSTrain, SliddedLabelsTrain = dropMaxAno(sliddedSTrain,sliddedYTrain)
    SliddedTSTest, SliddedLabelsTest = dropMaxAno(sliddedSTest,sliddedYTest)
    
    
    DataOriginal= {'Train': [sliddedSOriginalTrain,sliddedYOriginalTrain], 'Test':[sliddedSOriginalTest,sliddedYOriginalTest]}

    ###################################################################################################
    #
    # ------------------------------------------Data UnderSampling -----------------------------------
    #
    ###################################################################################################
    ContaminationRate, = UnderSampParams
    
        # 1. UnderSampling Training DataSet to get balanced class 
        
    SliddedTSTrain['Label'] = SliddedLabelsTrain.loc[:,'NbAnomaly']
    XtrainReg = SliddedTSTrain[SliddedTSTrain['Label']==0]
    ytrainReg = SliddedLabelsTrain[SliddedLabelsTrain['NbAnomaly']==0]

    XtrainOut = SliddedTSTrain[SliddedTSTrain['Label']==1]
    ytrainOut = SliddedLabelsTrain[SliddedLabelsTrain['NbAnomaly']==1]


    nout = XtrainOut.shape[0]
    nreg = XtrainReg.shape[0]
    n = min(nout,nreg)
    RefClass = np.argmin([nout,nreg]) 
    
    if RefClass==1:
        idxToKeep = list(random.sample(list(XtrainOut['TrackID']),k=n))

        # Undersample Outlier class 
        XtrainOutUnder = XtrainOut[XtrainOut['TrackID'].isin(idxToKeep)]
        ytrainOutUnder = ytrainOut[ytrainOut['TrackID'].isin(idxToKeep)]

        XtrainRegUnder = XtrainReg
        ytrainRegUnder = ytrainReg
    else:
        # Undersample Regular Class 
        idxToKeep = list(random.sample(list(XtrainReg['TrackID']),k=n))

        XtrainRegUnder = XtrainReg[XtrainReg['TrackID'].isin(idxToKeep)]
        ytrainRegUnder = ytrainReg[ytrainReg['TrackID'].isin(idxToKeep)]

        XtrainOutUnder = XtrainOut
        ytrainOutUnder = ytrainOut
        
    XtrainUnder = pd.concat([XtrainOutUnder,XtrainRegUnder]).sort_values(by='TrackID')
    ytrainUnder = pd.concat([ytrainOutUnder,ytrainRegUnder]).sort_values(by='TrackID')
        
            # 2. UnderSampling Test DataSet to get umbalanced class
            
    SliddedTSTest['Label'] = SliddedLabelsTest.loc[:,'NbAnomaly']
    XtestReg = SliddedTSTest[SliddedTSTest['Label']==0]
    ytestReg = SliddedLabelsTest[SliddedLabelsTest['NbAnomaly']==0]

    XtestOut = SliddedTSTest[SliddedTSTest['Label']==1]
    ytestOut = SliddedLabelsTest[SliddedLabelsTest['NbAnomaly']==1]


    nout = XtestOut.shape[0]
    nreg = XtestReg.shape[0]
    n = int(ContaminationRate*nreg)
    
    idxToKeep = list(random.sample(list(XtestOut['TrackID']),k=n))
    XtestOutUnder = XtestOut[XtestOut['TrackID'].isin(idxToKeep)]
    ytestOutUnder = ytestOut[ytestOut['TrackID'].isin(idxToKeep)]
    
    XtestUnder = pd.concat([XtestOutUnder,XtestReg]).sort_values(by='TrackID')
    ytestUnder = pd.concat([ytestOutUnder,ytestReg]).sort_values(by='TrackID')
    
    DataIdentification ={'Train' : [XtrainUnder,ytrainUnder],'Test':[XtestUnder,ytestUnder]}
    
    
    ###################################################################################################
    #
    # ------------------------------------------Data for Location -----------------------------------
    #
    ###################################################################################################
    
    
    SliddedTSTrain['Label'] = SliddedLabelsTrain.loc[:,'NbAnomaly']
    XTrainOut = SliddedTSTrain[SliddedTSTrain['Label']==1].sort_values(by='TrackID')
    yTrainOut = SliddedLabelsTrain[SliddedLabelsTrain['NbAnomaly']==1].sort_values(by='TrackID')


    SliddedTSTest['Label'] = SliddedLabelsTest.loc[:,'NbAnomaly']
    XTestOut = SliddedTSTest[SliddedTSTest['Label']==1].sort_values(by='TrackID')
    yTestOut = SliddedLabelsTest[SliddedLabelsTest['NbAnomaly']==1].sort_values(by='TrackID')
    
    DataLocation ={'Train' : [XTrainOut,yTrainOut],'Test':[XTestOut,yTestOut]}
    
    return DataIdentification, DataLocation, DataOriginal



def GenerateData(nStocks,S0,mu,sigma,Cov,T,Ntrain,Ntest,iSimul,
                 ContaminationParams, SliddingParams,UnderSampParams):
    
    
    seed = iSimul
    random.seed(seed)
    N = Ntrain + Ntest
    ###################################################################################################
    #
    # ------------------------------------------Paths Diffusion --------------------------------------
    #
    ###################################################################################################
    
    
    stocks,time = GBMsimulator(seed,S0,mu,sigma,Cov,T,N)
    StocksDFTrain = pd.DataFrame(stocks[:,:Ntrain])
    StocksDFTest = pd.DataFrame(stocks[:,Ntrain:])
    
    DataOriginal= {'Train': StocksDFTrain, 'Test':StocksDFTest}
    
    ###################################################################################################
    #
    # ------------------------------------------Paths Contamination -----------------------------------
    #
    ###################################################################################################
    
    alpha,windowSize,upBound,AnoNumberTrain,AnoNumberTest = ContaminationParams
    
    dfTSwAnomalyTrain,dfTSLabelTrain = AnomalyGeneratorDF(StocksDFTrain,AnoNumberTrain,windowSize,alpha,upBound)
    dfTSwAnomalyTest,dfTSLabelTest = AnomalyGeneratorDF(StocksDFTest,AnoNumberTest,windowSize,alpha,upBound)
    
    ###################################################################################################
    #
    # ------------------------------------------Data Augmentation : Slidding window -------------------
    #
    ###################################################################################################
    
    windowSize,lag = SliddingParams
    
    dfTSLabelOriginalTrain = pd.DataFrame(np.zeros(StocksDFTrain.values.shape))
    sliddedSOriginalTrain,sliddedYOriginalTrain  = rollingwindowDf(windowSize=206,lag=1,df=StocksDFTrain,Y=dfTSLabelOriginalTrain)
    
    dfTSLabelOriginalTest = pd.DataFrame(np.zeros(StocksDFTest.values.shape))
    sliddedSOriginalTest,sliddedYOriginalTest  = rollingwindowDf(windowSize=206,lag=1,df=StocksDFTest,Y=dfTSLabelOriginalTest)
    
    sliddedSTrain,sliddedYTrain  = rollingwindowDf(windowSize=206,lag=1,df=dfTSwAnomalyTrain,Y=dfTSLabelTrain)
    sliddedSTest,sliddedYTest  = rollingwindowDf(windowSize=206,lag=1,df=dfTSwAnomalyTest,Y=dfTSLabelTest)
    
    SliddedTSTrain, SliddedLabelsTrain = dropMaxAno(sliddedSTrain,sliddedYTrain)
    SliddedTSTest, SliddedLabelsTest = dropMaxAno(sliddedSTest,sliddedYTest)
    
    
    DataOriginal= {'Train': [sliddedSOriginalTrain,sliddedYOriginalTrain], 'Test':[sliddedSOriginalTest,sliddedYOriginalTest]}

    ###################################################################################################
    #
    # ------------------------------------------Data UnderSampling -----------------------------------
    #
    ###################################################################################################
    ContaminationRate, = UnderSampParams
    
        # 1. UnderSampling Training DataSet to get balanced class 
        
    SliddedTSTrain['Label'] = SliddedLabelsTrain.loc[:,'NbAnomaly']
    XtrainReg = SliddedTSTrain[SliddedTSTrain['Label']==0]
    ytrainReg = SliddedLabelsTrain[SliddedLabelsTrain['NbAnomaly']==0]

    XtrainOut = SliddedTSTrain[SliddedTSTrain['Label']==1]
    ytrainOut = SliddedLabelsTrain[SliddedLabelsTrain['NbAnomaly']==1]


    nout = XtrainOut.shape[0]
    nreg = XtrainReg.shape[0]
    n = min(nout,nreg)
    RefClass = np.argmin([nout,nreg]) 
    
    if RefClass==1:
        idxToKeep = list(random.sample(list(XtrainOut['TrackID']),k=n))

        # Undersample Outlier class 
        XtrainOutUnder = XtrainOut[XtrainOut['TrackID'].isin(idxToKeep)]
        ytrainOutUnder = ytrainOut[ytrainOut['TrackID'].isin(idxToKeep)]

        XtrainRegUnder = XtrainReg
        ytrainRegUnder = ytrainReg
    else:
        # Undersample Regular Class 
        idxToKeep = list(random.sample(list(XtrainReg['TrackID']),k=n))

        XtrainRegUnder = XtrainReg[XtrainReg['TrackID'].isin(idxToKeep)]
        ytrainRegUnder = ytrainReg[ytrainReg['TrackID'].isin(idxToKeep)]

        XtrainOutUnder = XtrainOut
        ytrainOutUnder = ytrainOut
        
    XtrainUnder = pd.concat([XtrainOutUnder,XtrainRegUnder]).sort_values(by='TrackID')
    ytrainUnder = pd.concat([ytrainOutUnder,ytrainRegUnder]).sort_values(by='TrackID')
        
            # 2. UnderSampling Test DataSet to get umbalanced class
            
    SliddedTSTest['Label'] = SliddedLabelsTest.loc[:,'NbAnomaly']
    XtestReg = SliddedTSTest[SliddedTSTest['Label']==0]
    ytestReg = SliddedLabelsTest[SliddedLabelsTest['NbAnomaly']==0]

    XtestOut = SliddedTSTest[SliddedTSTest['Label']==1]
    ytestOut = SliddedLabelsTest[SliddedLabelsTest['NbAnomaly']==1]


    nout = XtestOut.shape[0]
    nreg = XtestReg.shape[0]
    n = int(ContaminationRate*nreg)
    
    idxToKeep = list(random.sample(list(XtestOut['TrackID']),k=n))
    XtestOutUnder = XtestOut[XtestOut['TrackID'].isin(idxToKeep)]
    ytestOutUnder = ytestOut[ytestOut['TrackID'].isin(idxToKeep)]
    
    XtestUnder = pd.concat([XtestOutUnder,XtestReg]).sort_values(by='TrackID')
    ytestUnder = pd.concat([ytestOutUnder,ytestReg]).sort_values(by='TrackID')
    
    DataIdentification ={'Train' : [XtrainUnder,ytrainUnder],'Test':[XtestUnder,ytestUnder]}
    
    
    ###################################################################################################
    #
    # ------------------------------------------Data for Location -----------------------------------
    #
    ###################################################################################################
    
    
    SliddedTSTrain['Label'] = SliddedLabelsTrain.loc[:,'NbAnomaly']
    XTrainOut = SliddedTSTrain[SliddedTSTrain['Label']==1].sort_values(by='TrackID')
    yTrainOut = SliddedLabelsTrain[SliddedLabelsTrain['NbAnomaly']==1].sort_values(by='TrackID')


    SliddedTSTest['Label'] = SliddedLabelsTest.loc[:,'NbAnomaly']
    XTestOut = SliddedTSTest[SliddedTSTest['Label']==1].sort_values(by='TrackID')
    yTestOut = SliddedLabelsTest[SliddedLabelsTest['NbAnomaly']==1].sort_values(by='TrackID')
    
    DataLocation ={'Train' : [XTrainOut,yTrainOut],'Test':[XTestOut,yTestOut]}
    
    return DataIdentification, DataLocation, DataOriginal
    
    
    

    
    
    


    
            
            






    
    
    


    
    
    
    
    
    
    













































                
    
    

