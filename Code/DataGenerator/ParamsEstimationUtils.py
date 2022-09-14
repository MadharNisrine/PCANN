import numpy as np
import pylab as P
import pandas as pd
import seaborn as sns 
from matplotlib import cm
import numpy.linalg as npl
from scipy.stats import norm
from copy import deepcopy

def diffusePath(T,M,N,mu,sigma,So):
    dt = T/M
    # Normal random variables to simulated the Brownian motion
    Z = np. random . normal ( size =[N, M])
    # Initialization of the share price paths
    St = np. zeros ([N, M +1])
    St [: ,0] = So
    # Simulation of share price paths ( path length equal to M)
    for i in range (M):
        St [:,i +1] = St [:,i]* np.exp ((mu - sigma **2/2)* dt + sigma *np. sqrt (dt )* Z[:,i])
    return St

def estimateParams(ParamsDiff):
    # diffuse a stock path 
    # estimate the mean and the variance form this path
    S0,T,mu,sigma,M = ParamsDiff
    s = np.zeros(M+1)
    Z = np.random.normal(size=M)

    dt = T/M
    s[0] = 1
    #s[1:] = np.exp(((mu-sigma**2/2)*dt + sigma*np.sqrt(dt)*Z).cumsum())
    s = diffusePath(T,M,1,mu,sigma,S0).reshape(-1,)
    df_stock = pd.DataFrame()
    df_stock["stock"] = s
    logRet = np.log(df_stock).diff().dropna().values
    muHat = logRet.mean()
    sigmaHat = logRet.std()
    return s,muHat,sigmaHat


def ParamsEstimation(S):
    # estimate the mean and the variance form this path
    df_stock = pd.DataFrame()
    df_stock["stock"] = S
    logRet = np.log(df_stock).diff().dropna().values
    muHat = logRet.mean()
    sigmaHat = logRet.std()
    return muHat,sigmaHat


def VaR(ConfLevel,mu,std):
    return np.exp(norm.ppf(ConfLevel)*std+ mu)




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
    return sAno,sImpFF,sImpBF



