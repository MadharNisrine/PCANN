import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import Series
import numpy.linalg as npl
from numpy import load,save 
from random import randrange
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL


def apply_decomp_3d(X):
    ''' Apply decomposition to X according to STL algorithm
    inputs:
        - X : (nxd)-DataFrame with n-time series of length d
    returns:
        - decomp : 3-channel array object
                    - 1st channel : being the trend 
                    - 2nd channel : seasonality
                    - 3rd channel : residual
         - err : vector of errors between observed TS and reconstructed TS'''
    n,d = X.shape
    decomp_ = np.zeros((n,d,3))
    err = np.zeros(n)
    for i in tqdm(range(n)):
        x_serie = pd.Series(X[i,:],index=pd.date_range('1-1-2000',periods=206,freq='D'),name='TS')
        stl_ = STL(x_serie,seasonal=11)
        res_ = stl_.fit()
        decomp_[i,:,0] = res_.trend
        decomp_[i,:,1]= res_.seasonal
        decomp_[i,:,2] = res_.resid
        err[i] = npl.norm(x_serie-(res_.trend+res_.seasonal+res_.resid))
    return decomp_,err

