import numpy as np
import pandas as pd
from copy import deepcopy 
import numpy.linalg as npl


def slicingTS(TSOriginal,Y_i,windowsize):
    n = len(TSOriginal)
    nbShortTS = n//windowsize
    sliced = np.zeros((nbShortTS+1,windowsize))
    y_s = np.zeros((nbShortTS+1,windowsize))
    for i in range(nbShortTS):
        sliced[i,:] = TSOriginal[i*windowsize:(i+1)*windowsize]
        y_s[i,:] = Y_i[i*windowsize:(i+1)*windowsize]
    sliced[i+1,:] = TSOriginal[-206:]
    y_s[i+1,:] = Y_i[-206:]
    return sliced,y_s




def Imputation(AnoTS,cleanTS,modelPCA,windowsize):
    N,d = AnoTS.shape
    stocksAnoImpuPCA = deepcopy(cleanTS)
    stocksAnoImpuBF = deepcopy(cleanTS)
    stocksAnoImpuLinear = deepcopy(cleanTS)
    stocksAnoImpuLinearPCA = deepcopy(cleanTS)
    t = np.arange(stocksAnoImpuPCA.shape[1])

    for i in range(N):
        TSOriginalAno = AnoTS.iloc[i,:].values
        TSOriginal= cleanTS.iloc[i,:].values
        idxAno = t[(AnoTS.iloc[i,:].values - cleanTS.iloc[i,:].values)!=0]
        Y_i = [l if l in (idxAno) else 0 for l in range(t.shape[0])]
        
        slicedAno,y_s = slicingTS(TSOriginalAno,Y_i,windowsize)


        reconsRaw = modelPCA.inverse_transform(modelPCA.transform(slicedAno))
        for k in range(slicedAno.shape[0]):
            for j,l in enumerate(y_s[k,:]):
            
                if l!=0 and l!=(d-1):
                    stocksAnoImpuPCA.iloc[i,int(l)] = reconsRaw[k,j]
                    stocksAnoImpuBF.iloc[i,int(l)] = stocksAnoImpuBF.iloc[i,int(l+1)]
                    stocksAnoImpuLinear.iloc[i,int(l)] = (cleanTS.iloc[i,int(l+1)] + cleanTS.iloc[i,int(l-1)])/2
                elif l== (d-1):
                    stocksAnoImpuPCA.iloc[i,int(l)] = reconsRaw[k,j]
                    stocksAnoImpuBF.iloc[i,int(l)] = stocksAnoImpuBF.iloc[i,int(l-1)]
                    stocksAnoImpuLinear.iloc[i,int(l)] = cleanTS.iloc[i,int(l-1)]
                    
    return stocksAnoImpuBF,stocksAnoImpuPCA,stocksAnoImpuLinear

def computeCov(StocksPaths,vol):
    corrmat = np.corrcoef((np.log((StocksPaths[:,1:]) ) - np.log(StocksPaths[:,:-1])))
    n = len(vol)
    cov=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            cov[i,j] = corrmat[i,j]*vol[i]*vol[j]
    return cov

def ErrorOnCov(Cov,CovTrue):
    return npl.norm(Cov-CovTrue,ord='fro')
def ErrorOnTS(TrueTS,ImputedTS):
    return npl.norm((TrueTS.values - ImputedTS.values),axis=1)
    

def EvalImpuCov(AnoTS,CleanTS,ImpuBF,ImpuPCA,ImpuLinear,CovTrue,vol):
    
    CovImpuBF = computeCov(ImpuBF.values,vol)
    CovImpuPCA = computeCov(ImpuPCA.values,vol)
    CovImpuLinear = computeCov(ImpuLinear.values,vol)
    CovCheck = computeCov(CleanTS.values,vol)
    CovAno = computeCov(AnoTS.values,vol)
    Errors =[ErrorOnCov(CovAno,CovTrue),ErrorOnCov(CovImpuPCA,CovTrue),
                 ErrorOnCov(CovImpuLinear,CovTrue),ErrorOnCov(CovImpuBF,CovTrue),
                 ErrorOnCov(CovCheck,CovTrue)]
    
    return Errors
    

def EvalImputation(AnoTS,CleanTS,ImpuBF,ImpuPCA,ImpuLinear,AnoNumber):
    ImputationError = pd.DataFrame(columns=np.concatenate([np.arange(20),['Imput.Method']]),
                                                           index=np.arange(4))

    
    ImputationError.iloc[0,:-1] = ErrorOnTS(CleanTS,ImpuBF)/np.sqrt(AnoNumber)
    ImputationError.iloc[1,:-1] = ErrorOnTS(CleanTS,ImpuPCA)/np.sqrt(AnoNumber)
    ImputationError.iloc[2,:-1] = ErrorOnTS(CleanTS,ImpuLinear)/np.sqrt(AnoNumber)
    ImputationError.iloc[3,:-1] = ErrorOnTS(CleanTS,AnoTS)/np.sqrt(AnoNumber)
    ImputationError.iloc[:,-1] = ['BF','PCA','Linear','Ano']
    return ImputationError

    

















