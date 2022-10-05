
import pickle
import torch 
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pandas import Series
import numpy.linalg as npl
from numpy import load,save 
from random import randrange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from statsmodels.tsa.seasonal import seasonal_decompose


def convertAnoScoreToLabel(AnoScore,CalibParams):
    Pred = torch.tensor([1 if s > CalibParams['That'] else 0 for s in AnoScore])
    return Pred


def SelectTSAno(set_='Train',DataID=None,DataOriginal=None):
    DataID[set_][0] = DataID[set_][0].reset_index(drop=True)
    DataID[set_][1] = DataID[set_][1].reset_index(drop=True)
    DataOriginal[set_][0] = DataOriginal[set_][0].reset_index(drop=True)
    
    dataIDAnoOnly = DataID[set_][0][DataID[set_][1].loc[:,'NbAnomaly']==1]
    LabelsIDAnoOnly = DataID[set_][1][DataID[set_][1].loc[:,'NbAnomaly']==1]
    TrackidAnoID = DataID[set_][1][DataID[set_][1].loc[:,'NbAnomaly']==1].loc[:,'TrackID'].values
    dataIDwoAno = DataOriginal[set_][0][DataOriginal[set_][0].iloc[:,-1].isin(TrackidAnoID)]
    Labels_Loc = np.argmax(LabelsIDAnoOnly.iloc[:,:-3].values,axis=1)
    
    return dataIDwoAno,dataIDAnoOnly,TrackidAnoID,Labels_Loc



def SelectPredAno(ModelPred,CalibParams,TrackidAll,TrackidAnoID,step='ID'):
    if step=='ID':
        Pred = convertAnoScoreToLabel(ModelPred,CalibParams)
    else :
        Pred = ModelPred
    PredLabels = pd.DataFrame(Pred)
    PredLabels['TrackID'] = TrackidAll
    PredLabelsAno = PredLabels[PredLabels['TrackID'].isin(TrackidAnoID)].iloc[:,0].values
    return PredLabelsAno


def summaryShockAmplitude(dataIDwoAno,dataIDAnoOnly,Labels_Loc):
    nContaminatedTS = dataIDAnoOnly.shape[0]
    amplitudeAno = np.zeros(nContaminatedTS)

    for i in range(nContaminatedTS):
         amplitudeAno[i] = abs(dataIDAnoOnly.iloc[i,Labels_Loc[i]] / dataIDwoAno.iloc[i,Labels_Loc[i]] - 1)

    ampliAnoSummary = pd.DataFrame(amplitudeAno).describe()
    minShocks = ampliAnoSummary.loc['min'][0]
    q25Shocks = ampliAnoSummary.loc['25%'][0]
    q50Shocks = ampliAnoSummary.loc['50%'][0]
    q75Shocks = ampliAnoSummary.loc['75%'][0]
    maxShocks = ampliAnoSummary.loc['max'][0]
    
    idxSTAnoGroupsTrue = {'Min-q25':[],'q25-q50':[],'q50-q75':[],'q75-max':[]}
    for i,shock in enumerate(amplitudeAno):
        if shock<=q25Shocks:
            idxSTAnoGroupsTrue['Min-q25'].append(dataIDAnoOnly.iloc[i,-1])
        elif shock>q25Shocks and shock<=q50Shocks:
            idxSTAnoGroupsTrue['q25-q50'].append(dataIDAnoOnly.iloc[i,-1])
        elif shock>q50Shocks and shock<=q75Shocks:
            idxSTAnoGroupsTrue['q50-q75'].append(dataIDAnoOnly.iloc[i,-1])
        elif shock>q75Shocks and shock<=maxShocks:
            idxSTAnoGroupsTrue['q75-max'].append(dataIDAnoOnly.iloc[i,-1])
            
    AnoShocksRepartitiondf = pd.DataFrame(columns=["Min-25%","25%-50%","50%-75%","75%-Max"],index=['# TS'])
    AnoShocksRepartitiondf.loc["# TS",:] = [len(idxSTAnoGroupsTrue['Min-q25']),
                                     len(idxSTAnoGroupsTrue['q25-q50']),
                                     len(idxSTAnoGroupsTrue['q50-q75']),
                                     len(idxSTAnoGroupsTrue['q75-max'])]
    return amplitudeAno,AnoShocksRepartitiondf


def summaryPredContaminated_wrt_shocksAmplitudeRange(amplitudeAno,PredLabelsAno,Labels_Loc,step='ID'):
    if step == 'ID':
        Labels_Loc = np.ones(len(PredLabelsAno))
    ampliAnoSummary = pd.DataFrame(amplitudeAno).describe()
    minShocks = ampliAnoSummary.loc['min'][0]
    q25Shocks = ampliAnoSummary.loc['25%'][0]
    q50Shocks = ampliAnoSummary.loc['50%'][0]
    q75Shocks = ampliAnoSummary.loc['75%'][0]
    maxShocks = ampliAnoSummary.loc['max'][0]
    emptyDf = np.zeros((2,4))
    idxSTAnoGroupsPredDF = pd.DataFrame(emptyDf,columns=['Min-q25','q25-q50','q50-q75','q75-max'],index=['Contaminated','UnContaminated'])
    for i,shock in enumerate(amplitudeAno):
        if shock<=q25Shocks:
            if PredLabelsAno[i] == Labels_Loc[i]:
                idxSTAnoGroupsPredDF.loc['Contaminated','Min-q25'] += 1
            else:
                idxSTAnoGroupsPredDF.loc['UnContaminated','Min-q25'] += 1

        elif shock>q25Shocks and shock<=q50Shocks:
            if PredLabelsAno[i] == Labels_Loc[i]:
                idxSTAnoGroupsPredDF.loc['Contaminated','q25-q50'] += 1
            else:
                idxSTAnoGroupsPredDF.loc['UnContaminated','q25-q50'] += 1             


        elif shock>q50Shocks and shock<=q75Shocks:
            if PredLabelsAno[i] == Labels_Loc[i]:
                idxSTAnoGroupsPredDF.loc['Contaminated','q50-q75'] += 1
            else:
                idxSTAnoGroupsPredDF.loc['UnContaminated','q50-q75'] += 1
        elif shock>q75Shocks and shock<=maxShocks:
            if PredLabelsAno[i] == Labels_Loc[i]:
                idxSTAnoGroupsPredDF.loc['Contaminated','q75-max'] += 1
            else:
                idxSTAnoGroupsPredDF.loc['UnContaminated','q75-max'] += 1
    return idxSTAnoGroupsPredDF


def ResTable(amplitudeAno,idxSTAnoGroupsPredDF):
    TableResTestID = pd.DataFrame(columns=['Amplitude Range','Ratio'])
    ampliAnoSummary = pd.DataFrame(amplitudeAno).describe()
    minShocks = ampliAnoSummary.loc['min'][0]
    q25Shocks = ampliAnoSummary.loc['25%'][0]
    q50Shocks = ampliAnoSummary.loc['50%'][0]
    q75Shocks = ampliAnoSummary.loc['75%'][0]
    maxShocks = ampliAnoSummary.loc['max'][0]

    TableResTestID['Amplitude Range'] = ['['+'{:.3e}'.format(minShocks)+', '+'{:.3e}'.format(q25Shocks)+']',
                                        '['+'{:.3e}'.format(q25Shocks)+', '+'{:.3e}'.format(q50Shocks)+']',
                                        '['+'{:.3e}'.format(q50Shocks)+', '+'{:.3e}'.format(q75Shocks)+']',
                                        '['+'{:.3e}'.format(q75Shocks)+', '+'{:.3e}'.format(maxShocks)+']']

    TableResTestID['Ratio'] = idxSTAnoGroupsPredDF.loc["Contaminated",:].values/idxSTAnoGroupsPredDF.loc["# TS",:].values
    
    return TableResTestID

