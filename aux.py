import pandas as pd 
import sys
import numpy as np
from random import randint
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from joblib import dump, load



def model(inpath,outpath):

    np.random.seed(17)

    num_features = 13
    df1 = pd.read_csv(inpath)
    df1 = shuffle(df1)
    df1.reset_index(inplace=True, drop=True)

    #based on the number of feature columns and target variables
    x =df1.iloc[:,:num_features]
    y = df1.iloc[:,num_features:]
    #fitting regressor models decision trees, random forest, gradient boosting, xgboost
    rdregressor=RandomForestRegressor()
    dtregressor=DecisionTreeRegressor()
    xgboost = XGBRegressor()
    gradientboost = GradientBoostingClassifier()
    rdregressor.fit(x,y)
    dtregressor.fit(x,y)
    xgboostreg = MultiOutputRegressor(xgboost).fit(x,y)
    gbregressor = MultiOutputRegressor(gradientboost).fit(x,y)
    score_dt = cross_val_score(dtregressor, x,y,cv=5)
    score_rd = cross_val_score(rdregressor, x,y,cv=5)
    score_xg = cross_val_score(xgboostreg, x,y,cv=5)
    score_gb = cross_val_score(gbregressor, x,y,cv=5)
    #from IPython import embed;embed()
    maxima = max(score_dt.mean(), score_gb.mean(), score_xg.mean(), score_rd.mean())
    std = min(score_dt.std(), score_gb.std(), score_xg.std(), score_rd.std())
    if maxima == score_dt.mean():
        dump(dtregressor, outpath+'/model.joblib')
    if maxima == score_rd.mean():
        dump(rdregressor, outpath+'/model.joblib')
    if maxima == score_xg.mean():
        dump(xgboostreg, outpath+'/model.joblib')
    if maxima == score_gb.mean():
        dump(gbregressor, outpath+'/model.joblib')

    
def convert(boolData):
    if isinstance(boolData, bool):
        if boolData:
            return 1
        else: 
            return 0
    else:
        return boolData

def predict(inputlists):
    int_list=[]
    for i in inputlists:
        int_list.append(convert(i))
    regressor = load('/home/ram/Downloads/model.joblib')
    resultList=postProcessing(regressor.predict(np.array([int_list])))
    
    return resultList


def postProcessing(inputarray):
    
    covid=inputarray[0][0]
    malaria=inputarray[0][1]
    dengue=inputarray[0][2]
    jaundice=inputarray[0][3]
    inputarray = inputarray.flatten()
    inputarray = np.squeeze(inputarray)
    inputarray =  np.sort(inputarray)[::-1]
    testResult=""
    for value in inputarray:
        if covid==value :
            testResult+="Possibility of having Covid is {}\n".format(value)
        if malaria==value :
            testResult+="Possibility of having Malaria is {}\n".format(value)
        if dengue==value :
            testResult+="Possibility of having Dengue is {}\n".format(value)
        if jaundice==value :
            testResult+="Possibility of having Jaundice is {}\n".format(value)

    return testResult

