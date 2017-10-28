import math
import os
#path for mingw64 (XGBoost requirement)
pathx = 'C:\\Program Files\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = pathx + ';' + os.environ['PATH']

from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet


# dtr/ada params from submission 8 - RMSE : 3.4678
adaDTR_adaParams8 = {'n_estimators':400, 
                     'random_state':736283, 
                     'learning_rate':1}

adaDTR_dtrParams = {'max_depth':None, 
                    'min_samples_split':6, 
                    'max_features':None}

#etr params from submission 13 - RMSE : 3.2440
etrParams = {'n_estimators':600, 
             'random_state':736283, 
             'max_depth':None, 
             'max_features':175, 
             'min_samples_split':2, 
             'n_jobs':-1}

#xgb params from submission 22 - RMSE : 3.3746
xgbParams = {'learning_rate':0.024, 
             'n_estimators':4900, 
             'max_depth':6, 
             'min_child_weight':1,
             'gamma':0,
             'subsample':0.7,
             'colsample_bytree':0.6,
             'reg_alpha':1,
             'scale_pos_weight':1,
             'random_state':736283, 'n_jobs':-1}

rfgParams = {'n_estimators':390,
             'max_features':106,
             'min_samples_leaf':1,
             'max_depth':None,
             'random_state':736283,
             'n_jobs':-1}

etParams = {'criterion':'mse',
            'max_depth':12,
            'max_features':175,
            'min_samples_split':2,
            'min_samples_leaf':1,
            'random_state':736283}

adaParams = {'n_estimators':500, 'random_state':736283, 'learning_rate':1}

gbrParams = {'n_estimators':800, 
             'random_state':736283,
             'max_depth':8,
             'criterion':'friedman_mse',
             'max_features':117,
             'min_samples_split':3
             }

lgbmParams = {'boosting_type':'gbdt',
               'random_state':736283,
               'num_leaves':400,
               'min_data_in_leaf':5,
               'max_bin':500,
               'learning_rate':0.08,
               'n_estimators':300,
               'n_jobs':-1,
               'subsample':0.8
        }

def holdoutScore(X_test, y_test, model, name):
    #This function scores your hold-out RMSE

    #print out the model parameters used
    print("\n")
    print(model.get_params())

    score = model.score(X_test, y_test)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    
    print("\nModel used: {0}".format(name))
    print("Variance score (x 100): {0}".format(round((score*100),3)))
    print("RMSE: %.4f\n" % math.sqrt(mse))

def build_StackingModel():
    adaModel = AdaBoostRegressor(DecisionTreeRegressor(**adaDTR_dtrParams), **adaDTR_adaParams8)
    etrModel = ExtraTreesRegressor(**etrParams)
    xgbModel = XGBRegressor(**xgbParams)
    
    stregr = StackingRegressor(regressors=[adaModel, etrModel, xgbModel], meta_regressor=LinearRegression())
    
    return stregr

def build_StackingModelCV(posBias=False, simple=False, cvRun=5):
    
    if simple:
        adaModel = AdaBoostRegressor(DecisionTreeRegressor(),n_estimators=100)
        etrModel = ExtraTreesRegressor(n_estimators=100)
        xgbModel = XGBRegressor(n_estimators=100)
        rfgModel = RandomForestRegressor(n_estimators=100)
        gbrModel = GradientBoostingRegressor(n_estimators=100)
        adaETRModel = AdaBoostRegressor(ExtraTreeRegressor(),n_estimators=100)
        lgbmModel = LGBMRegressor(learning_rate=0.8,n_estimators=100)
    else:
        adaModel = AdaBoostRegressor(DecisionTreeRegressor(**adaDTR_dtrParams), **adaDTR_adaParams8)
        etrModel = ExtraTreesRegressor(**etrParams)
        xgbModel = XGBRegressor(**xgbParams)
        rfgModel = RandomForestRegressor(**rfgParams)
        gbrModel = GradientBoostingRegressor(**gbrParams)
        adaETRModel = AdaBoostRegressor(ExtraTreeRegressor(**etParams),**adaParams)
        lgbmModel = LGBMRegressor(**lgbmParams)

    if posBias:
        metaModel = ElasticNet(alpha=0, positive=True)
    else:
        metaModel = LinearRegression()
        
    stregr = StackingCVRegressor(regressors=[adaModel, etrModel, xgbModel,rfgModel,gbrModel,adaETRModel,lgbmModel], meta_regressor=metaModel, cv=cvRun)
    
    return stregr