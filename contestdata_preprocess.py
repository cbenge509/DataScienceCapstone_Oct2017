#*************************** IMPORTS ************************************
import pandas as pd
import numpy as np

#from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from numpy import sqrt

#*************************** STATICS ************************************
#input files
orig_trainData = "D:/DataScience/train_values.csv"
orig_trainLabels = "D:/DataScience/train_labels.csv"
orig_testData = "D:/DataScience/test_values.csv"

#output files
postProcess_trainData = "D:/DataScience/pp_train_values.csv"
postProcess_trainLabels = "D:/DataScience/pp_train_labels.csv"
postProcess_testData = "D:/DataScience/pp_test_values.csv"

#estimator hyperparameters
etrClassifyParams = {'n_estimators':100,
             'max_features':100,
             'max_depth':25,
             'min_samples_split':3,
             'random_state':736283,
             'n_jobs':-1}

#interpolated columns
impute_cols = [['school__faculty_salary',{'n_estimators':600, 'random_state':736283, 'max_depth':None, 'max_features':None, 'min_samples_split':2, 'n_jobs':-1}], 
    ['student__demographics_age_entry',{'n_estimators':600, 'random_state':736283, 'max_depth':None, 'max_features':175, 'min_samples_split':2, 'n_jobs':-1}]]

IDcolumnName = 'row_id'

#*********************** HELPER FUNCTIONS ********************************
def descretizeByValue(X, y, descretizeByCol, newColName,bins,binValues):
    combined = pd.concat([X,y[descretizeByCol]], axis=1,join='inner')
    combined[newColName] = pd.qcut(combined[descretizeByCol],bins,binValues)
    combined = combined.drop(labels=[descretizeByCol],axis=1)
    
    return combined

def synthesize(X, colA, colB, colC, newColName, category):
    
    X[newColName] = abs(
            ((-X[colB] + sqrt(abs(X[colB]**2 - (4 * X[colA] * X[colC])))) / (2 * X[colA])) - 
            ((-X[colB] - sqrt(abs(X[colB]**2 + (4 * X[colA] * X[colC])))) / (2 * X[colA]))
        )
    
    print("column [%s] created for %s." % (newColName, category))
    return X

def createIncomeBins(train, test, binColName):
    
    predictors = [x for x in train.columns if x not in [IDcolumnName,binColName]]
    
    model = ExtraTreesClassifier(**etrClassifyParams)
    model.fit(train[predictors], train[binColName])
    
    te_predict = model.predict(test[predictors])
    test[binColName] = te_predict
    
    return test

def interpolateFeatureValues(train, test, imputedColumn, etrParameters):
    
    #define the training predictors
    predictors = [x for x in train.columns if x not in [IDcolumnName, imputedColumn]]
    
    #partition data by training and prediction sets
    X_tr = train[(train[imputedColumn] > 0)].copy()    #populated 'imputed label' data from training file
    X_te = test[(test[imputedColumn] > 0)].copy()      #populated 'imputed lable' data from test file
    X_fit = pd.concat([X_tr, X_te], axis=0)            #combine training and test values
    y_fit = X_fit[imputedColumn]                       #create 1d array of predictors for fit
    
    X_trp = train[(train[imputedColumn] == 0)].copy()    #data with missing 'imputed label' information from training file
    X_tep = test[(test[imputedColumn] == 0)].copy()      #data with missing 'imputed label' information from test file
    
    #define/train the model
    model = ExtraTreesRegressor(**etrParameters)
    model.fit(X_fit[predictors], y_fit)
    
    #tr_predict = np.exp(model.predict(X_trp[predictors]))
    tr_predict = model.predict(X_trp[predictors])
    X_trp[imputedColumn] = tr_predict
    
    #te_predict = np.exp(model.predict(X_tep[predictors]))
    te_predict = model.predict(X_tep[predictors])
    X_tep[imputedColumn] = te_predict
    
    #update the training and testing base data with the predicted values
    train.update(X_trp[[imputedColumn]], join='left')
    test.update(X_tep[[imputedColumn]], join='left')
    
    return train, test


def cleanValues(X):
    # This function will clean the inbound college / graduate income prediction data.
    #meanImputer = Imputer(missing_values=0, strategy='mean',axis=0)
    #meanNaNImputer = Imputer(missing_values=np.NaN, strategy='mean',axis=0)
    
    #Map code values and re-type character data
    X.report_year = X.report_year.map({'year_a':1,'year_f':2, 'year_w':3, 'year_z':4})
    X.report_year = pd.to_numeric(X.report_year, errors='coerce')
    
    X.school__degrees_awarded_highest = X.school__degrees_awarded_highest.map({'Non-degree-granting':0,'Certificate degree':1, 'Associate degree':2, "Bachelor's degree":3, 'Graduate degree':4})
    X.school__degrees_awarded_highest = pd.to_numeric(X.school__degrees_awarded_highest, errors='coerce')
    
    X.school__degrees_awarded_predominant = X.school__degrees_awarded_predominant.map({'Not classified':0,'Predominantly certificate-degree granting':1,"Predominantly associate's-degree granting":2,"Predominantly bachelor's-degree granting":3,'Entirely graduate-degree granting':4})
    X.school__degrees_awarded_predominant = pd.to_numeric(X.school__degrees_awarded_predominant, errors='coerce')
    
    X.school__institutional_characteristics_level = X.school__institutional_characteristics_level.map({'2-year':2,'4-year':4,'Less-than-2-year':0.5})
    X.school__institutional_characteristics_level = pd.to_numeric(X.school__institutional_characteristics_level, errors='coerce')
    
    X.school__main_campus = X.school__main_campus.map({'Main campus':1,'Not main campus':0})
    X.school__main_campus = pd.to_numeric(X.school__main_campus, errors='coerce')
    
    X.school__online_only = X.school__online_only.map({'Not distance-education only':2, 'Distance-education only':1, np.NaN:0})
    X.school__online_only = pd.to_numeric(X.school__online_only, errors='coerce')

    #ordered state mapping from highest income to lowest income (nhl estimated based on region id)
    
    stateMap = {'twr':58,'iqy':57,'xws':56,'jgn':55,'kus':54,'rxy':53,'shi':52,'cmz':51,'oub':50,
        'oli':49,'wjh':48,'rya':47,'axc':46,'qim':45,'lff':44,'dfy':43,'dkf':42,'bkc':41,
        'wto':40,'znt':39,'zdl':38,'vvi':37,'luw':36,'cfi':35,'ccg':34,'oon':33,'oly':32,
        'tbs':31,'ezv':30,'dmg':29,'xve':28,'jfm':27,'eyi':26,'kdg':25,'hbt':24,'bww':23,
        'hww':22,'fxt':21,'idw':20,'kll':19,'ipu':18,'por':17,'uah':16,'slp':15,'rse':14,
        'rgs':13,'hqy':12,'hks':11,'jsu':10,'gkt':9,'pxv':8,'uuo':7,'nhl':6,'hmr':5,
        'slo':4,'ahh':3,'pdh':2,'fga':1}
    
    X.school__state = X.school__state.map(stateMap)
    X.school__state = pd.to_numeric(X.school__state, errors='coerce')
    
    X = pd.get_dummies(X,columns=['school__ownership','school__region_id'])
    
    #drop noisy features
    X = X.drop(labels=['admissions__act_scores_75th_percentile_cumulative',
        'admissions__act_scores_midpoint_cumulative',
        'admissions__act_scores_25th_percentile_english',
        'admissions__act_scores_25th_percentile_math',
        'admissions__act_scores_25th_percentile_writing',
        'admissions__act_scores_75th_percentile_english',
        'admissions__act_scores_75th_percentile_math',
        'admissions__act_scores_75th_percentile_writing',
        'admissions__sat_scores_25th_percentile_math',
        'admissions__sat_scores_25th_percentile_critical_reading',
        'admissions__sat_scores_75th_percentile_math',
        'admissions__sat_scores_75th_percentile_critical_reading',
        'student__demographics_first_generation'], axis=1)
    
    #derive male share demographics after mean averaging the few missing female demographics
    X.student__demographics_female_share = X.student__demographics_female_share.fillna(0.5)
    X['student__demographics_male_share'] = (1.0 - X.student__demographics_female_share)

    #fill all null's with 0 now
    X = X.fillna(0)
    
    return X

def removeDuplicates(X,y, labelColumn, features):
    #This function will find duplicate features, average the label values into a single row (training only), then delete all duplicates
    #second half of function removes outliers based on analysis of faculty salary and income
    
    data = X[X.duplicated(keep=False)]
    data = pd.concat([data, y[labelColumn]], axis=1, join='inner')
    
    avg = pd.DataFrame(data.groupby(features, as_index=False).mean())
    avg = data.merge(avg, on=features, how='left')
    avg.index = data.index

    data.income = avg.income_y
    combined = pd.concat([X, y[labelColumn]], axis=1, join='inner')
    combined.update(data, join='inner')    
    combined = combined.drop_duplicates(keep=False)
    combined.income = round(combined.income,5)
    combined.drop_duplicates(subset=features, keep=False, inplace=True)
    
    #outliers are identified and deleted here from the training set
    outliers = []
    outliers.extend(combined[(combined.school__faculty_salary > 15000) & (combined.income < 30)].index)
    outliers.extend(combined[(combined.school__faculty_salary > 20000)].index)
    outliers.extend(combined[(combined.income > 120)].index)
    combined.drop(outliers,inplace=True)
    
    #correct skew / kurtosis of label and important correlates by converting to normal distributions
    #combined[labelColumn] = np.log(combined[labelColumn])
    
    #product output
    labels = pd.DataFrame(combined[labelColumn])
    data = combined.drop(labels=[labelColumn], axis=1)
    
    return data, labels

def smoothLabel(y,labelColumn):
    #this function is used to apply natural logarithm for reduction in skew of dependent variable

    cols = y.columns
    labels = pd.DataFrame(np.log(y[labelColumn].copy()), columns=cols)

    return labels


#*************************** MAIN ROUTINE ************************************
def preprocess_data():
    #open all files
    X = pd.read_csv(orig_trainData, index_col=0)
    y = pd.read_csv(orig_trainLabels, index_col=0)
    T = pd.read_csv(orig_testData, index_col=0)
    
    #initial encoding, pruning, and NaN filling for train (X) and test (T) values
    print("\nCleaning / encoding raw data...")
    X = cleanValues(X)
    T = cleanValues(T)
    
    #impute important missing values
    for col, params in impute_cols:
        print("Predicting new [%s] values..." % col)
        X, T = interpolateFeatureValues(X, T, col, params)
    
    
    #NOTE: commands removed below resulted in nominal improvement in private score
    
    #synthesize distance of quadratic roots for 3 top features
    print("Synthesizing new columns.")
    X = synthesize(X, 'school__degrees_awarded_predominant_recoded', 'school__institutional_characteristics_level', 'school__faculty_salary', 'synthesized', 'training data')
    T = synthesize(T, 'school__degrees_awarded_predominant_recoded', 'school__institutional_characteristics_level', 'school__faculty_salary', 'synthesized', 'testing data')
    #X = synthesize(X, 'school__degrees_awarded_highest', 'school__degrees_awarded_predominant', 'school__institutional_characteristics_level', 'synthesized', 'training data')
    #T = synthesize(T, 'school__degrees_awarded_highest', 'school__degrees_awarded_predominant', 'school__institutional_characteristics_level', 'synthesized', 'testing data')
    
    
    print("Creating three income bins for classification on training data...")
    X = descretizeByValue(X, y, 'income', 'income_bin', 3, [1,2,3])
    
    print("Predicting income bins via classification on test data...")
    T = createIncomeBins(X, T, 'income_bin')
    
    #print("log smoothing dependent variable.")
    #y = smoothLabel(y, 'income')
    
    #print("removing duplicates and outliers.")
    #X, y = removeDuplicates(X, y, 'income', [x for x in X.columns if x not in [IDcolumnName]])
    
    #write the post-processed files out
    X.to_csv(postProcess_trainData)
    y.to_csv(postProcess_trainLabels)
    T.to_csv(postProcess_testData)
    
    print("post-processing file creation complete.")
