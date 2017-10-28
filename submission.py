#*************************** IMPORTS ************************************
import pandas as pd
import numpy as np
import capstone_model as ccd
import contestdata_preprocess as pp

#*************************** STATICS ************************************
outputFile = "D:/DataScience/submission60.csv"

IDcol = 'row_id'

file_TrainValues = "D:/DataScience/pp_train_values.csv"
file_TrainLabels = "D:/DataScience/pp_train_labels.csv"
file_TestValues = "D:/DataScience/pp_test_values.csv"

#*************************** MAIN ROUTINE ************************************
#pre-process all input data
pp.preprocess_data()

# Load the training data
X = pd.read_csv(file_TrainValues,index_col=0)
y = pd.read_csv(file_TrainLabels,index_col=0)
T = pd.read_csv(file_TestValues, index_col=0)

predictors = [x for x in X.columns if x not in [IDcol]]

print('Build and fit StackingCV model (no bias, complex, 15 folds)...')
model = ccd.build_StackingModelCV(posBias=False, simple=False, cvRun=15)
model.fit(X[predictors].as_matrix(), y['income'].as_matrix())

#*************************** PRODUCE OUTPUT ************************************
print("Predicting new values...")
y = model.predict(T[predictors].as_matrix())
T['income'] = y[:]

#Note: removed logarithm smoothing for income / nominal affect on score    
#print("correcting income value predictions...")
#T.income = np.exp(T.income)

Z = pd.DataFrame(X['income'])
Z.index.rename('row_id', inplace=True)

Z.to_csv(outputFile)
print ("File created!")
print ("\nTotal income estimated: {0}.".format(X.income.sum()))
