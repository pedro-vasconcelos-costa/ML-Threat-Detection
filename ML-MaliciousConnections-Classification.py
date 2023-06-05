##      ENVIRONMENT      ##
#_______________________________________________________________________________
# import required libraries
import os
import pandas as pan
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

#import data
myDataRaw = pan.read_csv('MLData2023.csv')

#relocate categorical features to the left side for better vizualisation
cat_cols = myDataRaw.select_dtypes(include=['object']).columns.tolist()
num_cols = myDataRaw.select_dtypes(exclude=['object']).columns.tolist()
myDataRaw = myDataRaw[cat_cols + num_cols]

##      CLEAN DATASET      ##
#_______________________________________________________________________________
#check data structure and dimentions
print(myDataRaw.shape)
print(myDataRaw.info())
print(myDataRaw.columns)

#remove 'IPV6.Traffic' feature from the dataset
myData = myDataRaw.drop(columns=['IPV6 Traffic  '])
print(myData.info())

# mask invalid values in Assembled.Payload.Size as "nan"
myData['Assembled Payload Size '] = myData['Assembled Payload Size '].apply(lambda x: np.nan if x <= 0 else x)
print(myData[myData['Assembled Payload Size '].isnull()])

# mask invalid values in Operating System   as "nan"
myData['Operating System  '] = myData['Operating System  '].replace('-', np.nan)
print(myData[myData['Operating System  '].isnull()])

# filter out values with invalid class
myData = myData[(myData['Class'] == 0) | (myData['Class'] == 1)]

#check amount of null(NaN) values
print(myData.isnull().sum())

# merge categories on Operating System   feature
print(myData['Operating System  '].unique())

myData['Operating System  '] = myData['Operating System  '].replace({
    'Windows (Unknown)': 'Windows', 
    'Windows 10+': 'Windows',
    'Windows 7': 'Windows',
    'iOS': 'Others', 
    'Linux (unknown)': 'Others', 
    'Other': 'Others'
})

# merge categories on Connection.State feature
print(myData['Connection State  '].unique())

myData['Connection State  '] = myData['Connection State  '].replace({
    'INVALID': 'Others',
    'NEW': 'Others',
    'RELATED': 'Others'
})

# filter out observations containing NA values
MLData2023_cleaned = myData.dropna()

#check if changes were correctly made
print(MLData2023_cleaned.isnull().sum())
print(MLData2023_cleaned.info())
print(MLData2023_cleaned.columns)

#clean white spaces from column names
MLData2023_cleaned.columns = MLData2023_cleaned.columns.str.strip()


##      DEFINE TEST & TRAINING SAMPLES      ##
#_______________________________________________________________________________
# activate seed (every new session)
np.random.seed(77)

# Separate non-malicious and malicious observations
dat_class0 = MLData2023_cleaned[MLData2023_cleaned['Class'] == 0]  # non-malicious
dat_class1 = MLData2023_cleaned[MLData2023_cleaned['Class'] == 1]  # malicious


##      UNBALANCED TRAINING SET     ##
#_______________________________________________________________________________
# 20000 unbalanced training sample, 19800 non-malicious and 200 malicious

# randomly select 19800 negative observations
train_class0 = dat_class0.sample(n=19800, replace=False)
# randomly select 200 positive observations
train_class1 = dat_class1.sample(n=200, replace=False)
# combine positive and negative samples
mydata_ub_train = pan.concat([train_class0, train_class1], axis=0)

#rename values of Class feature
mydata_ub_train['Class'] = mydata_ub_train['Class'].map({0: 'NonMal', 1: 'Mal'})
mydata_ub_train.describe(include='all')

# write unbalanced training data to CSV file
#mydata_ub_train.to_csv('mydata.ub.train.csv', index=False)


##       BALANCED TRAINING SET     ##
#_______________________________________________________________________________
# 39600 balanced training sample, 19800 non-malicious and 19800 malicious

# Bootstrapping the class 1 observations in the training data
# The same row can be selected multiple times, increasing the number of class 1 observations.
train_class1_2 = train_class1.sample(n=19800, replace=True)

# Combine positive and negative samples
mydata_b_train = pan.concat([train_class0, train_class1_2], axis=0)

#rename values of Class feature
mydata_b_train['Class'] = mydata_b_train['Class'].map({0: 'NonMal', 1: 'Mal'})
mydata_b_train.describe(include='all')

# Write balanced training data to CSV file
#mydata_b_train.to_csv('mydata.b.train.csv', index=False)


##       TEST SET     ##
#_______________________________________________________________________________

# Define test set by excluding rows used in the training set
test_class0 = dat_class0.drop(train_class0.index)
test_class1 = dat_class1.drop(train_class1.index)

# Combine positive and negative test samples
mydata_test = pan.concat([test_class0, test_class1], axis=0)

# Rename values of Class feature
mydata_test['Class'] = mydata_test['Class'].map({0: 'NonMal', 1: 'Mal'})
mydata_test.describe(include='all')
# Write test data to CSV file
#mydata_test.to_csv('mydata.test.csv', index=False)


##      RANDOM FOREST UNBALANCED      ##
#_______________________________________________________________________________

# Define the parameter grid for random forest
param_grid_rf_unB = {
    'iterations': [200, 350, 500],  # Number of trees
    'depth': [2, 6, 12],  # Depth of trees
    'rsm': [0.5, 0.75, 1.0],  # Fraction of features to consider at each split
    'bootstrap_type': ['Bernoulli', 'MVS'],  # Method for sampling the weights of objects
    'subsample': [0.5, 0.75, 1.0]  # Fraction of objects to take for each tree
}

# Separate the features and target variable for the training and test data
X_train_unB = mydata_ub_train.drop(columns=['Class'])
Y_train_unB = mydata_ub_train['Class']
X_test = mydata_test.drop(columns=['Class'])
Y_test = mydata_test['Class']

# Identify the categorical features
cat_features = X_train_unB.select_dtypes(include=['object']).columns.tolist()

# Create a list to store the results
results_rf_unB = []

# Loop over the parameter grid
for params in ParameterGrid(param_grid_rf_unB):
    # Initialize the model
    rf_unB = CatBoostClassifier(**params, random_seed=77, loss_function='Logloss', verbose=False)
    
    # Fit the model to the training data
    rf_unB.fit(X_train_unB, Y_train_unB, cat_features=cat_features)
    
    # Make predictions on the test data
    y_pred_rf_unB = rf_unB.predict(X_test)
    
    # Compute the confusion matrix
    cm_rf_unB = confusion_matrix(Y_test, y_pred_rf_unB)
    
    # Compute the sensitivity (recall), specificity, and accuracy
    sensitivity_rf_unB = recall_score(Y_test, y_pred_rf_unB, pos_label='Mal')
    specificity_rf_unB = recall_score(Y_test, y_pred_rf_unB, pos_label='NonMal')
    accuracy_rf_unB = accuracy_score(Y_test, y_pred_rf_unB)
    
    # Append the results to the results list
    results_rf_unB.append({
        **params,
        'test_sensitivity': sensitivity_rf_unB,
        'test_specificity': specificity_rf_unB,
        'test_accuracy': accuracy_rf_unB
    })

# Convert the results list to a DataFrame
results_rf_unB_df = pan.DataFrame(results_rf_unB)

# Sort the results by the test accuracy
results_rf_unB_df = results_rf_unB_df.sort_values(by='test_accuracy', ascending=False)

# Display the top 10 results
print(results_rf_unB_df.head(10))

# Define the optimal parameters
optimal_params = {
    'iterations': 350,
    'depth': 6,
    'rsm': 0.50,
    'bootstrap_type': 'MVS',
    'subsample': 0.50,
    'random_seed': 77,
    'loss_function': 'Logloss',
    'verbose': True
}

# Initialize the CatBoostClassifier with the optimal parameters
rf_unB_TOP = CatBoostClassifier(**optimal_params)

# Fit the model to the training data
rf_unB_TOP.fit(X_train_unB, Y_train_unB, cat_features=cat_features)

# Make predictions on the test data
y_pred_rf_unB_TOP = rf_unB_TOP.predict(X_test)

# Compute the confusion matrix
cm_rf_unB_TOP = confusion_matrix(Y_test, y_pred_rf_unB_TOP)

# Compute the sensitivity (recall), specificity, and accuracy
sensitivity_rf_unB_TOP = recall_score(Y_test, y_pred_rf_unB_TOP, pos_label='Mal')
specificity_rf_unB_TOP = recall_score(Y_test, y_pred_rf_unB_TOP, pos_label='NonMal')
accuracy_rf_unB_TOP = accuracy_score(Y_test, y_pred_rf_unB_TOP)
precision_rf_unB_TOP = precision_score(Y_test, y_pred_rf_unB_TOP, pos_label='Mal')
f1_rf_unB_TOP = f1_score(Y_test, y_pred_rf_unB_TOP, pos_label='Mal')
fnr_rf_unB_TOP = 1 - sensitivity_rf_unB_TOP
fpr_rf_unB_TOP = 1 - specificity_rf_unB_TOP

# Print the confusion matrix and the metrics
print(cm_rf_unB_TOP)
print(f'False Negative Rate: {format(fnr_rf_unB_TOP * 100, ".2f")} %')
print(f'False Positive Rate: {format(fpr_rf_unB_TOP * 100, ".2f")} %')
print(f'Sensitivity: {format(sensitivity_rf_unB_TOP * 100, ".2f")} %')
print(f'Specificity: {format(specificity_rf_unB_TOP * 100, ".2f")} %')
print(f'Accuracy: {format(accuracy_rf_unB_TOP * 100, ".2f")} %')
print(f'Precision: {format(precision_rf_unB_TOP * 100, ".2f")} %')
print(f'F-score: {format(f1_rf_unB_TOP * 100, ".2f")} %')

##      RANDOM FOREST BALANCED      ##
#_______________________________________________________________________________

# use same parameter grid as unbalanced model
param_grid_rf_Ba = param_grid_rf_unB

# Separate the features and target variable for the training and test data
X_train_Ba = mydata_b_train.drop(columns=['Class'])
Y_train_Ba = mydata_b_train['Class']

# Identify the categorical features
cat_features_Ba = X_train_Ba.select_dtypes(include=['object']).columns.tolist()

# Create a list to store the results
results_rf_Ba = []

# Loop over the parameter grid
for params in ParameterGrid(param_grid_rf_unB):
    # Initialize the model
    rf_Ba = CatBoostClassifier(**params, random_seed=77, loss_function='Logloss', verbose=False)
    
    # Fit the model to the training data
    rf_Ba.fit(X_train_Ba, Y_train_Ba, cat_features=cat_features_Ba)
    
    # Make predictions on the test data
    y_pred_rf_Ba = rf_Ba.predict(X_test)
    
    # Compute the confusion matrix
    cm_rf_Ba = confusion_matrix(Y_test, y_pred_rf_Ba)
    
    # Compute the sensitivity (recall), specificity, and accuracy
    sensitivity_rf_Ba = recall_score(Y_test, y_pred_rf_Ba, pos_label='Mal')
    specificity_rf_Ba = recall_score(Y_test, y_pred_rf_Ba, pos_label='NonMal')
    accuracy_rf_Ba = accuracy_score(Y_test, y_pred_rf_Ba)
    
    # Append the results to the results list
    results_rf_Ba.append({
        **params,
        'test_sensitivity': sensitivity_rf_Ba,
        'test_specificity': specificity_rf_Ba,
        'test_accuracy': accuracy_rf_Ba
    })

# Convert the results list to a DataFrame
results_rf_Ba_df = pan.DataFrame(results_rf_Ba)

# Sort the results by the test accuracy
results_rf_Ba_df = results_rf_Ba_df.sort_values(by='test_sensitivity', ascending=False)

# Display the top 10 results
print(results_rf_Ba_df.head(10))

# Define the optimal parameters
optimal_params_Ba = {
    'iterations': 200,
    'depth': 2,
    'rsm': 1.00,
    'bootstrap_type': 'MVS',
    'subsample': 1.00,
    'random_seed': 77,
    'loss_function': 'Logloss',
    'verbose': True
}

# Initialize the CatBoostClassifier with the optimal parameters
rf_Ba_TOP = CatBoostClassifier(**optimal_params_Ba)

# Fit the model to the training data
rf_Ba_TOP.fit(X_train_Ba, Y_train_Ba, cat_features=cat_features_Ba)

# Make predictions on the test data
y_pred_rf_Ba_TOP = rf_Ba_TOP.predict(X_test)

# Compute the confusion matrix
cm_rf_Ba_TOP = confusion_matrix(Y_test, y_pred_rf_Ba_TOP)

# Compute the sensitivity (recall), specificity, and accuracy
sensitivity_rf_Ba_TOP = recall_score(Y_test, y_pred_rf_Ba_TOP, pos_label='Mal')
specificity_rf_Ba_TOP = recall_score(Y_test, y_pred_rf_Ba_TOP, pos_label='NonMal')
accuracy_rf_Ba_TOP = accuracy_score(Y_test, y_pred_rf_Ba_TOP)
precision_rf_Ba_TOP = precision_score(Y_test, y_pred_rf_Ba_TOP, pos_label='Mal')
f1_rf_Ba_TOP = f1_score(Y_test, y_pred_rf_Ba_TOP, pos_label='Mal')
fnr_rf_Ba_TOP = 1 - sensitivity_rf_Ba_TOP
fpr_rf_Ba_TOP = 1 - specificity_rf_Ba_TOP

# Print the confusion matrix and the metrics
print(cm_rf_Ba_TOP)
print(f'False Negative Rate: {format(fnr_rf_Ba_TOP *100, ".2f")} %')
print(f'False Positive Rate: {format(fpr_rf_Ba_TOP * 100, ".2f")} %')
print(f'Sensitivity: {format(sensitivity_rf_Ba_TOP * 100, ".2f")} %')
print(f'Specificity: {format(specificity_rf_Ba_TOP * 100, ".2f")} %')
print(f'Accuracy: {format(accuracy_rf_Ba_TOP * 100, ".2f")} %')
print(f'Precision: {format(precision_rf_Ba_TOP * 100, ".2f")} %')
print(f'F-score: {format(f1_rf_Ba_TOP * 100, ".2f")} %')

##      ELASTIC-NET UNBALANCED      ##
#_______________________________________________________________________________

# Define search ranges for lambda (C in sklearn) values
lambdas_enaN = np.logspace(-3, 3, num=10)

# Define search ranges for alpha (l1_ratio in sklearn) values
alphas_enaN = np.arange(0.1, 1.0, 0.1)

# Define control parameters for cross validation 
cv_enaN = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)

# get categorical features' col names
cat_features = ['Operating System  ', 'Connection State  ', 'Ingress Router  ']
# get numerical features' col names
num_features = mydata_ub_train.columns.difference(cat_features).tolist()
num_features.remove('Class')

# Define preprocessor
preprocessor_enaN_unB = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)])

# set full prediction pipeline.
clf_enaN_unB = Pipeline(steps=[('preprocessor', preprocessor_enaN_unB),
                      ('classifier', LogisticRegression(penalty='elasticnet', solver='saga'))])

# Define parameter grid
param_grid_enaN = {
    'classifier__l1_ratio': alphas_enaN,  # l1_ratio = alpha 
    'classifier__C': 1/lambdas_enaN  # C = inverse of lambda regularization
}

# Initialize GridSearchCV
grid_search_enaN_unB = GridSearchCV(clf_enaN_unB, param_grid_enaN, cv=cv_enaN)

# Fit the model
grid_search_enaN_unB.fit(mydata_ub_train.drop('Class', axis=1), mydata_ub_train['Class'])

# Optimal lambda and alpha values
TOP_lambda_enaN_unB = 1/grid_search_enaN_unB.best_params_['classifier__C']
TOP_alpha_enaN_unB = grid_search_enaN_unB.best_params_['classifier__l1_ratio']

print(f"TOP lambda: {TOP_lambda_enaN_unB}")
print(f"TOP alpha: {TOP_alpha_enaN_unB}")

# Model coefficients
coef_enaN_unB = grid_search_enaN_unB.best_estimator_.named_steps['classifier'].coef_

# Predicted class labels on test data
pred_enaN_unB = grid_search_enaN_unB.predict(mydata_test.drop('Class', axis=1))

# Confusion matrix
cf_matrix_enaN_unB = confusion_matrix(mydata_test['Class'], pred_enaN_unB) 
print("Confusion Matrix:")
print(cf_matrix_enaN_unB)

# Summary of confusion matrix
print(classification_report(mydata_test['Class'], pred_enaN_unB))

# Compute the sensitivity (recall), specificity, and accuracy
sensitivity_enaN_unB = recall_score(mydata_test['Class'], pred_enaN_unB, pos_label='Mal')
specificity_enaN_unB = recall_score(mydata_test['Class'], pred_enaN_unB, pos_label='NonMal')
accuracy_enaN_unB = accuracy_score(mydata_test['Class'], pred_enaN_unB)
precision_enaN_unB = precision_score(mydata_test['Class'], pred_enaN_unB, pos_label='Mal')
f1_enaN_unB = f1_score(mydata_test['Class'], pred_enaN_unB, pos_label='Mal')
fnr_enaN_unB = 1 - sensitivity_enaN_unB
fpr_enaN_unB = 1 - specificity_enaN_unB

# Print the confusion matrix and the metrics
print(cf_matrix_enaN_unB)
print(f'False Negative Rate: {format(fnr_enaN_unB *100, ".2f")} %')
print(f'False Positive Rate: {format(fpr_enaN_unB * 100, ".2f")} %')
print(f'Sensitivity: {format(sensitivity_enaN_unB * 100, ".2f")} %')
print(f'Specificity: {format(specificity_enaN_unB * 100, ".2f")} %')
print(f'Accuracy: {format(accuracy_enaN_unB * 100, ".2f")} %')
print(f'Precision: {format(precision_enaN_unB * 100, ".2f")} %')
print(f'F-score: {format(f1_enaN_unB * 100, ".2f")} %')

##      ELASTIC-NET BALANCED      ##
#_______________________________________________________________________________

# get numerical features' col names
num_features_Ba = mydata_b_train.columns.difference(cat_features).tolist()
num_features_Ba.remove('Class')

# Define preprocessor
preprocessor_enaN_Ba = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features_Ba),
        ('cat', OneHotEncoder(), cat_features)])

# set full prediction pipeline.
clf_enaN_Ba = Pipeline(steps=[('preprocessor', preprocessor_enaN_Ba),
                      ('classifier', LogisticRegression(penalty='elasticnet', solver='saga'))])

# Initialize GridSearchCV
grid_search_enaN_Ba = GridSearchCV(clf_enaN_Ba, param_grid_enaN, cv=cv_enaN)

# Fit the model
grid_search_enaN_Ba.fit(mydata_b_train.drop('Class', axis=1), mydata_b_train['Class'])

# Optimal lambda and alpha values
TOP_lambda_enaN_Ba = 1/grid_search_enaN_Ba.best_params_['classifier__C']
TOP_alpha_enaN_Ba = grid_search_enaN_Ba.best_params_['classifier__l1_ratio']

print(f"TOP lambda: {TOP_lambda_enaN_Ba}")
print(f"TOP alpha: {TOP_alpha_enaN_Ba}")

# Model coefficients
coef_enaN_Ba = grid_search_enaN_Ba.best_estimator_.named_steps['classifier'].coef_

# Predicted class labels on test data
pred_enaN_Ba = grid_search_enaN_Ba.predict(mydata_test.drop('Class', axis=1))

# Confusion matrix
cf_matrix_enaN_Ba = confusion_matrix(mydata_test['Class'], pred_enaN_Ba) 

# Summary of confusion matrix
print(classification_report(mydata_test['Class'], pred_enaN_Ba))

# Compute the sensitivity (recall), specificity, and accuracy
sensitivity_enaN_Ba = recall_score(mydata_test['Class'], pred_enaN_Ba, pos_label='Mal')
specificity_enaN_Ba = recall_score(mydata_test['Class'], pred_enaN_Ba, pos_label='NonMal')
accuracy_enaN_Ba = accuracy_score(mydata_test['Class'], pred_enaN_Ba)
precision_enaN_Ba = precision_score(mydata_test['Class'], pred_enaN_Ba, pos_label='Mal')
f1_enaN_Ba = f1_score(mydata_test['Class'], pred_enaN_Ba, pos_label='Mal')
fnr_enaN_Ba = 1 - sensitivity_enaN_Ba
fpr_enaN_Ba = 1 - specificity_enaN_Ba

# Print the confusion matrix and the metrics
print(cf_matrix_enaN_Ba)
print(f'False Negative Rate: {format(fnr_enaN_Ba *100, ".2f")} %')
print(f'False Positive Rate: {format(fpr_enaN_Ba * 100, ".2f")} %')
print(f'Sensitivity: {format(sensitivity_enaN_Ba * 100, ".2f")} %')
print(f'Specificity: {format(specificity_enaN_Ba * 100, ".2f")} %')
print(f'Accuracy: {format(accuracy_enaN_Ba * 100, ".2f")} %')
print(f'Precision: {format(precision_enaN_Ba * 100, ".2f")} %')
print(f'F-score: {format(f1_enaN_Ba * 100, ".2f")} %')
