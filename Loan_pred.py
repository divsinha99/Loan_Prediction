# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Loading the data
loan_data = pd.read_csv("train.csv", sep=',')
loan_test = pd.read_csv("test.csv", sep=',')
loan_data.head()


#Build a pivot table Credit history vs Loan status
def percConvert(ser):
    return ser/float(ser[-1])
pd.crosstab(loan_data['Credit_History'],loan_data['Loan_Status'],margins=True).apply(percConvert, axis=1)


# Imputation of null values in original data set
loan_data['Gender'].fillna(str(loan_data['Gender'].mode()), inplace=True)
loan_data['Married'].fillna(str(loan_data['Married'].mode()), inplace=True)
loan_data['Dependents'].fillna(str(loan_data['Dependents'].mode()), inplace = True)
loan_data['Self_Employed'].fillna('No', inplace = True)
# Imputing LoanAmount from a pivot table
table = loan_data.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
loan_data['LoanAmount'].fillna(loan_data[loan_data['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
loan_data['Loan_Amount_Term'].fillna(float(loan_data['Loan_Amount_Term'].mode()), inplace = True)
loan_data['Credit_History'].fillna(1.0, inplace = True)
# check for any further missing values:
loan_data.isnull().sum()


# Imputation of null values in test data set
loan_test['Gender'].fillna(str(loan_test['Gender'].mode()), inplace=True)
loan_test['Dependents'].fillna(str(loan_test['Dependents'].mode()), inplace = True)
loan_test['Self_Employed'].fillna('No', inplace = True)
# Imputing LoanAmount from a pivot table
table = loan_test.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
loan_test['LoanAmount'].fillna(loan_test[loan_test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
loan_test['Loan_Amount_Term'].fillna(float(loan_test['Loan_Amount_Term'].mode()), inplace = True)
loan_test['Credit_History'].fillna(1.0, inplace = True)
# check for any further missing values:
loan_test.isnull().sum()



# Using label encoder to convert all of these into categorical values:
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    loan_data[i] = le.fit_transform(loan_data[i])

#using label encoder for test dataset to transform into categorical values
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    loan_test[i] = le.fit_transform(loan_test[i])


# Check for outliers of loan_amount:
loan_data['LoanAmount'].hist(bins=20)
# Boxplot shows there are too many outliers for Loan amount category:
loan_data.boxplot(column='LoanAmount')
# Applying logarithmic transformation for loanamount:
loan_data['TotalIncome'] = loan_data['ApplicantIncome'] + loan_data['CoapplicantIncome']
loan_data['relLoanAmount'] = (loan_data['LoanAmount']/loan_data['TotalIncome'])
loan_data['EMI/TotIncome'] = (loan_data['LoanAmount']*0.095*(1+0.095)**(loan_data['Loan_Amount_Term'])/(1+0.095)**((loan_data['Loan_Amount_Term'])-1)/(loan_data['TotalIncome']))

#Applying to test data
loan_test['TotalIncome'] = loan_test['ApplicantIncome'] + loan_test['CoapplicantIncome']
loan_test['relLoanAmount'] = (loan_test['LoanAmount']/loan_test['TotalIncome'])
loan_test['EMI/TotIncome'] = (loan_test['LoanAmount']*0.095*(1+0.095)**(loan_test['Loan_Amount_Term'])/(1+0.095)**((loan_test['Loan_Amount_Term'])-1)/(loan_test['TotalIncome']))




                                                    
loan_data_pred = loan_data.drop(['Loan_Status'], axis=1)
loan_data_pred = loan_data_pred.drop(['Loan_ID'], axis=1)
loan_test_pred = loan_test.drop(['Loan_ID'], axis=1)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(loan_data_pred)
loan_data_pred_sc = scaler.transform(loan_data_pred)
loan_test_scaled = scaler.transform(loan_test_pred)


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1,random_state=101)
#predictors = ['Credit_History','relLoanAmount', 'TotalIncome_log','LoanAmount_log', 'EMI/TotIncome']
#predictors = ['Credit_History','relLoanAmount', 'EMI/TotIncome','Education','Property_Area']  ## - LB 79%
predictors = ['Credit_History','TotalIncome','relLoanAmount', 'EMI/TotIncome','Loan_Amount_Term','Property_Area']

k = []
for i in range (0,len(loan_data_pred.columns)):
    if loan_data_pred.columns[i] in predictors:
        k.append(i)
print('k :', k)


outcome = ['Loan_Status']
#Fit the model:
model.fit(loan_data_pred_sc[:,k],loan_data[outcome])
#Make predictions on test data set:
predictions_f = model.predict(loan_test_scaled[:,k])  
#Make predictions on training set for measuring accuracy:
predictions = model.predict(loan_data_pred_sc[:,k])

#Print accuracy
accuracy = metrics.accuracy_score(predictions,loan_data[outcome])
print("Accuracy : %s" % "{0:.3%}".format(accuracy))
#Perform k-fold cross-validation with 5 folds
kf = KFold(loan_data.shape[0], n_folds=5)
error = []
for train, test in kf:
    # Filter training data
    train_predictors = (loan_data_pred_sc[:,k][train,:])
    # The target we're using to train the algorithm.
    train_target = loan_data[outcome].iloc[train]
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    #Record error from each cross-validation run
    error.append(model.score(loan_data_pred_sc[:,k][test,:], loan_data[outcome].iloc[test]))
print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

##-----------------------------------------------------------------------------------

##   Checking AUC-ROC for diff leaf sizes:

print("AUC - ROC : ", roc_auc_score(loan_data[outcome],model.oob_prediction))





##-----------------------------------------------------------------------------------
#Create a series with feature importances:
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area',
       'TotalIncome', 'relLoanAmount',
       'EMI/TotIncome']
       
d = []
for i in range (0,len(loan_data_pred.columns)):
    if loan_data_pred.columns[i] in predictor_var:
       d.append(i)
print('d :', d)

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
model.fit(loan_data_pred_sc[:,d],loan_data[outcome])
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)



######---------------------------------------------------------------------------------
pred = pd.DataFrame(predictions_f)
res = pd.concat([loan_test['Loan_ID'], pred], axis = 1)
res[0] = res[0].apply(lambda x: 'Y' if x== 1 else 'N')
res = res.rename(columns={0: "Loan_Status"})
res['Loan_Status'].value_counts()
res.to_csv('sample_submission.csv', index=False)



