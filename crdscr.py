# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import mlflow
pd.set_option('display.max_columns', 200)

# %%
crdscr = pd.read_csv('credit_score.csv', index_col=0)
crdscr.head(3)

# %%
crdscr.info()

# %%
crdscr.nunique()

# %%
crdscr['Credit_Score'].value_counts()

# %%
# Dataset is slightly imbalanced
sns.countplot(x=crdscr['Credit_Score'])

# %%
# checking Null Vlaue in the Dataset
crdscr.isnull().sum()

# %%
#Checking Duplicate Values in the Data set
crdscr.duplicated().sum()

# %% [markdown]
# # Feature Selection Techinques using SelectKBest Class

# %%
# Data Spliting for feature selection
X= crdscr.iloc[:,:-1]
y= crdscr.iloc[:,-1]

# %%
X_scl= crdscr.iloc[:,:-1]
y_scl= crdscr.iloc[:,-1]

# %%
scale = MinMaxScaler()
scale.fit_transform(X_scl,y_scl)

# %%
# feature selection using SelecKBest class
best_feat = SelectKBest()
best_feat =best_feat.fit(X,y)
f_select_pvalue = best_feat.pvalues_
f_select = best_feat.scores_

# %%
f_pvalue=pd.DataFrame(f_select_pvalue)
f_pvalue

# %%
feat_name = pd.DataFrame(X.columns,columns=['Feature_name'])

# %%
#making and Concotenating dataframe of Highly Correlated Feature Scoresa and PValue
f_select = pd.DataFrame(f_select,columns=['Scores'])
top_feature = pd.concat((feat_name,f_select),axis=1)
top_feature=pd.concat((top_feature, f_pvalue), axis=1)
top_feature.rename(columns={'Feature_name':'feat_name','Scores':'scores',0:'p_value'}, inplace =True)
top_feature

# %%
top_feature.sort_values(by='scores',ascending=False)

# %%
#Correlated and Features Scores and PValue for Feature Selection
sns.set_theme()
fig, axes = plt.subplots(1,2,figsize=(25,8))
fig.set_figheight(8)
fig.set_figwidth(25)
fig.suptitle('Highly Correlated Features Scores and P_Value')
sns.barplot(ax=axes[0],x='scores', y='feat_name', data=top_feature).set(title="Highly correlated features Scores using SeleckBest")
sns.barplot(ax=axes[1],x='p_value', y='feat_name', data=top_feature).set(title="P_Value Scores features using SeleckBest")
plt.show()

# %% [markdown]
# Finding correlation Features using Pearson correlation

# %%
#Correlation features used Corr Function
crdscr_corr_df = crdscr.corr()
crdscr_corr_df

# %%
# Making Correlated Matrix as Dataframe
crdscr_corr_df = crdscr_corr_df.reset_index(drop=True)
crdscr_corr_df

# %%
#Heat Map of Correlation matrix
plt.figure(figsize=(20,10))
sns.heatmap(crdscr.corr(), annot=True,cmap='viridis',fmt=".2f")

# %%
# Checking contanst Variable using Varaince Threshold for Feature Selection.
selector = VarianceThreshold(threshold=0)
selector.fit(X)
selector.get_support()

# %%
#Distiribution plts of some important feature
sns.set_theme()
fig,ax = plt.subplots(1,4, figsize=(25,5))
sns.distplot(crdscr['Annual_Income'], ax=ax[0],color='r')
sns.distplot(crdscr['Age'], ax=ax[1],color='r')
sns.distplot(crdscr['Credit_Score'], ax=ax[2],color='r')
sns.distplot(crdscr['Payment_Behaviour'], ax=ax[3],color='r')

# %%
crdscr.columns

# %%
# Credit Score correlatioship interms of CustomersAge, Annual Income, payment Behaviour, Monthly Inhand Salary, Amount Invested monthly 
fig, ax = plt.subplots(2,3, figsize=(30,15))
sns.scatterplot(data=crdscr, x=crdscr['Credit_Score'], y=crdscr['Age'], size=100, hue='Credit_Score', ax=ax[0][0]).set_title('CreditScore Vs Age')
sns.scatterplot(data=crdscr, x=crdscr['Credit_Score'], y=crdscr['Annual_Income'],hue='Credit_Score', ax=ax[0][1]).set_title('CreditScore Vs Annual Income')
sns.scatterplot(data=crdscr, x=crdscr['Credit_Score'], y=crdscr['Num_of_Loan'],hue='Credit_Score', ax=ax[0][2]).set_title('CreditScore Vs Num of Loan')
sns.scatterplot(data=crdscr, x=crdscr['Credit_Score'], y=crdscr['Monthly_Inhand_Salary'],hue='Credit_Score', ax=ax[1][0]).set_title('CreditScore Vs Monthly Inhand Salary')
sns.scatterplot(data=crdscr, x=crdscr['Credit_Score'], y=crdscr['Outstanding_Debt'],hue='Credit_Score',ax=ax[1][1]).set_title('CreditScore Vs Outstanding Debt')
sns.scatterplot(data=crdscr, x=crdscr['Credit_Score'], y=crdscr['Amount_invested_monthly'],hue='Credit_Score', ax=ax[1][2]).set_title('CreditScore Vs Amount Invested Monthly')

# %% [markdown]
# # TRAINING AND TEST DATA SPLIT

# %%
# Train and Test Data Splitting for model training
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# %%
print(X_train.shape)
print(y_train.shape)

# %%
X_train.head(3)

# %%
y_train.head(3)

# %%
# Dataset is sligthly imbalanced
sns.countplot(data=crdscr, x="Credit_Score")

# %%
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

# %% [markdown]
# # Applying SMOTE for balancing the dataset

# %%
# SMOTE Objet creation
sm = SMOTE(random_state=2)
X_train_rsamp,y_train_rsamp = sm.fit_resample(X_train, y_train)

# %%
print('After OverSampling, the shape of train_X: {}'.format(X_train_rsamp.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_rsamp.shape))
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_rsamp == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_rsamp == 0)))

# %% [markdown]
# # Model Building Process

# %%
# Algorith Selection for model training
model_lr_crdscr = LogisticRegression()
model_dt_crdscr = DecisionTreeClassifier()
model_rf_crdscr = RandomForestClassifier()
model_xgb_crdscr = XGBClassifier()

# %% [markdown]
# # MLFLOW EXPERIMENT SETUP

# %%
# Creating MLFLow Experiment setup
mlflow.set_experiment("Credit Score Experiment")
mlflow.sklearn.autolog()

# %%
#mlflow Experiment - Logistic Regression
with mlflow.start_run(run_name="Logistic regression") as run:
    model_lr_crdscr.fit(X_train_rsamp,y_train_rsamp)
    predslr = model_lr_crdscr.predict(X_test)

# %%
print("Logistic Regression with balanced Dataset :")
print("Model Score on Training data :",model_lr_crdscr.score(X_train_rsamp, y_train_rsamp))
print("Model Score on Test Data: ", model_lr_crdscr.score(X_test, y_test))

# %%
# Experiment using Descion Tree Classifier
with mlflow.start_run(run_name="Decision Tree classiifer") as run:
    model_dt_crdscr.fit(X_train_rsamp,y_train_rsamp)
    predslr = model_dt_crdscr.predict(X_test)

# %%
print("Decision Tree Classifier with balanced Dataset, The training and test score :")
print("Model Score on Training data :",model_dt_crdscr.score(X_train_rsamp, y_train_rsamp))
print("Model Score on Test Data: ", model_dt_crdscr.score(X_test, y_test))

# %%
# Experiment using Ensemble Technique of Random Forest Classifier
with mlflow.start_run(run_name="Random Forest Classifier") as run:
    model_rf_crdscr.fit(X_train_rsamp,y_train_rsamp)
    predslr = model_rf_crdscr.predict(X_test)

# %%
print("Ranodm Forest classifier with balanced Dataset :")
print("Model Score on Training data :",model_rf_crdscr.score(X_train_rsamp, y_train_rsamp))
print("Model Score on Test Data: ", model_rf_crdscr.score(X_test, y_test))

# %%
# Experiment using Ensemble Technique of XGBoost Classifier
with mlflow.start_run(run_name="XGBooost Classifier") as run:
    model_xgb_crdscr.fit(X_train_rsamp,y_train_rsamp)
    predslr = model_xgb_crdscr.predict(X_test)

# %%
print("XGBooost classifier with balanced Dataset :")
print("Model Score on Training data :",model_xgb_crdscr.score(X_train_rsamp, y_train_rsamp))
print("Model Score on Test Data: ", model_xgb_crdscr.score(X_test, y_test))

# %% [markdown]
# # HYPER PARAMETER TUNING USING GRIDSEARCH CV

# %%


# %%


# %%


# %%



