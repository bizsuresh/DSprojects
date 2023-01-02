# %%
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import mlflow

# %%
mobile = pd.read_csv('mobile.csv')
mobile.head(3)

# %%
plt.figure(figsize=(10,10))
sns.countplot(mobile)

# %%
dst_mob=mobile.select_dtypes(exclude='object')
dst_mob

# %%
mobile.info()

# %%
plt.hist(mobile.price_range)

# %%
sns.distplot(mobile.price_range)

# %%
mobile.shape

# %%
mobile.nunique()

# %%
mobile.columns

# %%
X = mobile.iloc[:,:-1]
y = mobile.iloc[:,-1]

# %%
X.shape

# %%
y.head(3)

# %% [markdown]
# # Feature selection based on Kbest using chi2

# %%
best_features_score = SelectKBest(score_func=chi2, k=10)
best_features_score.fit(X,y)
best_features = pd.DataFrame(best_features_score.scores_, columns = ['score'])
best_features
Xcols = pd.DataFrame(X.columns, columns = ['Feature_name'])
best_scores = pd.concat([Xcols, best_features], axis=1)
best_scores

# %% [markdown]
# #Top 10 best feature score

# %%
best_scores.nlargest(10,'score')

# %%
#train and TEst ata splits
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=124)

# %%
#Scaling the Feature
scaler = StandardScaler()
mob_scaled=scaler.fit_transform(X)
mob_scaled

# %%
X_train, X_test, y_train, y_test = train_test_split(mob_scaled,y, test_size=0.3, random_state=124)

# %%
#Model Selection
model_lr_mob = LogisticRegression()
model_lr_mob_scaled = LogisticRegression()
model_dt_mob = DecisionTreeClassifier()
model_rf_mob = RandomForestClassifier()

# %%
pipe_lr = Pipeline([('Standard Scaler', StandardScaler(), LogisticRegression())])

# %% [markdown]
# #ML FLOW INTEGRATION
# 

# %%
#Setting up experiment using MLFOW
mlflow.set_experiment("Classification Experiment")
mlflow.sklearn.autolog()

# %%
# Experiment with logistic regression Algorithm with standard scaler scaled value
with mlflow.start_run(run_name="Scaled Logistic regression") as run:
    model_lr_mob_scaled.fit(X_train,y_train)
    predslr = model_lr_mob_scaled.predict(X_test)

# %%
print("Logistic Regression with scaled value Model Training and Test Score Below :")
print("Model Score on Training data :",model_lr_mob_scaled.score(X_train, y_train))
print("Model Score on Test Data: ", model_lr_mob_scaled.score(X_test, y_test))

# %%
# Experiment with logistic regression Algorithm with out scaling
with mlflow.start_run(run_name="Logistic regression") as run:
    model_lr_mob.fit(X_train,y_train)
    predslr = model_lr_mob.predict(X_test)

# %%
print("Logistic Regression model Training and Test Score Below :")
print("Model Score on Training data :",model_lr_mob.score(X_train, y_train))
print("Model Score on Test Data: ", model_lr_mob.score(X_test, y_test))

# %%
# Experiment with Decisioin Tree Classifier Algorithm with out scaling
with mlflow.start_run(run_name="Decisioin Tree Classifier") as run:
    model_dt_mob.fit(X_train,y_train)
    preds_dt = model_dt_mob.predict(X_test)

# %%
print("Decision Treee Classifier Training and Test Score Below :")
print("Model Score on Training data :",model_dt_mob.score(X_train, y_train))
print("Model Score on Test Data: ", model_dt_mob.score(X_test, y_test))

# %%
# Experiment with Random Forest classifier Algorithm with out scaling
with mlflow.start_run(run_name="Random Forest classifier") as run:
    model_rf_mob.fit(X_train,y_train)
    preds_rf = model_rf_mob.predict(X_test)

# %%
print("Random Forest Classifier Training and Test Score Below :")
print("Model Score on Training data :",model_rf_mob.score(X_train, y_train))
print("Model Score on Test Data: ", model_rf_mob.score(X_test, y_test))


