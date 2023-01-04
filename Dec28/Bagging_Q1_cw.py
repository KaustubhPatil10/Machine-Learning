import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor

concrete = pd.read_csv(r"C:\PG_DBDA\Advance analytics classwork\Advanced_Analytics\Cases\Concrete Strength\Concrete_Data.csv")

X = concrete.drop('Strength', axis = 1)
y = concrete['Strength']

########## Grid Search CV ==============================

kfold = KFold(n_splits = 5, shuffle = True, random_state= 2022)
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()
dtr = DecisionTreeRegressor(random_state=2022)

bagging = BaggingRegressor(random_state= 2022, n_estimators = 15)
print(bagging.get_params())

params = {'estimator': [lr, ridge, lasso, elastic, dtr]}
gcv = GridSearchCV(bagging, param_grid = params, cv = kfold,
                   verbose = 3, scoring = 'r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
