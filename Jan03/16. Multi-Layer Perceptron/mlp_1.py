import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)
mm = MinMaxScaler()
mlp = MLPClassifier(activation='logistic',random_state=2022)
pipe = Pipeline([('MM',mm),('MLP',mlp)])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = pipe.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

####################### Grid Search CV ############################
pipe = Pipeline([('MM',mm),('MLP',mlp)])
print(pipe.get_params())

params = {'MLP__hidden_layer_sizes':[(20,10,5),(10,5),(50,)],
          'MLP__activation':['tanh','logistic','identity'],
          'MLP__learning_rate':['constant','invscaling','adaptive'],
          'MLP__learning_rate_init':[0.001, 0.3, 0.5]}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='roc_auc', verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################## Concrete ###########################
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")
concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
mm = MinMaxScaler()
mlp = MLPRegressor(random_state=2022)
pipe = Pipeline([('MM',mm),('MLP',mlp)])
params = {'MLP__hidden_layer_sizes':[(6,4,3),(7,5),(10,)],
          'MLP__learning_rate':['constant','invscaling','adaptive'],
          'MLP__learning_rate_init':[0.001, 0.3, 0.5]}
gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='r2', verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############################ HR Data ###################################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")
hr = pd.read_csv('HR_comma_sep.csv')
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
mm = MinMaxScaler()
mlp = MLPClassifier(random_state=2022)
pipe = Pipeline([('MM',mm),('MLP',mlp)])
params = {'MLP__hidden_layer_sizes':[(20,10,5),(30,20,10),(40,30,10)],
          'MLP__activation':['tanh','logistic','identity'],
          'MLP__learning_rate':['constant','invscaling','adaptive'],
          'MLP__learning_rate_init':[0.001, 0.3, 0.5]}
gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='roc_auc', verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
