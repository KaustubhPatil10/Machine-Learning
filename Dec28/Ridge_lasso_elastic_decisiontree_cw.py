import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

concrete = pd.read_csv(r"C:\PG_DBDA\Advance analytics classwork\Advanced_Analytics\Cases\Concrete Strength\Concrete_Data.csv")

X = concrete.drop('Strength', axis = 1)
y = concrete['Strength']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)

# ridge, lasso, elasticnet

ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()
dtr = DecisionTreeRegressor(random_state= 2022)
voting = VotingRegressor([('RIDGE', ridge),
                          ('LASSO', lasso),
                          ('ELASTIC', elastic),
                          ('TREE', dtr)])

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))

# EVALUATING REGRESSORS SEPARATELY

# ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred)


# lasso
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred)

# elasticnet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
y_pred = elasticnet.predict(X_test)
r2_elasticnet = r2_score(y_test, y_pred)


dtr = DecisionTreeRegressor(random_state = 2022)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, y_pred)

# weighted
voting = VotingRegressor([('RIDGE', ridge),
                          ('LASSO', lasso),
                          ('ELASTIC', elastic),
                          ('TREE', dtr)],
                         weights = [r2_ridge, r2_lasso, 
                                    r2_elasticnet, r2_dtr,])
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))


######## Grid Search CV ###############################

kfold = KFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)
ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()
dtr = DecisionTreeRegressor(random_state = 2022)
voting = VotingRegressor([('RIDGE', ridge),
                          ('LASSO', lasso),
                          ('ELASTIC', elastic),
                          ('TREE', dtr)])
print(voting.get_params())

params = {'RIDGE__alpha': np.linspace(0.001, 5, 5),
          'LASSO__alpha': np.linspace(0.001, 5, 5),
          'ELASTIC__alpha': np.linspace(0.001, 5, 5),
          'ELASTIC__l1_ratio': np.linspace(0, 1, 5),
          'TREE__max_depth': [None, 3],
          'TREE__min_samples_split': [2, 5, 10],
          'TREE__min_samples_leaf': [1, 5, 10]}

gcv = GridSearchCV(voting, param_grid = params, verbose = 3,
                   cv = kfold, scoring = 'r2')

gcv.fit(X, y)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_) 



