import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV


brupt=pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

lr = LogisticRegression()
bagging = BaggingClassifier(base_estimator = lr,
                            random_state= 2022, n_estimators = 15)

bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
y_pred_prob = bagging.predict_proba(X_test)[:, 1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))

### SVM Linear ======================================================
from sklearn.svm import SVC

svm_lr = SVC(kernel='linear',probability=True,random_state=2022)
bagging = BaggingClassifier(base_estimator = svm_lr,
                            random_state= 2022, n_estimators = 15)

bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))

### SVM radial =========================================================

scaler = StandardScaler()

svm_rbf = SVC(kernel='rbf',probability=True,random_state=2022)
bagging = BaggingClassifier(base_estimator = svm_rbf,
                            random_state= 2022, n_estimators = 15)

bagging.fit(X_train, y_train)
y_pred=bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))


#### Linear DA =======================================================

da = LinearDiscriminantAnalysis()
bagging = BaggingClassifier(base_estimator = da,
                            random_state= 2022, n_estimators = 15)

bagging.fit(X_train, y_train)
y_pred=bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))

### DTree =========================================================

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
bagging = BaggingClassifier(base_estimator = dtc,
                            random_state= 2022, n_estimators = 15)

bagging.fit(X_train, y_train)
y_pred=bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))

###### Grid Search CV ==============================================


kfold = StratifiedKFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)
lr = LogisticRegression()
scaler = StandardScaler()
svm_l = SVC(kernel = 'linear')
pipe_l = Pipeline([('STD', scaler),('SVM', svm_l)])
svm_r = SVC(kernel = 'rbf')
pipe_r = Pipeline([('STD', scaler),('SVM', svm_r)])
da = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state= 2022)
bagging = BaggingClassifier(random_state= 2022, n_estimators = 15)
print(bagging.get_params())
params = {'estimator': [lr, pipe_l, pipe_r, da, dtc]}
gcv = GridSearchCV(bagging, param_grid = params, cv = kfold,
                   verbose = 3, scoring = 'roc_auc')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


