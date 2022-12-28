import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
from sklearn.ensemble import VotingRegressor,VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression 


bank=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)


X=bank.drop(['D','YR'],axis=1)
y=bank['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
    
                                                    train_size=0.7)
lr=LogisticRegression()
bagging =BaggingClassifier(estimator=lr, 
                          random_state=2022, n_estimators=15)
bagging.fit(X_train,y_train)
y_pred=bagging.predict(X_test)
y_pred_prob=bagging.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred_prob))


###for SVM linear ###
from sklearn.svm import SVC
scaler = StandardScaler()
svm = SVC(probability=True, random_state=2022, kernel='linear')
pipe_svm = Pipeline([('STD', scaler),('SVM',svm)])


bagging= BaggingClassifier(estimator=pipe_svm,n_estimators=15, random_state=2022)

bagging.fit(X_train,y_train)
y_pred=bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))


#########################################################################
########### 3) FOR SVM (RADIAL)

    
scaler = StandardScaler()
svm = SVC(probability=True, random_state=2022, kernel='rbf')
pipe_r= Pipeline([('STD', scaler),('SVM',svm)])


bagging= BaggingClassifier(estimator=pipe_r,n_estimators=15, random_state=2022)

bagging.fit(X_train,y_train)
y_pred=bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))


############################################################################
####### 4) for LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
da = LinearDiscriminantAnalysis()
bagging= BaggingClassifier(estimator=da,n_estimators=15, random_state=2022)

bagging.fit(X_train,y_train)
y_pred=bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

###########################################################################
####### 5)  FOR DECISION TREE CLASSIFIER

dtc=DecisionTreeClassifier(random_state=2022)
bagging= BaggingClassifier(base_estimator=dtc,n_estimators=15, random_state=2022)

bagging.fit(X_train,y_train)
y_pred=bagging.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bagging.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

###Grid Search cv ###
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
lr=LogisticRegression() 
scaler = StandardScaler()
svm = SVC(probability=True, random_state=2022, kernel='linear')
pipe_svm = Pipeline([('STD', scaler),('SVM',svm)])
svm = SVC(probability=True, random_state=2022, kernel='rbf')
pipe_r = Pipeline([('STD', scaler),('SVM',svm)])
da = LinearDiscriminantAnalysis()
dtc=DecisionTreeClassifier(random_state=2022)
bagging= BaggingClassifier(n_estimators=15, random_state=2022)
print(bagging.get_params())
params={'estimator':[lr,pipe_svm,pipe_r,da,dtc]}

gcv=GridSearchCV(bagging,param_grid=params,verbose=3,
                 scoring='roc_auc',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

###Concrete Strength ###





