import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

brupt=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

svm=SVC(kernel='linear',probability=True,random_state=2022)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

######Grid Search Cv ###

params={'C':np.linspace(0.001,10,20)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='linear',probability=True,random_state=2022)
gcv=GridSearchCV(svm, param_grid=params,cv=kfold,verbose=3,
                 scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

####BY USING Polynomoial ######
brupt=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

svm=SVC(kernel='poly',probability=True,random_state=2022)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

######Grid Search Cv ###

params={'C':np.linspace(0.001,10,20),
        'degree':[2,3,4],
        'coef0':np.linspace(-2,4,5)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='poly',probability=True,random_state=2022)
gcv=GridSearchCV(svm, param_grid=params,cv=kfold,verbose=3,
                 scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


####BY USING Radial ######
brupt=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

svm=SVC(kernel='rbf',probability=True,random_state=2022)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

######Grid Search Cv ###

params={'C':np.linspace(0.001,10,20),
        'gamma':np.linspace(0.001,10,20)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='rbf',probability=True,random_state=2022)
gcv=GridSearchCV(svm, param_grid=params,cv=kfold,verbose=3,
                 scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
