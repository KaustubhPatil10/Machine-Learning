import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder,StandardScaler


kyphosis=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec26\Kyphosis\Kyphosis.csv")


X=kyphosis.drop(['Kyphosis'],axis=1)
y=kyphosis['Kyphosis']


le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

svm=SVC(kernel='linear',probability=True,random_state=2022)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))


#With scaling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler=StandardScaler()
svm=SVC(kernel='linear',probability=True,random_state=2022)
pipe=Pipeline([('STD',scaler),('SVM',svm)])
print(pipe.get_params())

params={'SVM__C':np.linspace(0.001,10,20)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='linear',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,
                 scoring='roc_auc')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)


##Using Polynomial ###

kyphosis=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec26\Kyphosis\Kyphosis.csv")


X=kyphosis.drop(['Kyphosis'],axis=1)
y=kyphosis['Kyphosis']


le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

svm=SVC(kernel='poly',probability=True,random_state=2022)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

#With scaling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler=StandardScaler()
svm=SVC(kernel='poly',probability=True,random_state=2022)
pipe=Pipeline([('STD',scaler),('SVM',svm)])
print(pipe.get_params())

params={'SVM__C':np.linspace(0.001,10,20),
        'SVM__degree':[2,3,4],
        'SVM__coef0':np.linspace(-2,4,5)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='poly',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,
                 scoring='roc_auc')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)

###Radial##
kyphosis=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec26\Kyphosis\Kyphosis.csv")


X=kyphosis.drop(['Kyphosis'],axis=1)
y=kyphosis['Kyphosis']


le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

svm=SVC(kernel='rbf',probability=True,random_state=2022)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler=StandardScaler()
svm=SVC(kernel='rbf',probability=True,random_state=2022)
pipe=Pipeline([('STD',scaler),('SVM',svm)])
print(pipe.get_params())

params={'SVM__C':np.linspace(0.001,10,20),
        'SVM__gamma':np.linspace(0.001,10,20)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='rbf',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,
                 scoring='roc_auc')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)

