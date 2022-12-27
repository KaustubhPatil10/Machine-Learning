import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,StandardScaler

satellite=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec26\Satellite Imaging\Satellite.csv",sep=";")

X=satellite.drop(['classes'],axis=1)
y=satellite['classes']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X,le_y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

db=LinearDiscriminantAnalysis()
db.fit(X_train,y_train)
y_pred=db.predict(X_test)
Y_pred_prob=db.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, Y_pred_prob))

####K-Folds CV ###
db=LinearDiscriminantAnalysis()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(db,X,y,scoring='neg_log_loss',cv=kfold)
print(results.mean())


#Quardatic##
db=QuadraticDiscriminantAnalysis()
db.fit(X_train,y_train)
y_pred=db.predict(X_test)
Y_pred_prob=db.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, Y_pred_prob))

####K-Folds CV ###

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(db,X,y,scoring='neg_log_loss',cv=kfold)
print(results.mean())

###Gaussian NB

nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
Y_pred_prob=nb.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, Y_pred_prob))

####K-Folds CV ###

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(nb,X,y,scoring='neg_log_loss',cv=kfold)
print(results.mean())

