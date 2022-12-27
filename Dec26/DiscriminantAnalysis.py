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

brupt=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)
db=LinearDiscriminantAnalysis()
db.fit(X_train,y_train)
y_pred=db.predict(X_test)
Y_pred_prob=db.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, Y_pred_prob))

####K-Folds CV ###

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(db,X,y,scoring='roc_auc',cv=kfold)
print(results.mean())


####QuardaticDiscriminant Analysis##

brupt=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)
db=QuadraticDiscriminantAnalysis()
db.fit(X_train,y_train)
y_pred=db.predict(X_test)
Y_pred_prob=db.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, Y_pred_prob))

####K-Folds CV ###

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(db,X,y,scoring='roc_auc',cv=kfold)
print(results.mean())

#####Vehicle Quardatic Discriminant Analysis #####
from sklearn.preprocessing import LabelEncoder,StandardScaler

vehicle= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec24\Vehicle Silhouettes\Vehicle.csv")
X=vehicle.drop('Class',axis=1)
y=vehicle['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)



X_train, X_test, y_train, y_test = train_test_split(X,le_y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

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

#########Linear Discriminant Analysis###


vehicle= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec24\Vehicle Silhouettes\Vehicle.csv")
X=vehicle.drop('Class',axis=1)
y=vehicle['Class']

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

###ROC AUC OVO
from sklearn.multiclass import OneVsOneClassifier                                    
db=LinearDiscriminantAnalysis()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(db,X,y,scoring='',cv=kfold)
print(results.mean())