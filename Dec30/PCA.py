import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from pca import pca
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


brupt=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)


scaler=StandardScaler()
prcomp=PCA()
pipe=Pipeline([('STD',scaler),('PCA',prcomp)])
components=pipe.fit_transform(X_train)
print(np.cumsum(prcomp.explained_variance_ratio_*100))

svm=SVC(kernel='linear',probability=True,random_state=2022)
pd_PC_trn=pd.DataFrame(components[:,:8],
                       columns=['PC'+str(i) for i in np.arange(1,9)])

svm.fit(pd_PC_trn,y_train)

tst_comp=pipe.transform(X_test)
pd_PC_tst=pd.DataFrame(tst_comp[:,:8],
                       columns=['PC'+str(i) for i in np.arange(1,9)])



y_pred=svm.predict(pd_PC_tst)
print(accuracy_score(y_test, y_pred))


y_pred_prob=svm.predict_proba(pd_PC_tst)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

###### Grid Search CV ######
scaler = StandardScaler()
prcomp = PCA()
svm = SVC(probability = True, random_state=2022, kernel = 'linear')

pipe_pca_svm = Pipeline([('STD', scaler),
                         ('PCA', prcomp), ('SVM', svm)])
print(pipe_pca_svm.get_params())
params = {'PCA__n_components': [0.75, 0.8, 0.85, 0.9, 0.95],
          'SVM__C': [0.4, 1, 2, 2.5]}

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
gcv = GridSearchCV(pipe_pca_svm, param_grid = params, cv = kfold,
                   scoring = 'roc_auc', verbose = 3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


#########################################################################
############### hr ##################################

hr = pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X=dum_hr.drop('left',axis=1)
y=dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

###### Grid Search CV ######
scaler = StandardScaler()
prcomp = PCA()
svm = SVC(probability = True, random_state=2022, kernel = 'linear')

pipe_pca_svm = Pipeline([('STD', scaler),
                         ('PCA', prcomp), ('SVM', svm)])
print(pipe_pca_svm.get_params())
params = {'PCA__n_components': [0.75, 0.8, 0.85, 0.9, 0.95],
          'SVM__C': [0.4, 1, 2, 2.5]}

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
gcv = GridSearchCV(pipe_pca_svm, param_grid = params, cv = kfold,
                   scoring = 'roc_auc', verbose = 3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##### Bankruptcy ######

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


bank=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)


X=bank.drop(['D','YR'],axis=1)
y=bank['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

scaler = StandardScaler()
X_trn_scl= scaler.fit_transform(X_train)
#finding the best cluster based on the silhouette
sil=[]
for i in np.arange(2,10):
    km=KMeans(n_clusters=i,random_state=2022)
    km.fit(X_trn_scl)
    labels=km.predict(X_trn_scl)
    sil.append(silhouette_score(X_trn_scl, labels))
print(sil)

#Best Cluster
Ks = np.arange(2,10)
i_max = np.argmax(sil)
best_k = Ks[i_max]
print("Best K =", best_k)



scaler = StandardScaler()
km=KMeans(n_clusters= best_k,random_state=2022)
pipe=Pipeline([('STD',scaler),('KM',km)])
pipe.fit(X_train)
labels=pipe.predict(X_train)


X_train['Cluster']=labels

X_train['Cluster']=X_train['Cluster'].astype('category')

X_trn_ohe=pd.get_dummies(X_train)

###Removing Cluster_3 as it is not present in X_test
from sklearn.ensemble import RandomForestClassifier 
X_test['Cluster'] = labels 
X_test['Cluster'] = X_test['Cluster'].astype('category') 
X_tst_ohe = pd.get_dummies(X_test) 
X_tst_ohe.info()





X_trn_ohe.drop('Cluster_3',axis=1,inplace=True)
rf=RandomForestClassifier(random_state=2022)
rf.fit(X_trn_ohe,y_train)

y_pred_prob=rf.predict_proba(X_tst_ohe)
print(accuracy_score(y_test, y_pred))


y_pred_prob=rf.predict_proba(X_tst_ohe)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
