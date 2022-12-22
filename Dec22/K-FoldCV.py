import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor


mowers = pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Datasets\RidingMowers.csv")


dum_mow = pd.get_dummies(mowers, drop_first = True)

X = dum_mow.drop('Response_Not Bought', axis = 1)
y = dum_mow['Response_Not Bought']

knn=KNeighborsClassifier(n_neighbors=3)
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

#Accuracy
cross_val_score(knn,X,y,cv=kfold)

#ROC AUC
results=cross_val_score(knn,X,y,cv=kfold,scoring='roc_auc')
print(results.mean())

#For Finding the best k

acc =[]
Ks = [1,3,5,7,9,11,13,15]
for i in Ks:
    knn = KNeighborsRegressor(n_neighbors=i)
    result = cross_val_score(knn,X,y,cv=kfold,scoring='roc_auc')
    acc.append(result.mean())
print(acc)
i_max = np.argmax(acc)
best_k = Ks[i_max] 
print("Best n_neighbors =", best_k) 
    
print("Best Score= ",acc[i_max])

#Grid search CV without loop find best k and best score
from sklearn.model_selection import GridSearchCV
params={'n_neighbors':[1,3,5,7,9,11,13,15] }
knn=KNeighborsClassifier()
gcv=GridSearchCV(knn,param_grid=params,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


####Boston####
from sklearn.model_selection import KFold
boston=pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Datasets\Boston.csv")
dum_bos = pd.get_dummies(boston, drop_first = True)

X = dum_bos.drop('medv',axis = 1)
y = dum_bos['medv']


kfold=KFold(n_splits=5,shuffle=True,random_state=2022)


#Grid search CV without loop find best k and best score
from sklearn.model_selection import GridSearchCV
params={'n_neighbors':np.arange(1,16)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(knn,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


    
