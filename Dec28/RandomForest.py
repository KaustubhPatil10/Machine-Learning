import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso,Ridge,ElasticNet


bank=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)


X=bank.drop(['D','YR'],axis=1)
y=bank['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
    
                                                    train_size=0.7)

rf=RandomForestClassifier(random_state=2022)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
print(accuracy_score(y_test, y_pred))


y_pred_prob=rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

##Grid search cv ###

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params ={'max_features':np.arange(3,15)}

gcv=GridSearchCV(rf,param_grid=params,verbose=3,
                 scoring='roc_auc',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


###### Feature Importance Plot ###
import matplotlib.pyplot as plt

best_model=gcv.best_estimator_
imps=best_model.feature_importances_
plt.barh(X.columns,imps)
plt.show()

i_sorted=np.argsort(-imps)
n_sorted=X.columns[i_sorted]
imp_sort=imps[i_sorted]
plt.barh(n_sorted,imp_sort)
plt.title("sorted features importances")
plt.show()

##### HR Data ####
hr=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec24\human-resources-analytics\HR_comma_sep.csv")
dum_hr=pd.get_dummies(hr,drop_first=True)

X=dum_hr.drop('left',axis=1)
y=dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

rf=RandomForestClassifier(random_state=2022)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
print(accuracy_score(y_test, y_pred))


y_pred_prob=rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

##Grid search cv ###

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
.

gcv=GridSearchCV(rf,param_grid=params,verbose=3,
                 scoring='roc_auc',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


###### Feature Importance Plot ###
import matplotlib.pyplot as plt

best_model=gcv.best_estimator_
imps=best_model.feature_importances_
plt.barh(X.columns,imps)
plt.show()

### Sorted Graph ###

i_sorted=np.argsort(-imps)
n_sorted=X.columns[i_sorted]
imp_sort=imps[i_sorted]
plt.barh(n_sorted,imp_sort)
plt.title("sorted features importances")
plt.show()























