import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

insurance = pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Medical Cost Personal\insurance.csv")

dum_ins = pd.get_dummies(insurance, drop_first = True)

X = dum_ins.drop('charges', axis = 1)
y = dum_ins['charges']

##Linear Regression
from sklearn.linear_model import LinearRegression
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
lr=LinearRegression()
results=cross_val_score(lr,X,y,cv=kfold,scoring='r2')
print(results.mean())



### Using Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
knn=KNeighborsRegressor()
pipe=Pipeline([('STD',scaler),('KNN',knn)])
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


#predicting on unlabel data y not in data 
knn=KNeighborsRegressor(n_neighbors=7)
pipe=Pipeline([('STD',scaler),('KNN',knn)])
pipe.fit(X,y)

tst_insure= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Medical Cost Personal\tst_insure.csv")
dum_tst=pd.get_dummies(tst_insure,drop_first=True)
print(X.dtypes)
print(dum_tst.dtypes)
prdictions=pipe.predict(dum_tst)


##or using Grid Serch
pd_cv=pd.DataFrame(gcv.cv_results_)
best_model=gcv.best_estimator_
tst_insure= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Medical Cost Personal\tst_insure.csv")
dum_tst=pd.get_dummies(tst_insure,drop_first=True)
predictions=pipe.predict(dum_tst)

