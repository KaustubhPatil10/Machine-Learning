import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold,GridSearchCV


job=pd.read_csv("JobSalary2.csv")

#finding NAs in columns
job.isnull().sum()

#dropping the rows with NA values
job.dropna()

#constant imputer
imp=SimpleImputer(strategy='constant',fill_value=50)
imp.fit_transform(job)

#Mean Imputer

imp=SimpleImputer(strategy='mean')
imp.fit_transform(job)

#Median Imputer
imp=SimpleImputer(strategy='median')
np_imp=imp.fit_transform(job)

#job.median()

#Convert into Dataframe
pd_imp=pd.DataFrame(np_imp,columns=job.columns)

######Chemical Process ####

chemdata= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Chemical Process Data\ChemicalProcess.csv")


X=chemdata.drop('Yield',axis=1)
y=chemdata['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)


imp=SimpleImputer(strategy='mean')
imp.fit_transform(chemdata)


imp=SimpleImputer(strategy='mean')
x_trn_tf=imp.fit_transform(X_train)
x_tst_tf=imp.transform(X_test)


##Linear Regression find r2 score
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_trn_tf,y_train)
lr.predict(x_tst_tf)

y_pred=lr.predict(x_tst_tf)
print(r2_score(y_test,y_pred))

#using Pipeline find r2 score
from sklearn.pipeline import Pipeline
imp=SimpleImputer(strategy='mean')
lr=LinearRegression()
pipe=Pipeline([('IMPUTE',imp),('LR',lr)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
print(r2_score(y_test, y_pred))


##Grid Search using r2 score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
imp=SimpleImputer(strategy='mean')
scaler=StandardScaler()
knn=KNeighborsRegressor()
pipe=Pipeline([('IMPUTE',imp),('STD',scaler),('KNN',knn)])
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
params={'KNN__n_neighbors':np.arange(1,11)}
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#MEan and Median At a time it show only bigger r2_score median/mean

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
imp=SimpleImputer()
scaler=StandardScaler()
knn=KNeighborsRegressor()
pipe=Pipeline([('IMPUTE',imp),('STD',scaler),('KNN',knn)])
print(pipe.get_params()) #adding these statement
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
params={'IMPUTE__strategy':['mean','median'],###adding Impute Strategy
        'KNN__n_neighbors':np.arange(1,11)}
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
