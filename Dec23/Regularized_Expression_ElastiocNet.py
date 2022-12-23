import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold,GridSearchCV

concrete=pd.read_csv("C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")

X=concrete.drop('Strength',axis=1)
y=concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)


elastic=ElasticNet()
elastic.fit(X_train,y_train)
y_pred=elastic.predict(X_test)
print(r2_score(y_test,y_pred))



#Grid search

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
elastic=ElasticNet()
params={'alpha':np.linspace(0.001,11,20),'l1_ratio':np.linspace(0,1,5)}
gcv=GridSearchCV(elastic,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


###Boston###
boston=pd.read_csv("Boston.csv")

X=boston.drop('medv',axis=1)
y=boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)
elastic=ElasticNet()
elastic.fit(X_train,y_train)
y_pred=elastic.predict(X_test)
print(r2_score(y_test,y_pred))



#Grid search

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
elastic=ElasticNet()
params={'alpha':np.linspace(0.001,11,20),'l1_ratio':np.linspace(0,1,5)}
gcv=GridSearchCV(elastic,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
