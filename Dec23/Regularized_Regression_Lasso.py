import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold,GridSearchCV



concrete=pd.read_csv("C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")

X=concrete.drop('Strength',axis=1)
y=concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)


lasso=Lasso()
lasso.fit(X_train,y_train)
y_pred=lasso.predict(X_test)
print(r2_score(y_test,y_pred))



#Grid search
#np.linspace(1,11)# It give 50 values with equally spaced

#np.linspace(1,11,10)#It give 10 values with equally spaced
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
lasso=Lasso()
params={'alpha':np.linspace(0.001,11,20)}
gcv=GridSearchCV(lasso,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)