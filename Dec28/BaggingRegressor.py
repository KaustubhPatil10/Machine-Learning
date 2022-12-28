import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
from sklearn.ensemble import VotingRegressor,VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.svm import SVC

concrete=pd.read_csv("C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")

X=concrete.drop('Strength',axis=1)
y=concrete['Strength']



###Grid Search cv ###

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
lr=LinearRegression() 
ridge=Ridge()
lasso=Lasso()
elastic=ElasticNet()
dtr=DecisionTreeRegressor(random_state=2022)

bagging= BaggingRegressor(n_estimators=15, random_state=2022)
print(bagging.get_params())

params={'estimator':[lr,ridge,lasso,elastic,dtr]}

gcv=GridSearchCV(bagging, param_grid=params,verbose=3,
                 scoring='r2',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
