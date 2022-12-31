import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.ensemble import RandomForestClassifier



train=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\train.csv")

test=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\test.csv")

X_train=train.drop(['ID','TARGET'],axis=1)
y_train=train['TARGET']




rf=RandomForestClassifier(random_state=2022)
rf.fit(X_train,y_train)

X_test=test.drop('ID',axis=1)

y_pred_prob=rf.predict_proba(X_test)[:,1]

submit=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\sample_submission.csv")
submit['TARGET']=y_pred_prob
submit.to_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec30\vaibhavskaggle2.csv",index=False)





