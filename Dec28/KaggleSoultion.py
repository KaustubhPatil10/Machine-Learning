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
from sklearn.ensemble import RandomForestRegressor



train=pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Dec20\train.csv")

test=pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Dec20\test.csv")

X_train=train.drop(['datetime','count','registered','casual'],axis=1)
y_train=train['count']




rf=RandomForestRegressor(random_state=2022)
rf.fit(X_train,y_train)

X_test=test.drop('datetime',axis=1)

y_pred=np.round(rf.predict(X_test))

submit=pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Dec20\sampleSubmission.csv")
submit['count']=y_pred
submit.to_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec28\vaibhavskaggle.csv",index=False)





