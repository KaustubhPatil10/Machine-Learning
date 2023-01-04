import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec28\train.csv")
test = pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec28\test.csv")

X_train = train.drop(['count','datetime','registered','casual'], axis = 1)
y_train = train['count']


rf = RandomForestRegressor(random_state = 2022)
rf.fit(X_train, y_train)

X_test = test.drop('datetime', axis = 1)

y_pred = np.round(rf.predict(X_test))

submit = pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec28\sampleSubmission.csv")
submit ['count'] = y_pred
submit.to_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec28\sbt_rf_2022.csv", index = False)
