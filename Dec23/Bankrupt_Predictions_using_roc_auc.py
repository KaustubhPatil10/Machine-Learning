import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor



bank= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=bank.drop(['YR','D'],axis=1)
y=bank['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)

scaler = StandardScaler()
knn=KNeighborsClassifier(n_neighbors=3)
pipe=Pipeline([('STD',scaler),('KNN',knn)])
pipe.fit(X_train,y_train)


y_pred=pipe.predict(X_test)
print(accuracy_score(y_test,y_pred))

y_pred_prob =pipe.predict_proba(X_test)
print(log_loss(y_test,y_pred_prob))



##Grid Search using roc_auc

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
knn=KNeighborsClassifier()
pipe=Pipeline([('STD',scaler),('KNN',knn)])
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16,2)}
gcv=GridSearchCV(pipe,param_grid=params,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


##or using Grid Serch Predicting on unlabelled data

bank_test= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\testBankruptcy.csv",index_col=0)
best_model=gcv.best_estimator_
predictions=best_model.predict(bank_test)
print(predictions)
