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


bank=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)


X=bank.drop(['D','YR'],axis=1)
y=bank['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
    
                                                    train_size=0.7)
clf=GradientBoostingClassifier(random_state=2022)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


y_pred_prob=clf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

##Grid search cv ###

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params ={'n_estimators':[50,100,150],
         'max_depth':[1,2,3,4],
         'learning_rate':[0.01,0.1,0.15,0.2,0.3]}
clf=GradientBoostingClassifier(random_state=2022)
gcv=GridSearchCV(clf,param_grid=params,verbose=3,
                 scoring='roc_auc',cv=kfold)
print(clf.get_params())
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

######## XGB #####
from xgboost import XGBClassifier

xgb_model=XGBClassifier(random_state=2022)
print(xgb_model.get_params)
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params ={'n_estimators':[50,100,150],
         'max_depth':[1,2,3,4],
         'learning_rate':[0.01,0.1,0.15,0.2,0.3]}
clf=GradientBoostingClassifier(random_state=2022)
gcv=GridSearchCV(xgb_model,param_grid=params,verbose=3,
                 scoring='roc_auc',cv=kfold)
print(clf.get_params())
gcv.fit(X,y)
print(gcv.best_params_)

print(gcv.best_score_)

#### ImageSegmentation  ####
image_seg= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Image Segmentation\Image_Segmention.csv")
X=image_seg.drop('Class',axis=1)
y=image_seg['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

### XGB ###
xgb_model=XGBClassifier(random_state=2022)
print(xgb_model.get_params)
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params ={'n_estimators':[50,100,150],
         'max_depth':[1,2,3,4],
         'learning_rate':[0.01,0.1,0.15,0.2,0.3]}
clf=GradientBoostingClassifier(random_state=2022)
gcv=GridSearchCV(xgb_model,param_grid=params,verbose=3,
                 scoring='neg_log_loss',cv=kfold)
print(clf.get_params())
gcv.fit(X,le_y)
print(gcv.best_params_)

print(gcv.best_score_)


##### Concrete Strength ###
concrete=pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")

X=concrete.drop(['Strength'],axis=1)
y=concrete['Strength']



### XGB Regressor ###
from xgboost import XGBRegressor
xgb_model=XGBRegressor(random_state=2022)
print(xgb_model.get_params)
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
params ={'n_estimators':[50,100,150],
         'max_depth':[1,2,3,4],
         'learning_rate':[0.01,0.1,0.15,0.2,0.3]}
clf=GradientBoostingClassifier(random_state=2022)
gcv=GridSearchCV(xgb_model,param_grid=params,verbose=3,
                 scoring='r2',cv=kfold)
print(clf.get_params())
gcv.fit(X,y)
print(gcv.best_params_)

print(gcv.best_score_)