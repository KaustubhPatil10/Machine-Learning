import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score



bank=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)


X=bank.drop(['D','YR'],axis=1)
y=bank['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)



lr=LogisticRegression()
nb=GaussianNB()
da=LinearDiscriminantAnalysis()
rf=RandomForestClassifier(random_state=2022)

### W/o pass through 
stack=StackingClassifier([('LR',lr),('NB',nb),('DA',da)],final_estimator=rf)

stack.fit(X_train,y_train)
y_pred=stack.predict(X_test)
print(accuracy_score(y_test, y_pred))


y_pred_prob=stack.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

##With pass through

stack=StackingClassifier([('LR',lr),('NB',nb),('DA',da)],final_estimator=rf,passthrough=True)

stack.fit(X_train,y_train)
y_pred=stack.predict(X_test)
print(accuracy_score(y_test, y_pred))


y_pred_prob=stack.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

##### Grid search CV ######

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
print(stack.get_params())
params ={'LR__C':np.linspace(0,5,5),
         'final_estimator__max_features':[2,4,6,8]}

gcv=GridSearchCV(stack,param_grid=params,verbose=3,
                 scoring='roc_auc',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

####### Vehicle Silhouttes ###
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

vehicle= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec30\Vehicle Silhouettes\Vehicle.csv")
X=vehicle.drop('Class',axis=1)
y=vehicle['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)



X_train, X_test, y_train, y_test = train_test_split(X,le_y,
                                                    random_state=2022,
                                                    train_size=0.7)


da=LinearDiscriminantAnalysis()
svm=SVC(kernel='linear',probability=True,random_state=2022)
scaler=StandardScaler()
pipe_svm=Pipeline([('STD',scaler),('SVML',svm)])

dtc = DecisionTreeClassifier(random_state = 2022)
clf=XGBClassifier(random_state=2022)


### Grid search CV ###
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
stack=StackingClassifier([('SVM',svm),('DTC',dtc),('DA',da)],final_estimator=clf,passthrough=True)

print(stack.get_params())
params ={ 'SVM__SVML__C':np.linspace(0.001,8,10),
          'TREE__max_depth': [4,None],
          'TREE__min_samples_split': [2,4,10],
          'TREE__min_samples_leaf': [1,4],
         'final_estimator__max_features':[50,100],
         'final_estimator__learing_rate':[0.1,0.5],
         'final_estimator__max_depth':[3,5]}

gcv=GridSearchCV(stack,param_grid=params,verbose=3,
                 scoring='neg_log_loss',cv=kfold)

gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)



#### Concrete Strength #####
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

concrete =pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")
X =concrete.drop('Strength',axis=1)
y=concrete['Strength']

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)                                                
elastic=ElasticNet()
dtr = DecisionTreeRegressor(random_state=2022)
clf=XGBRegressor(random_state=2022)


stack=StackingRegressor([('ELASTIC',elastic),('TREE',dtr)],final_estimator=clf,passthrough=True)

print(stack.get_params())
params ={'TREE__max_depth':[None,3],
         'TREE__min_samples_split': [2,5,10],
         'TREE__min_samples_leaf':[1,5],
         'ELASTIC__alpha':np.linspace(0,10,5),
         'ELASTIC__l1_ratio': np.linspace(0,1,5),
         'final_estimator__max_features':[50,100],
         'final_estimator__learing_rate':[0.1,0.5],
         'final_estimator__max_depth':[3,5]}

gcv=GridSearchCV(stack,param_grid=params,verbose=3,
                 scoring='r2',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
