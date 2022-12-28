import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold,cross_val_score
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingRegressor,VotingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,StandardScaler

image_seg= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Image Segmentation\Image_Segmention.csv")
X=image_seg.drop('Class',axis=1)
y=image_seg['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
lr=LogisticRegression()
nb=GaussianNB()
da=LinearDiscriminantAnalysis()
scaler=StandardScaler()
svm=SVC(kernel='linear',probability=True,random_state=2022)
pipe=Pipeline([('STD',scaler),('SVM',svm)])

kfold =StratifiedKFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)
voting=VotingClassifier([('LR',lr),
                         ('NB',nb),
                         ('LDA',da),('SVML',pipe)],voting='soft')

print(voting.get_params())

params = {'SVML__SVM__C':np.linspace(0.001,5,5),
          'LR__C':np.linspace(0.001,5,5)}


gcv = GridSearchCV(voting, param_grid=params, verbose=3, 
                   cv=kfold, scoring='neg_log_loss')

gcv.fit(X, y)
pd_cv=pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)


######Concrete Strength ####
from sklearn.linear_model import Lasso,Ridge,ElasticNet

concrete =pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")
X =concrete.drop('Strength',axis=1)
y=concrete['Strength']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                   train_size=0.7)
ridge=Ridge()
lasso=Lasso()
elastic=ElasticNet()
dtr = DecisionTreeRegressor(random_state = 2022)
voting=VotingRegressor([('Ridge',ridge),
                         ('Lasso',lasso),
                         ('ElasticNet',elastic),('DTR',dtr)])

voting.fit(X_train, y_train)
y_pred=voting.predict(X_test)
print(r2_score(y_test, y_pred))


###Evaluating the regressors Separetely
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_ridge=r2_score(y_test, y_pred)

                                                 
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_lasso=r2_score(y_test, y_pred)

elastic=ElasticNet()
elastic.fit(X_train, y_train)
y_pred = elastic.predict(X_test)
r2_elastic=r2_score(y_test, y_pred)

dtr = DecisionTreeRegressor(random_state = 2022)
dtr.fit(X_train, y_train)
y_pred =dtr.predict(X_test)
r2_dtr=r2_score(y_test, y_pred)


voting=VotingRegressor([('Ridge',ridge),
                         ('Lasso',lasso),
                         ('ElasticNet',elastic),('DTR',dtr)],weights=[r2_ridge,r2_lasso,r2_elastic,r2_dtr])


voting.fit(X_train, y_train)
y_pred=voting.predict(X_test)
print(r2_score(y_test, y_pred))



##### Using Grid Search Cv ###

voting = VotingRegressor([('Ridge',ridge),
                         ('Lasso',lasso),
                         ('ElasticNet',elastic),('TREE',dtr)])
print(voting.get_params())
kfold =KFold(n_splits=5, shuffle=True, 
                        random_state=2022)

params = {'RIDGE__alpha':np.linspace(0.001,5,5),
          'LASSO__alpha':np.linspace(0.001,5,5),
           'TREE__max_depth':[3,None],
           'TREE__min_samples_split':[2, 5 ,10],
            'TREE__min_samples_leaf':[1, 5, 10],
            'ELASTIC__alpha':np.linspace(0.001,5,5),
            'ELASTIC__l1_ratio': np.linspace(0,1,5)}


gcv = GridSearchCV(voting, param_grid=params, verbose=3, 
                   cv=kfold, scoring='r2')

gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

