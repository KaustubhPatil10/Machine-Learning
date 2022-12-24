import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold




image_seg= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Image Segmentation\Image_Segmention.csv")
X=image_seg.drop('Class',axis=1)
y=image_seg['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

#With scaling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler=StandardScaler()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
lr=LogisticRegression(solver='saga',random_state=2022)
pipe=Pipeline([('STD',scaler),('LR',lr)])
print(pipe.get_params())

params={'LR__penalty':['l1','l2','elasticnet',None],
        'LR__C':np.linspace(0.001,4,5),
        'LR__l1_ratio':np.linspace(0.001,1,5),
        'LR__multi_class':['ovr','multinomial']}

gcv=GridSearchCV(pipe,param_grid=params,verbose=3,
                 scoring='neg_log_loss',cv=kfold)
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)

#####Vehicle Sihouttes####


vehicle= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec24\Vehicle Silhouettes\Vehicle.csv")
X=vehicle.drop('Class',axis=1)
y=vehicle['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

#With scaling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler=StandardScaler()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
lr=LogisticRegression(solver='saga',random_state=2022)
pipe=Pipeline([('STD',scaler),('LR',lr)])
print(pipe.get_params())

params={'LR__penalty':['l1','l2','elasticnet',None],
        'LR__C':np.linspace(0.001,4,5),
        'LR__l1_ratio':np.linspace(0.001,1,5),
        'LR__multi_class':['ovr','multinomial']}

gcv=GridSearchCV(pipe,param_grid=params,verbose=3,
                 scoring='neg_log_loss',cv=kfold)
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)
































































































































.