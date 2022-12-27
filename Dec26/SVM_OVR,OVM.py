import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder,StandardScaler



image_seg= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Image Segmentation\Image_Segmention.csv")
X=image_seg.drop('Class',axis=1)
y=image_seg['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler=StandardScaler()
svm=SVC(kernel='linear',probability=True,random_state=2022)
pipe=Pipeline([('STD',scaler),('SVM',svm)])
print(pipe.get_params())

params={'SVM__C':np.linspace(0.001,10,20),
        'SVM__decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='linear',probability=True,random_state=2022)
gcv=GridSearchCV(pipe,param_grid=params,cv=kfold,verbose=3,
                 scoring='neg_log_loss')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)


###Ploynomial####
#With scaling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler=StandardScaler()
svm=SVC(kernel='poly',probability=True,random_state=2022)
pipe=Pipeline([('STD',scaler),('SVM',svm)])
print(pipe.get_params())

params={'SVM__C':np.linspace(0.001,10,20),
        'SVM__degree':[2,3,4],
        'SVM__coef0':np.linspace(-2,4,5),
        'SVM__decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='poly',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,
                 scoring='neg_log_loss')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)

####Radial####
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler=StandardScaler()
svm=SVC(kernel='rbf',probability=True,random_state=2022)
pipe=Pipeline([('STD',scaler),('SVM',svm)])
print(pipe.get_params())

params={'SVM__C':np.linspace(0.001,10,20),
        'SVM__gamma':np.linspace(0.001,10,20),
        'SVM__decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='rbf',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,
                 scoring='neg_log_loss')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)



