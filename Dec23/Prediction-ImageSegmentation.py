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


image_seg= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Image Segmentation\Image_Segmention.csv")
X=image_seg.drop('Class',axis=1)
y=image_seg['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

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


##Grid Search using neg_lg_loss

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
knn=KNeighborsClassifier()
pipe=Pipeline([('STD',scaler),('KNN',knn)])
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16)}
gcv=GridSearchCV(pipe,param_grid=params,scoring='neg_log_loss',cv=kfold)
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)

##or using Grid Serch Predicting on unlabelled data

tst_image= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Image Segmentation\tst_img.csv")
best_model=gcv.best_estimator_
predictions=best_model.predict(tst_image)

print(le.inverse_transform(predictions))