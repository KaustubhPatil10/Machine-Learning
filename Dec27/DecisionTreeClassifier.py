import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
hr = pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first= True )
X = dum_hr.drop('left', axis = 1)
y = dum_hr['left']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

dtc = DecisionTreeClassifier(random_state= 2022,
                             max_depth = 3)

dtc.fit(X_train, y_train)

plt.figure(figsize = (40,20))
plot_tree(dtc, feature_names = X.columns,
          class_names = ['0','1'], fontsize = 14)
plt.show()

y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_preb_prob = dtc.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_preb_prob))

############# Grid Search ################

dtc = DecisionTreeClassifier(random_state = 2022)
params = {'max_depth': [2,3,4,5,None],
          'min_samples_split': [2,5,10],
          'min_samples_leaf': [1,5,10]}
kfold = StratifiedKFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)

gcv = GridSearchCV(dtc, param_grid = params, verbose = 3,
                   cv = kfold, scoring = 'roc_auc')

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_) 

best_model =gcv.best_estimator_
plt.figure(figsize = (50,30))
plot_tree(best_model, feature_names= X.columns,
          class_names =['0','1'], fontsize= 13)

plt.show() 

###Best Model Feature Importances plot ####

print(best_model.feature_importances_)
imps=best_model.feature_importances_
plt.barh(X.columns,imps)
plt.title('Feature Importance Plot')
plt.show()

######################### BAnkruptcy  ############
brupt=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

dtc = DecisionTreeClassifier(random_state= 2022,
                             max_depth = 3)

dtc.fit(X_train, y_train)

############# Grid Search ################

dtc = DecisionTreeClassifier(random_state = 2022)
params = {'max_depth': [2,3,4,5,None],
          'min_samples_split': [2,5,10],
          'min_samples_leaf': [1,5,10]}
kfold = StratifiedKFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)

gcv = GridSearchCV(dtc, param_grid = params, verbose = 3,
                   cv = kfold, scoring = 'roc_auc')

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_) 


best_model =gcv.best_estimator_
plt.figure(figsize = (50,30))
plot_tree(best_model, feature_names= X.columns,
          class_names =['0','1'], fontsize= 13)

plt.show() 

###Best Model Feture Importances ####

print(best_model.feature_importances_)
imps=best_model.feature_importances_
plt.barh(X.columns,imps)
plt.show()



############# Vehicle ##################
from sklearn.preprocessing import LabelEncoder,StandardScaler
vehicle= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec24\Vehicle Silhouettes\Vehicle.csv")
X=vehicle.drop('Class',axis=1)
y=vehicle['Class']

le=LabelEncoder() 
le_y=le.fit_transform(y)
print(le.classes_)

############# Grid Search ################

dtc = DecisionTreeClassifier(random_state = 2022)
params = {'max_depth': [2,3,4,5,None],
          'min_samples_split': [2,5,10],
          'min_samples_leaf': [1,5,10]}
kfold = StratifiedKFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)

gcv = GridSearchCV(dtc, param_grid = params, verbose = 3,
                   cv = kfold, scoring = 'neg_log_loss')

gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_) 


best_model =gcv.best_estimator_
plt.figure(figsize = (50,30))
plot_tree(best_model, feature_names= X.columns,
          class_names =le.classes_, fontsize= 13)

plt.show()

print(best_model.feature_importances_)
imps=best_model.feature_importances_
plt.barh(X.columns,imps)
plt.show()


### Heart Attack ####
from sklearn.preprocessing import LabelEncoder,StandardScaler
heart= pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec27\Heart Attack\heart.csv")
X=heart.drop('output',axis=1)
y=heart['output']



############# Grid Search ################

dtc = DecisionTreeClassifier(random_state = 2022)
params = {'max_depth': [2,3,4,5,None],
          'min_samples_split': [2,5,10],
          'min_samples_leaf': [1,5,10]}
kfold = StratifiedKFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)

gcv = GridSearchCV(dtc, param_grid = params, verbose = 3,
                   cv = kfold, scoring = 'roc_auc')

gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_) 



print(best_model.feature_importances_)
imps=best_model.feature_importances_
plt.barh(X.columns,imps)
plt.show()
