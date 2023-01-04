import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score

brupt=pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

rf = RandomForestClassifier(random_state = 2022)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = rf.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))

######### Grid Search CV ===============================

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state= 2022)
params = {'max_features': np.arange(3,15)}

gcv = GridSearchCV(rf, param_grid = params, cv = kfold, 
                   scoring = 'roc_auc', verbose = 3)

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################ Feature importance plot ================
import matplotlib.pyplot as plt
best_model =gcv.best_estimator_
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.show()

i_sorted = np.argsort(-imps)
n_sorted = X.columns[i_sorted]
imp_sort = imps[i_sorted]
plt.barh(n_sorted, imp_sort)
plt.title(" Sorted Features Importances")
plt.show()

###############################################################
########## hr data #####################################

hr = pd.read_csv(r"C:\PG_DBDA\Advance analytics classwork\Advanced_Analytics\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first= True )
X = dum_hr.drop('left', axis = 1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)


rf = RandomForestClassifier(random_state = 2022)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = rf.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))

#======================= grid search cv ============
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state= 2022)
params = {'max_features': np.arange(3,15)}

gcv = GridSearchCV(rf, param_grid = params, cv = kfold, 
                   scoring = 'roc_auc', verbose = 3)

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

# feature importance plot =================
best_model =gcv.best_estimator_
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.show()

i_sorted = np.argsort(-imps)
n_sorted = X.columns[i_sorted]
imp_sort = imps[i_sorted]
plt.barh(n_sorted, imp_sort)
plt.title(" Sorted Features Importances")
plt.show()

##############################################################
################### image segmentation ####################

image = pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec23\Image Segmentation\Image_Segmention.csv")

X = image.drop('Class', axis = 1)
y = image['Class']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)



rf = RandomForestClassifier(random_state = 2022)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))


#======================= grid search cv ============
kfold = StratifiedKFoldX_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)(n_splits = 5, shuffle = True, random_state= 2022)
params = {'max_features': np.arange(3,15)}

gcv = GridSearchCV(rf, param_grid = params, cv = kfold, 
                   scoring = 'neg_log_loss', verbose = 3)

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

# feature importance plot =================
best_model =gcv.best_estimator_
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.show()

i_sorted = np.argsort(-imps)
n_sorted = X.columns[i_sorted]
imp_sort = imps[i_sorted]
plt.barh(n_sorted, imp_sort)
plt.title(" Sorted Features Importances")
plt.show()

#####################################################################
#################### insurance ###################################

insurance = pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec23\Medical Cost Personal\insurance.csv")
insur = pd.get_dummies(insurance, drop_first = True)
X = insur.drop('charges', axis = 1)
y = insur['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(r2_score(y_test, y_pred))
#======================= grid search cv ============
kfold = KFold(n_splits = 5, shuffle = True, random_state= 2022)
params = {'max_features': np.arange(2,9)}

gcv = GridSearchCV(rf, param_grid = params, cv = kfold, 
                   scoring = 'r2', verbose = 3)

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)













