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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

brupt=pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)
clf = GradientBoostingClassifier(random_state = 2022)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = clf.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_pred_prob))

#======================= grid search cv ============
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state= 2022)
params = {'n_estimators': [50, 100, 150],
          'max_depth': [1, 2, 3, 4],
          'learning_rate': [0.01, 0.1, 0.15, 0.2, 0.3]}
clf = GradientBoostingClassifier(random_state = 2022)
print(clf.get_params())
gcv = GridSearchCV(clf, param_grid = params, cv = kfold, 
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

########################## XGB ############################

from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state = 2022)
print(xgb_model.get_params())
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state= 2022)
params = {'n_estimators': [50, 100, 150],
          'max_depth': [1, 2, 3, 4],
          'learning_rate': [0.01, 0.1, 0.15, 0.2, 0.3]}
clf = GradientBoostingClassifier(random_state = 2022)
print(clf.get_params())
gcv = GridSearchCV(clf, param_grid = params, cv = kfold, 
                   scoring = 'roc_auc', verbose = 3)

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##################################################################
########################## image segmentation ======================
from sklearn.preprocessing import LabelEncoder
image = pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec23\Image Segmentation\Image_Segmention.csv")

X = image.drop('Class', axis = 1)
y = image['Class']


le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

###### XGB Classifier =========================
xgb_model = XGBClassifier(random_state = 2022)
print(xgb_model.get_params())
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state= 2022)
params = {'n_estimators': [50, 100, 150],
          'max_depth': [1, 2, 3, 4],
          'learning_rate': [0.01, 0.1, 0.15, 0.2, 0.3]}
clf = GradientBoostingClassifier(random_state = 2022)
print(clf.get_params())
gcv = GridSearchCV(xgb_model, param_grid = params, cv = kfold, 
                   scoring = 'neg_log_loss', verbose = 3)

gcv.fit(X, le_y)
print(gcv.best_params_)
print(gcv.best_score_)

#######################################################################
###################### concrete strength #######################

concrete = pd.read_csv(r"C:\PG_DBDA\Advance analytics classwork\Advanced_Analytics\Cases\Concrete Strength\Concrete_Data.csv")

X = concrete.drop('Strength', axis = 1)
y = concrete['Strength']


###### XGB Regressor =========================
from xgboost import XGBRegressor
xgb_model = XGBRegressor(random_state = 2022)
print(xgb_model.get_params())
kfold = KFold(n_splits = 5, shuffle = True, random_state= 2022)
params = {'n_estimators': [50, 100, 150],
          'max_depth': [1, 2, 3, 4],
          'learning_rate': [0.01, 0.1, 0.15, 0.2, 0.3]}
clf = GradientBoostingRegressor(random_state = 2022)
print(clf.get_params())
gcv = GridSearchCV(xgb_model, param_grid = params, cv = kfold, 
                   scoring = 'r2', verbose = 3)

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






