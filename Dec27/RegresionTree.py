import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt



housing =pd.read_csv("Housing.csv")
dum_house =pd.get_dummies(housing,drop_first=True)
X =dum_house.drop('price',axis=1)
y=dum_house['price']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)


dtc = DecisionTreeRegressor(random_state= 2022,
                             max_depth = 2)

dtc.fit(X_train, y_train)


plt.figure(figsize = (25,10))
plot_tree(dtc, feature_names = X.columns,
           fontsize = 14,filled=True)

###Grid Search CV

dtc = DecisionTreeRegressor(random_state = 2022)
params = {'max_depth': [2,3,4,5,None],
          'min_samples_split': [2,5,10],
          'min_samples_leaf': [1,5,10]}
kfold =KFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)

gcv = GridSearchCV(dtc, param_grid = params, verbose = 3,
                   cv = kfold, scoring = 'r2')

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_) 


###Best Model Feature Importances plot ####
best_model=gcv.best_estimator_
print(best_model.feature_importances_)
imps=best_model.feature_importances_
plt.barh(X.columns,imps)
plt.title('Feature Importance Plot')
plt.show()


#####Concrete Strength ####

concrete =pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")
X =concrete.drop('Strength',axis=1)
y=concrete['Strength']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)


dtc = DecisionTreeRegressor(random_state= 2022,
                             max_depth = 2)

dtc.fit(X_train, y_train)


plt.figure(figsize = (25,10))
plot_tree(dtc, feature_names = X.columns,
           fontsize = 14,filled=True)

###Grid Search CV

dtc = DecisionTreeRegressor(random_state = 2022)
params = {'max_depth': [2,3,4,5,None],
          'min_samples_split': [2,5,10],
          'min_samples_leaf': [1,5,10]}
kfold =KFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)

gcv = GridSearchCV(dtc, param_grid = params, verbose = 3,
                   cv = kfold, scoring = 'r2')

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_) 


best_model =gcv.best_estimator_
plt.figure(figsize = (50,30))
plot_tree(best_model, feature_names= X.columns,
          class_names =['0','1'], fontsize= 13)

plt.show() 


###Best Model Feature Importances plot ####
best_model=gcv.best_estimator_
print(best_model.feature_importances_)
imps=best_model.feature_importances_
plt.barh(X.columns,imps)
plt.title('Feature Importance Plot')
plt.show()

###Sorted Plot
i_sorted=np.argsort(-imps)
n_sorted=X.columns[i_sorted]
imp_sort=imps[i_sorted]
plt.barh(n_sorted,imp_sort)
plt.title('Sorted Feautures Importances')
plt.show()


