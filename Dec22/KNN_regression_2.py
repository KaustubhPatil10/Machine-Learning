import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

concrete = pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\Concrete Strength\Concrete_Data.csv")

X = concrete.drop('Strength', axis =1)
y = concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022,
                                                    train_size=0.7)

knn = KNeighborsRegressor(n_neighbors = 1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


#r2_score before scaling.
acc =[]
Ks = np.arange(1,16)
for i in Ks:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc.append(r2_score(y_test, y_pred))

print(acc)
i_max = np.argmax(acc)
best_k = Ks[i_max] 
print("Best n_neighbors =", best_k) 
    
print("Best Score= ",acc[i_max])


#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_trn_scl = scaler.transform(X_train)
X_tst_scl = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_trn_scl, y_train)

y_pred = knn.predict(X_tst_scl)
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

#r2_score after scaling
acc =[]
Ks = np.arange(1,16)
for i in Ks:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_trn_scl, y_train)
    y_pred = knn.predict(X_tst_scl)
    acc.append(r2_score(y_test, y_pred))
    
print(acc)
i_max = np.argmax(acc)
best_k = Ks[i_max] 
print("Best n_neighbors =", best_k) 
    
print("Best Score= ",acc[i_max])    