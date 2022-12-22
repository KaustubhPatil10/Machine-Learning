import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

mowers = pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Datasets\RidingMowers.csv")


dum_mow = pd.get_dummies(mowers, drop_first = True)

X = dum_mow.drop('Response_Not Bought', axis = 1)
y = dum_mow['Response_Not Bought']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify =y, random_state=2022,
                                                    train_size=0.7)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.scatterplot(data= mowers, x= 'Income', y = 'Lot_Size',
               hue = 'Response')
plt.show()

# for finding best accuracy score
#Loop
acc =[]
Ks = [1,3,5,7,9,11,13,15]
for i in Ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred))
    
i_max = np.argmax(acc)
best_k = Ks[i_max] 
print("Best n_neighbors =", best_k)   

#---------------------------------------------------------
# using roc......
    
from sklearn.metrics import roc_curve, roc_auc_score
##loop
acc=[]
Ks = [1,3,5,7,9,11,13,15]
for i in Ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_prob = knn.predict_proba(X_test)[:,1]
    acc.append(roc_auc_score(y_test, y_pred_prob))

i_max = np.argmax(acc)
best_k = Ks[i_max] 
print("Best n_neighbors =", best_k) 
print(roc_auc_score(y_test, y_pred_prob))

#=========================================
#log Loss

##loop
from sklearn.metrics import log_loss

# Evaluation of Model 2 probe
#print(log_loss(comp_prob['y_test'], comp_prob['yprob_2']))


acc=[]
Ks = [1,3,5,7,9,11,13,15]
for i in Ks:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_prob = knn.predict_proba(X_test)[:,1]
    acc.append(-log_loss(y_test, y_pred_prob))


i_max = np.argmax(acc)
best_k = Ks[i_max] 
print("Best n_neighbors =", best_k) 
print(log_loss(y_test, y_pred_prob))
                 