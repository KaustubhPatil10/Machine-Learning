import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


hr = pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first= True )
X = dum_hr.drop('left', axis = 1)
y = dum_hr['left']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

dtc=DecisionTreeClassifier(random_state=2022)
scaler=StandardScaler()
svm=SVC(kernel='linear',probability=True,random_state=2022)
pipe_svm =Pipeline([('STD',scaler),('SVM',svm)])
da=LinearDiscriminantAnalysis()
voting=VotingClassifier([('TREE',dtc),
                         ('SVM_P',pipe_svm),
                         ('LDA',da)],voting='soft')

voting.fit(X_train,y_train)
y_pred=voting.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

#Separetely evaluating clssifiers
dtc.fit(X_train,y_train)
y_pred_prob=voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))
roc_dtc=roc_auc_score(y_test, y_pred_prob)

pipe_svm.fit(X_train,y_train)
y_pred_prob=voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))
roc_svm=roc_auc_score(y_test, y_pred_prob)

da.fit(X_train,y_train)
y_pred_prob=voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))
roc_da=roc_auc_score(y_test, y_pred_prob)

print(roc_dtc,roc_svm,roc_da)

###Weighted
voting=VotingClassifier([('TREE',dtc),
                         ('SVM_P',pipe_svm),
                         ('LDA',da)],voting='soft',
                        weights=[roc_dtc,roc_svm,roc_da])



#####Bankruptcy####
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression 


brupt=pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",index_col=0)

X=brupt.drop(['D','YR'],axis=1)
y=brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,train_size=0.7)



scaler=StandardScaler()
svm=SVC(probability = True, random_state=2022)
pipe =Pipeline([('STD',scaler),('SVM',svm)])
pipe.fit(X_train, y_train)
y_pred_prob=pipe.predict_proba(X_test)[:,1]
roc_svm=roc_auc_score(y_test, y_pred_prob)

lr =LogisticRegression()
lr.fit(X_train,y_train)
y_pred_prob=lr.predict_proba(X_test)[:,1]
roc_lr=roc_auc_score(y_test, y_pred_prob)
                     
dtc=DecisionTreeClassifier(random_state=2022)
dtc.fit(X_train,y_train)
y_pred_prob=dtc.predict_proba(X_test)[:,1]
roc_dtc=roc_auc_score(y_test, y_pred_prob)
                     

# W/o Weights
voting=VotingClassifier([('TREE',dtc),
                         ('SVM_P',pipe),
                         ('LR',lr)],voting='soft')
voting.fit(X_train,y_train)
y_pred_prob=voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))


##With Weight

voting=VotingClassifier([('TREE',dtc),
                         ('SVM_P',pipe_svm),
                         ('LR',lr)],voting='soft',
                        weights=[roc_svm,roc_lr,roc_dtc])
voting.fit(X_train,y_train)
y_pred_prob=voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))


###Grid Search Cv #####
voting=VotingClassifier([('TREE',dtc),
                         ('SVM_P',pipe),
                         ('LR',lr)],voting='soft')
print(voting.get_params())
kfold=StratifiedKFold(n_splits=5,random_state=2022,shuffle=True)

params={}