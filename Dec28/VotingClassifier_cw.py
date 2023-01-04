import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

image = pd.read_csv(r"C:\PG_DBDA\Machine Learning\Machine-Learning\Dec23\Image Segmentation\Image_Segmention.csv")

X = image.drop('Class', axis = 1)
y = image['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,
                                                    random_state=2022,
                                                    train_size=0.7)

lr = LogisticRegression()
nb = GaussianNB()
da = LinearDiscriminantAnalysis()
scaler = StandardScaler()
svm = SVC(kernel='linear', probability = True, random_state = 2022)
pipe = Pipeline([('STD', scaler), ('SVM', svm)])

kfold = StratifiedKFold(n_splits = 5, shuffle = True, 
                        random_state= 2022)
voting = VotingClassifier([('NB', nb),
                         ('SVML', pipe),
                         ('LR', lr),('LDA', da)],
                          voting='soft')
print(voting.get_params())

params = {'LR__C': np.linspace(0.001, 5, 5),
          'SVML__SVM__C': np.linspace(0.001, 5, 5)}

gcv = GridSearchCV(voting, param_grid = params, verbose = 3,
                   cv = kfold, scoring = 'neg_log_loss')

gcv.fit(X, y)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_) 














