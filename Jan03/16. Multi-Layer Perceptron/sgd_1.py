import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
tinydata = pd.read_csv("tinydata.csv", index_col=0)
dum_tiny = pd.get_dummies(tinydata, drop_first=True)
sgd = SGDClassifier(loss='log_loss', random_state=2022)
X = dum_tiny.drop('Acceptance_like', axis=1)
y = dum_tiny['Acceptance_like']

sgd.fit(X, y)
print(sgd.coef_)
print(sgd.intercept_)

tst_tiny = np.array([[0.3,0.4]])
y_pred = sgd.predict(tst_tiny)
print(y_pred)
y_pred_prob = sgd.predict_proba(tst_tiny)
print(y_pred_prob)


##################### MLPClassifier ###################
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(2,),
                    activation='logistic',
                    random_state=2022)

mlp.fit(X, y)
print(mlp.coefs_)
print(mlp.intercepts_)

tst_tiny = np.array([[0.3,0.4]])
y_pred = mlp.predict(tst_tiny)
print(y_pred)
y_pred_prob = mlp.predict_proba(tst_tiny)
print(y_pred_prob)