import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import h2o



h2o.init()

train =h2o.import_file(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\train.csv",
                    destination_frame="train")

test =h2o.import_file(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\test.csv",
                    destination_frame="test")
print(train.col_names)
print(test.col_names)
all_cols =train.col_names
y = 'TARGET'
X = all_cols[1:-1]

train['TARGET'] = train['TARGET'].asfactor()
print(train['TARGET'].levels())

#### H2O Generalized Linear Estimator =============
h2o_gbm = H2OGeneralizedLinearEstimator(seed = 2022)
h2o_gbm.train(x = X, y = y, training_frame = train,
             validation_frame = test,
             model_id = "h2o__gf_Santander")
print(h2o_gbm.auc(valid = True))
print(h2o_gbm.confusion_matrix())

y_pred = h2o_gbm.predict(test_data = test)
y_pred_df = y_pred.as_data_frame()


submit = pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\sample_submission.csv")
submit['TARGET']=y_pred_df['p1']
submit.to_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Dec31\kaustubh_H2o_kaggle_GLE_.csv",index=False)

