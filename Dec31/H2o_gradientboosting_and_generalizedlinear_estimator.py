import h2o
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
h2o.init()

df =h2o.import_file(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",
                    destination_frame="brupt")

print(df.col_names)
all_cols = df.col_names
y = 'D'
X = all_cols[3:]

df['D'] = df['D'].asfactor()
print(df['D'].levels())

train, test = df.split_frame(ratios = [.7], seed = 2022,
                             destination_frames = ['train','test'])
print(df.shape) 
print(train.shape)
print(test.shape)


h2o_gf = H2OGradientBoostingEstimator(seed = 2022)
h2o_gf.train(x = X, y = y, training_frame = train,
             validation_frame = test,
             model_id = "h2o_gf_cancer")
print(h2o_gf.auc(valid = True))
print(h2o_gf.confusion_matrix())

y_pred = h2o_gf.predict(test_data = test)
y_pred_df = y_pred.as_data_frame()

print(h2o_gf.logloss( valid = True))


####### H2OGeneralizedLinearEstimator ####

import h2o
import pandas as pd
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
h2o.init()

df =h2o.import_file(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",
                    destination_frame="brupt")

print(df.col_names)
all_cols = df.col_names
y = 'D'
X = all_cols[3:]

df['D'] = df['D'].asfactor()
print(df['D'].levels())

train, test = df.split_frame(ratios = [.7], seed = 2022,
                             destination_frames = ['train','test'])
print(df.shape) 
print(train.shape)
print(test.shape)


h2o_lr = H2OGeneralizedLinearEstimator(seed = 2022)
h2o_lr.train(x = X, y = y, training_frame = train,
             validation_frame = test,
             model_id = "h2o_lr_cancer")
print(h2o_lr.auc(valid = True))
print(h2o_lr.confusion_matrix())

y_pred = h2o_lr.predict(test_data = test)
y_pred_df = y_pred.as_data_frame()

print(h2o_lr.logloss(valid = True))

###### H2OXGBoostEstimator ##### 
# the code wont run shows error.
import h2o
import pandas as pd
from h2o.estimators.xgboost import H2OXGBoostEstimator
h2o.init()

df =h2o.import_file(r"C:\Kaustubh Vaibhav\Machine Learning\Dec23\Bankruptcy\Bankruptcy.csv",
                    destination_frame="brupt")

print(df.col_names)
all_cols = df.col_names
y = 'D'
X = all_cols[3:]

df['D'] = df['D'].asfactor()
print(df['D'].levels())

train, test = df.split_frame(ratios = [.7], seed = 2022,
                             destination_frames = ['train','test'])
print(df.shape) 
print(train.shape)
print(test.shape)


h2o_xgb = H2OXGBoostEstimator(seed = 2022)
h2o_xgb.train(x = X, y = y, training_frame = train,
             validation_frame = test,
             model_id = "h2o_xgb_cancer")
print(h2o_xgb.auc(valid = True))
print(h2o_xgb.confusion_matrix())

y_pred = h2o_xgb.predict(test_data = test)
y_pred_df = y_pred.as_data_frame()

print(h2o_xgb.logloss( valid = True))
