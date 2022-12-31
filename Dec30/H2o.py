import h2o
import pandas as pd
from h2o.estimators.random_forest import H2ORandomForestEstimator
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


h2o_rf = H2ORandomForestEstimator(seed = 2022)
h2o_rf.train(x = X, y = y, training_frame = train,
             validation_frame = test,
             model_id = "h2o_rf_cancer")
print(h2o_rf.auc())
print(h2o_rf.confusion_matrix())

y_pred = h2o_rf.predict(test_data = test)
y_pred_df = y_pred.as_data_frame()

print(h2o_rf.logloss(train = False, valid = True,
                     xval = False))

