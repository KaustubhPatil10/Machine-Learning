import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv(r"C:\Kaustubh Vaibhav\Advance Analystics\Datasets\AusGas.csv")
df

df.plot.line(x = 'Month', y = 'GasProd')
plt.show()
y=df['GasProd']
y_train=y[:464]
y_test=y[464:]
################ Auto REgression #####################
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


model=AutoReg(y_train,lags=3)
model_fit=model.fit()
print('Coefficents : %s' %model_fit.params)

#Make Predictions
predictions=model_fit.predict(start=len(y_train),
                      end=len(y_train)+len(y_test)-1,
                      dynamic=False)

error=mean_squared_error(y_test, predictions)

print('Test RMSE:%.3f '%sqrt(error))

### plot results
plt.plot(y_test,label='Test')
plt.plot(predictions,color='red',label='predicated')
plt.legend(loc='best')
plt.show()

################## MA ##########################
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


model=ARIMA(y_train,order=(0,0,2))
model_fit=model.fit()
print('Coefficents : %s' %model_fit.params)
 
#Make Predictions
predictions=model_fit.predict(start=len(y_train),
                      end=len(y)-1,
                      dynamic=False)

error=mean_squared_error(y_test, predictions)

print('Test RMSE:%.3f '%sqrt(error))

### plot results
plt.plot(y_test,label='Test')
plt.plot(predictions,color='red',label='predicated')
plt.legend(loc='best')
plt.show()

##########Auto ARIMA ##############
from pmdarima.arima import auto_arima
model=auto_arima(y_train, trace=True,
                 error_action='ignore',
                 suppress_warnings=True)

forecast=model.predict(n_periods=len(y_test))

##plot the predictions for Validation set
plt.plot(y_train,label='Train',color="blue")
plt.plot(y_test,label='valid',color="pink")
plt.plot(forecast,label='Predictions',color='purple')
plt.show()

###Plot Results 
plt.plot(y_test,label='Test')
plt.plot(predictions,color='red',label='Forecast')
plt.legend(loc='best')
plt.show()



######################### SARIMA ######################
from pmdarima.arima import auto_arima
model=auto_arima(y_train, trace=True,
                 error_action='ignore',
                 suppress_warnings=True,seasonal=True,m=12)

forecast=model.predict(n_periods=len(y_test))

##plot the predictions for Validation set
plt.plot(y_train,label='Train',color="blue")
plt.plot(y_test,label='valid',color="pink")
plt.plot(forecast,label='Predictions',color='purple')
plt.show()

###Plot Results 
plt.plot(y_test,label='Test')
plt.plot(predictions,color='red',label='Forecast')
plt.legend(loc='best')
plt.show()
