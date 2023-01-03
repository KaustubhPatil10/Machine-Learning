import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cars93=pd.read_csv("Cars93.csv")
sns.boxplot(y='Price', data=cars93)
plt.show()


q1=cars93['Price'].quantile(q=0.25)
q3=cars93['Price'].quantile(q=0.75)

IQR=cars93['Price'].quantile(q=0.75)-cars93['Price'].quantile(q=0.25)

lim_IQR=1.5*IQR

upper_IQR=q3+lim_IQR
lower_IQR=q1-lim_IQR

outliers_df = cars93[(cars93['Price'] > upper_IQR) | (cars93['Price'] < lower_IQR)]
outliers_df['Price']

############# Function #################
df=cars93
column='Price'
def detect_outliers(df,column):
    q1=cars93[column].quantile(q=0.25)
    q3=cars93[column].quantile(q=0.75)

    IQR=q3-q1

    lim_IQR=1.5*IQR

    upper_IQR=q3+lim_IQR
    lower_IQR=q1-lim_IQR

    outliers_df = cars93[(cars93[column] > upper_IQR) | (cars93[column] < lower_IQR)]
    return outliers_df[column].to_list();

detect_outliers(cars93, 'MPG.city')

##############################################################

housing = pd.read_csv("Housing.csv")
sns.boxplot(y='price', data=housing)
plt.show()

# function #######
# df=housing
# column='price'
def detect_outliers(df, column):
    
    q1=housing[column].quantile(q=0.25)
    q3=housing[column].quantile(q=0.75)

    IQR=q3-q1

    lim_IQR=1.5*IQR

    upper_IQR=q3+lim_IQR
    lower_IQR=q1-lim_IQR

    outliers_df = housing[(housing[column] > upper_IQR) | (housing[column] < lower_IQR)]
    return outliers_df[column].to_list();

gh = detect_outliers(housing, 'price')
print(gh)










