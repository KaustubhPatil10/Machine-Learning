import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cars93 = pd.read_csv("Cars93.csv")
sns.boxplot(y='Price', data=cars93)
plt.show()

q1 = cars93['Price'].quantile(q=0.25)
q3 = cars93['Price'].quantile(q=0.75)
iqr = q3 - q1
print("Inter-Quartile Range =",iqr)

lim_iqr = 1.5*iqr

upper_iqr = q3 + lim_iqr
lower_iqr = q1 - lim_iqr

outliers_df = cars93[(cars93['Price'] > upper_iqr) | (cars93['Price'] < lower_iqr)]  
outliers_df['Price']

############### Function ###################


def detect_outliers(df, column):
    q1 = df[column].quantile(q=0.25)
    q3 = df[column].quantile(q=0.75)
    iqr = q3 - q1

    lim_iqr = 1.5*iqr

    upper_iqr = q3 + lim_iqr
    lower_iqr = q1 - lim_iqr

    outliers_df = df[(df[column] > upper_iqr) | (df[column] < lower_iqr)]  
    return outliers_df[column].to_list();

detect_outliers(cars93, 'MPG.city')

sns.boxplot(y='MPG.city', data=cars93)
plt.show()

housing = pd.read_csv("Housing.csv")

detect_outliers(housing, 'price')

sns.boxplot(y='price', data=housing)
sns.swarmplot(y=detect_outliers(housing, 'price'), data=housing)
plt.show()



