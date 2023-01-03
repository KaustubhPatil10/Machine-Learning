import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


mowers=pd.read_csv("RidingMowers.csv")
X=mowers.drop('Response',axis=1)

y=mowers['Response']

clf=IsolationForest(contamination=0.05,
                    random_state=2022)
clf.fit(X)
pred_outliers=clf.predict(X)
mowers['outliers']=pred_outliers
#mowers['outliers']=mowers['outliers'].astype(str)

sns.scatterplot(x='Income',y='Lot_Size',
                hue='outliers',data=mowers)
plt.show()


########## Milk.CSV ####
milk=pd.read_csv("Milk.csv",index_col=0)
X=milk.drop('protein',axis=1)

y=milk['protein']

clf=IsolationForest(contamination=0.05,
                    random_state=2022)
clf.fit(X)
pred_outliers=clf.predict(X)
milk['outliers']=pred_outliers
milk['outliers']=milk['outliers'].astype(str)

sns.scatterplot(x='protein',y='water',
                hue='outliers',data=milk)
plt.show()