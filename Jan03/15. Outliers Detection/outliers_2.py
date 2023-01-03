import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

mowers = pd.read_csv("RidingMowers.csv")
X = mowers.drop('Response', axis=1)
y = mowers['Response']

clf = IsolationForest(contamination=0.05,
                            random_state=2022)

clf.fit(X)
pred_outliers = clf.predict(X)
mowers['outliers'] = pred_outliers
mowers['outliers'] = mowers['outliers'].astype(str)

sns.scatterplot(x='Income',y='Lot_Size',
                hue='outliers', data=mowers)
plt.show()

################## milk.csv #####################

milk = pd.read_csv("milk.csv", index_col=0)


clf = IsolationForest(contamination=0.05,
                            random_state=2022)

clf.fit(milk)
pred_outliers = clf.predict(milk)
milk['outliers'] = pred_outliers

