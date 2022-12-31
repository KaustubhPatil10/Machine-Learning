import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os
os.chdir(r'C:\Training\Academy\Statistics (Python)\Association Rules datasets')


fp_df = pd.read_csv('Faceplate.csv',index_col=0)
fp_df = fp_df.astype(bool)

# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2, use_colnames=True)

# and convert into rules
rules = association_rules(itemsets)

########################### Cosmetics ##########################

cosmetics = pd.read_csv("Cosmetics.csv", index_col=0) 
cosmetics = cosmetics.astype(bool)

# create frequent itemsets
itemsets = apriori(cosmetics, min_support=0.1, use_colnames=True)

# and convert into rules
rules = association_rules(itemsets)
rules = rules.sort_values(by=['support','lift','confidence'],
                          ascending=False)

rules[['antecedents', 'consequents',
       'support', 'confidence', 'lift']]


######################### Groceries ###########################
from mlxtend.preprocessing import TransactionEncoder
groceries = []
with open("groceries.csv","r") as f:groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

te = TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)
te_ary

fp_df = pd.DataFrame(te_ary,columns=te.columns_)

# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.005, use_colnames=True)

# and convert into rules
rules = association_rules(itemsets,min_threshold=0.5)

###################### DatasetA ###############################
dataA = []
with open("DataSetA.csv","r") as f:dataA = f.read()
dataA = dataA.split("\n")

dataA_list = []
for i in dataA:
    dataA_list.append(i.split(","))

te = TransactionEncoder()
te_ary = te.fit(dataA_list).transform(dataA_list)
te_ary

fp_df = pd.DataFrame(te_ary,columns=te.columns_)
fp_df = fp_df.iloc[:,1:]


# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.005, use_colnames=True)

# and convert into rules
rules = association_rules(itemsets,min_threshold=0.5)

