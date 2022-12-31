import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

fp_df = pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\Association Datasets\Faceplate.csv", index_col = 0)
fp_df = fp_df.astype(bool)

# create frequent itemsets
itemsets = apriori(fp_df, min_support = 0.2, use_colnames = True)


# and convert into rules 
rules = association_rules(itemsets)


############################################################
############  cosmetics ===========================

fp_df = pd.read_csv(r"C:\Kaustubh Vaibhav\Machine Learning\Cases\Association Datasets\Cosmetics.csv", index_col = 0)
fp_df = fp_df.astype(bool)

# create frequent itemsets
itemsets = apriori(fp_df, min_support = 0.1, use_colnames = True)


# and convert into rules 
rules = association_rules(itemsets)
rules = rules.sort_values(by = ['lift','confidence'],
                          ascending = False)

rules[['antecedents','consequents',
       'support','confidence','lift']]



##############################################################
############### Groceries ################
from mlxtend.preprocessing import TransactionEncoder
groceries=[]
with open ("C:\Kaustubh Vaibhav\Machine Learning\Cases\Association Datasets\Groceries.csv","r") as f:groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
groceries_list
te = TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)
te_ary

fp_df = pd.DataFrame(te_ary,columns=te.columns_)

####Create Frequent itemsets
itemsets = apriori(fp_df, min_support = 0.005, use_colnames = True)


# and convert into rules 
rules = association_rules(itemsets,min_threshold=0.5)
rules = rules.sort_values(by = ['lift','confidence'],
                          ascending = False)

rules[['antecedents','consequents',
       'support','confidence','lift']]


###########################################################
################## DataSetA.csv ##################

from mlxtend.preprocessing import TransactionEncoder
dataset=[]
with open ("C:\Kaustubh Vaibhav\Machine Learning\Cases\Association Datasets\DataSetA.csv","r") as f:dataset = f.read()
dataset = dataset.split("\n")

dataset_list = []
for i in dataset :
    dataset_list.append(i.split(","))
dataset_list
te = TransactionEncoder()
te_ary = te.fit(dataset_list).transform(dataset_list)
te_ary

fp_df = pd.DataFrame(te_ary,columns=te.columns_)
fp_df=fp_df.iloc[:,1:] ##for dropping the last col. that contains aonly ''
####Create Frequent itemsets
itemsets = apriori(fp_df, min_support = 0.005, use_colnames = True)


# and convert into rules 
rules = association_rules(itemsets,min_threshold=0.5)
rules = rules.sort_values(by = ['lift','confidence'],
                          ascending = False)

rules[['antecedents','consequents',
       'support','confidence','lift']]                              









