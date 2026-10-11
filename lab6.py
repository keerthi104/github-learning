
%pip install mlxtend

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Read the CSV file
filename = 'Assignment-1_Data.csv'
df = pd.read_csv(filename, encoding='unicode_escape')

print("Successfully read the CSV file.")

# Data Cleaning and Preprocessing

# Remove rows where CustomerID is missing
df.dropna(axis=0, subset=['CustomerID'], inplace=True)

# Convert BillNo to string
df['BillNo'] = df['BillNo'].astype('str')

# Remove credit/cancelled transactions
df = df[~df['BillNo'].str.contains('C')]

# Remove leading/trailing spaces from item names
df['Itemname'] = df['Itemname'].str.strip()

print("\nCleaned DataFrame preview:\n", df.head())

# Market Basket Analysis

# Group by BillNo and create a list of items for each transaction
transactions = df.groupby('BillNo')['Itemname'].apply(list).tolist()

# Convert transaction data into one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("\nOne-hot encoded DataFrame preview:\n", df_encoded.head())

# Apply Apriori algorithm
# min_support=0.01 means itemsets appearing in at least 1% of transactions are considered frequent

frequent_itemsets = apriori(
    df_encoded,
    min_support=0.01,
    use_colnames=True
)

print("\nFrequent Itemsets:\n", frequent_itemsets)

# Generate association rules
# min_threshold=0.5 means minimum confidence is 50%

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print("\nAssociation Rules:\n", rules)