#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary modules
import random
import csv
import os
import itertools
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
import time


# In[ ]:


## Part 1: Data Preparation


# In[2]:


# List of items seen in supermarkets
items = ['Milk', 'Cheese', 'Yogurt', 'Chicken', 'Beef', 'Bread', 'Chips', 'Cookies', 'Soda', 'Juice']

# Create a transaction
def transaction(items):
    return random.sample(items, random.randint(1, len(items)))

# Create a database with the transactions
def database(items, transactions, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for _ in range(transactions):
            writer.writerow(transaction(items))


# In[3]:


# For the initial database
database(items, 20, 'database1.csv')

# Creating 4 additional, different databases
for i in range(2, 6):
    database(items, 20, f'database{i}.csv')

# Test creation of database
print("Databases have been created successfully.")


# In[4]:


## Part 2: Algorithm Implementation and Comparison


# In[6]:


### Brute Force Algorithm


# In[5]:


# Read transactions from a CSV file
def read_transactions(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        return list(reader)

# Count the frequency of an itemset in the transactions
def count_frequency(itemset, transactions):
    return sum(1 for transaction in transactions if set(itemset).issubset(transaction))

# Generate all possible itemsets of a certain size
def generate_itemsets(items, size):
    return list(itertools.combinations(items, size))

# Identify frequent itemsets using the brute force method
def find_frequent_itemsets(items, transactions, min_frequency):
    size = 1
    while True:
        itemsets = generate_itemsets(items, size)
        frequent_itemsets = [itemset for itemset in itemsets if count_frequency(itemset, transactions) >= min_frequency]
        if not frequent_itemsets:
            break
        size += 1
    return frequent_itemsets

# Generate association rules from the frequent itemsets
def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
         for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                consequent = tuple(item for item in itemset if item not in antecedent)
                confidence = count_frequency(itemset, transactions) / count_frequency(antecedent, transactions)
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent))
    return rules

# Read the transactions from the CSV file
transactions = read_transactions('database1.csv')

# Find the frequent itemsets
frequent_itemsets = find_frequent_itemsets(items, transactions, min_frequency=10)

# Generate the association rules
rules = generate_association_rules(frequent_itemsets, min_confidence=0.5)

print("Association rules generated successfully.")


# In[7]:


### Brute Force Algorithm


# In[10]:


# Read transactions from a CSV file
def read_transactions(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        return list(reader)

# Count the frequency of an itemset in the transactions
def count_frequency(itemset, transactions):
    return sum(1 for transaction in transactions if set(itemset).issubset(transaction))

# Generate all possible itemsets of a certain size
def generate_itemsets(items, size):
    return list(itertools.combinations(items, size))

# Identify frequent itemsets using the brute force method
def find_frequent_itemsets(items, transactions, min_frequency):
    size = 1
    while True:
        itemsets = generate_itemsets(items, size)
        frequent_itemsets = [itemset for itemset in itemsets if count_frequency(itemset, transactions) >= min_frequency]
        if not frequent_itemsets:
            break
        size += 1
    return frequent_itemsets

# Generate association rules from the frequent itemsets
def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                consequent = tuple(item for item in itemset if item not in antecedent)
                confidence = count_frequency(itemset, transactions) / count_frequency(antecedent, transactions)
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent))
    return rules

# Read the transactions from the CSV file
transactions = read_transactions('database1.csv')

# Find the frequent itemsets
frequent_itemsets = find_frequent_itemsets(items, transactions, min_frequency=10)

# Generate the association rules
rules = generate_association_rules(frequent_itemsets, min_confidence=0.5)

print("Association rules generated successfully.")


# In[11]:


### Apriori and FP-Growth


# In[12]:


# Read the databases
def read_all_databases(database_filenames):
    all_transactions = []
    for filename in database_filenames:
        transactions = read_transactions(filename)
        all_transactions.extend(transactions)
    return all_transactions

# List of database filenames
database_filenames = ['database1.csv', 'database2.csv', 'database3.csv', 'database4.csv', 'database5.csv']

# Read all the transactions from all databases
all_transactions = read_all_databases(database_filenames)

# Convert the transactions into a DataFrame
te = TransactionEncoder()
te_ary = te.fit(all_transactions).transform(all_transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Minimum support
min_support = 0.1

# Minimum confidence
min_confidence = 0.5

# Run the brute force algorithm and measure the time
start = time.time()
frequent_itemsets_brute_force = find_frequent_itemsets(items, all_transactions, min_support)
rules_brute_force = generate_association_rules(frequent_itemsets_brute_force, min_confidence)
end = time.time()
print(f"Brute force method took {end - start} seconds.")

# Run the Apriori algorithm and measure the time
start = time.time()
frequent_itemsets_apriori = apriori(df, min_support=min_support, use_colnames=True)
end = time.time()
print(f"Apriori algorithm took {end - start} seconds.")

# Run the FP-Growth algorithm and measure the time
start = time.time()
frequent_itemsets_fpgrowth = fpgrowth(df, min_support=min_support, use_colnames=True)
end = time.time()
print(f"FP-Growth algorithm took {end - start} seconds.")

# Compare the results
print("Brute force method found {} frequent itemsets.".format(len(frequent_itemsets_brute_force)))
print("Apriori algorithm found {} frequent itemsets.".format(len(frequent_itemsets_apriori)))
print("FP-Growth algorithm found {} frequent itemsets.".format(len(frequent_itemsets_fpgrowth)))


# In[13]:


## Performance Analysis & Conclusion


# In[15]:


##In conclusion, the project successfully implemented and compared three different algorithms for frequent itemset mining and association rule generation: brute force, Apriori, and FP-Growth. The results showed that both Apriori and FP-Growth produced the same number of frequent itemsets, while the brute force method did not find any. In terms of performance, FP-Growth was the fastest, followed by Apriori, and then the brute force method. Overall, this project demonstrated the practical application and comparison of different algorithms in data mining.


# In[ ]:




